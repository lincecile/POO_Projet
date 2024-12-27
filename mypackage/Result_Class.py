import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Result:
    """Classe pour analyser et visualiser les résultats du backtest."""

    def __init__(self, data: pd.DataFrame, positions: pd.DataFrame, trades: pd.DataFrame):
        self.data = data
        self.positions = positions
        self.trades = trades
        self.returns, self.returns_no_cost = self._calculate_returns()
        
        statistics = self._calculate_statistics()
        self.statistics = {key: series['portfolio'] for key, series in statistics.items()}
        self.statistics_each_asset = {key: series.drop('portfolio') for key, series in statistics.items()}

    def _calculate_returns(self) -> pd.DataFrame:
        """Calcul des rendements de la stratégie pour chaque actif"""
        returns = pd.DataFrame(index=self.positions.index)
        returns_without_cost = pd.DataFrame(index=self.positions.index)

        for asset in self.positions.columns:
            price_returns = self.data[asset].pct_change()
            strategy_returns = price_returns * self.positions[asset].shift(1)
            returns_without_cost[asset] = strategy_returns.fillna(0)

            # Prise en compte des coûts si des trades existent pour cet actif
            if not self.trades.empty:
                asset_trades = self.trades[self.trades['asset'] == asset]
                
                if not asset_trades.empty:
                    strategy_returns.loc[asset_trades.index] -= asset_trades['cost']
            returns[asset] = strategy_returns.fillna(0)
        
        # Ajout d'une colonne pour le rendement total du portefeuille
        returns['portfolio'] = returns.mean(axis=1)  # Moyenne simple, peut être modifiée pour pondération personnalisée
        returns_without_cost['portfolio'] = returns_without_cost.mean(axis=1)  # Moyenne simple, peut être modifiée pour pondération personnalisée

        return returns, returns_without_cost
    
    def _calculate_statistics(self) -> dict:
        """Calcul des statistiques de performance de la stratégie"""
        
        total_return = (1 + self.returns).prod() - 1                                    # Performance total
        annual_return = ((1 + total_return) ** (252 / len(self.returns))) - 1 if len(self.returns) > 0 else pd.Series(0, index=self.returns.columns)        # Performance annualisée
        volatility = self.returns.std() * np.sqrt(252)                                  # Volatilité annualisée

        # Ratio de Sharpe
        sharpe_ratio = pd.Series([an_return / vol if vol != 0 else 0 for an_return, vol in zip(annual_return, volatility)], index=annual_return.index)
        
        # Maximum Drawdown
        cumulative_returns = (1 + self.returns).cumprod()
        drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
        max_drawdown = drawdowns.min()

        # Ratio de Sortino
        downside_returns = self.returns[self.returns < 0]
        downside_deviation = self.returns[self.returns < 0].apply(lambda col: np.sqrt((col ** 2).mean()) * np.sqrt(252) if not col.empty else 0, axis=0)
        sortino_ratio = pd.Series([an_return / dev if dev != 0 else np.nan for an_return, dev in zip(annual_return, downside_deviation)], index=annual_return.index)
        
        # VaR (Value at Risk) à 95%
        var_95 = self.returns.apply(lambda col: col.quantile(0.05))

        # CVaR (Conditional Value at Risk) à 95%
        cvar_95 = self.returns.apply(lambda col: col[col <= col.quantile(0.05)].mean() if len(col[col <= col.quantile(0.05)]) > 0 else 0)
        
        # Gain/Pertes moyen (Profit/Loss Ratio)
        avg_gain = self.returns[self.returns > 0].mean() if len(self.returns[self.returns > 0]) > 0 else 0
        avg_loss = self.returns[self.returns < 0].mean() if len(self.returns[self.returns < 0]) > 0 else 0
        profit_loss_ratio = pd.Series([abs(avg_g / avg_l) if avg_l != 0 else np.nan for avg_g, avg_l in zip(avg_gain, avg_loss)], index=avg_gain.index)
        
        # Pourcentage de trades gagnants
        win_rate = self.returns.apply(lambda col: col[col > 0].count() / col[col != 0].count() if col[col != 0].count() > 0 else 0)

        # Facteur de profitabilité
        profit_factor = self.returns.apply(lambda col: abs(col[col > 0].sum() / col[col < 0].sum()) if col[col < 0].sum() != 0 else 1)
        
        # Nombre de trades
        if not self.trades.empty:
            nb_trade = self.trades['asset'].value_counts()
            nb_trade['portfolio'] = nb_trade.sum()
        else:
            nb_trade = pd.Series(0, index=self.returns.columns)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'profit_factor': profit_factor,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino_ratio,
            'VaR_95%': var_95,
            'CVaR_95%': cvar_95,
            'Profit/Loss_Ratio': profit_loss_ratio,
            'num_trades': nb_trade,
            'win_rate': win_rate  
        }

    def plot(self, name_strat: str, backend: str = 'matplotlib', include_costs: bool = True):
        """Visualise les résultats du backtest."""

        if backend not in ['matplotlib', 'seaborn', 'plotly']:
            raise ValueError(f"Backend invalide. Valeurs possibles : {', '.join(['matplotlib', 'seaborn', 'plotly'])}")
        
        returns_to_use = self.returns if include_costs else self.returns_no_cost
        cumulative_returns = (1 + returns_to_use).cumprod()
        
        titre = 'avec' if include_costs else 'sans'

        if backend == 'matplotlib':
            #fig = plt.figure(figsize=(12, 6))
            cumulative_returns.plot(title=f'Rendements cumulatifs {name_strat} {titre} coûts inclus')
            plt.tight_layout()
        
        elif backend == 'seaborn':
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=cumulative_returns).set_title(f'Rendements cumulatifs {name_strat} {titre} coûts inclus')
            plt.tight_layout()
        
        elif backend == 'plotly':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values,
                                name=f'Rendements cumulatifs {titre} coûts inclus'))
            fig.update_layout(title_text=f'Résultats du Backtest {name_strat}', height=600)
            fig.show()


def compare_results(results: dict, backend: str = 'matplotlib'):
    """Compare les résultats de plusieurs stratégies."""
    stats_comparison = pd.DataFrame([r.statistics for r in results.values()], index=results.keys())

    # liste des métrics avec valeurs très supérieur à 1 
    liste_sep = stats_comparison.loc[:, (stats_comparison > 20).any()].columns

    # Statistiques pour chaque dictionnaire
    stats_sep = stats_comparison[liste_sep]               # séparation des métrics avec valeurs très supérieur à 1  
    stats_main = stats_comparison.drop(columns=liste_sep)

    if backend == 'matplotlib':
        fig, ax1 = plt.subplots(figsize=(12, 6))
        stats_sep.plot(kind='bar', ax=ax1, color='lightblue', alpha=0.5, label=liste_sep)

        #ax1.set_ylabel('Nombre de Trades')
        ax1.set_xlabel('Stratégies')
        ax1.set_title('Comparaison des Stratégies')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        #other_metrics_normalized = other_metrics / other_metrics.abs().max()
        stats_main.plot(kind='bar', ax=ax2)#, alpha=0.7, width=0.8)

        ax2.set_ylabel('Métriques')
        ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
        plt.tight_layout()

    elif backend == 'seaborn':
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Convertir les données pour seaborn
        stats_sep_long = stats_sep.reset_index().melt(
            id_vars='index', 
            var_name='Metric', 
            value_name='Value'
        )
        stats_main_long = stats_main.reset_index().melt(
            id_vars='index', 
            var_name='Metric', 
            value_name='Value'
        )

        # Graphique pour les métriques séparées
        sns.barplot(
            data=stats_sep_long,
            x='index',
            y='Value',
            hue='Metric',
            ax=ax1,
            palette='coolwarm',
            alpha=0.5
        )
        
        ax1.set_xlabel('Stratégies')
        ax1.set_title('Comparaison des Stratégies')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(loc='upper left')

        # Créer un axe secondaire
        ax2 = ax1.twinx()

        # Graphique pour les métriques principales
        sns.barplot(
            data=stats_main_long,
            x='index',
            y='Value',
            hue='Metric',
            ax=ax2
        )
        ax2.set_ylabel('Métriques')
        ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
        plt.tight_layout()

    elif backend == 'plotly':
        fig = go.Figure()
        for col in stats_comparison.columns:
            fig.add_trace(go.Bar(name=col, x=list(stats_comparison.index), y=stats_comparison[col]))
        fig.update_layout(title='Comparaison des stratégies', barmode='group', height=600)
        fig.show()

        return fig
