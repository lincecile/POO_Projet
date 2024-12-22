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
        self.returns = self._calculate_returns()
        
        statistics = self._calculate_statistics()
        self.statistics = {key: series['portfolio'] for key, series in statistics.items()}
        self.statistics_each_asset = {key: series.drop('portfolio') for key, series in statistics.items()}
        
    def _calculate_returns(self) -> pd.DataFrame:
        """Calcul des rendements de la stratégie pour chaque actif"""
        returns = pd.DataFrame(index=self.positions.index)
        
        for asset in self.positions.columns:
            price_returns = self.data[asset].pct_change()
            strategy_returns = price_returns * self.positions[asset].shift(1)
            
            # Prise en compte des coûts si des trades existent pour cet actif
            if not self.trades.empty:
                asset_trades = self.trades[self.trades['asset'] == asset]
                if not asset_trades.empty:
                    strategy_returns.loc[asset_trades.index] -= asset_trades['cost']
                    
            returns[asset] = strategy_returns.fillna(0)
        
        # Ajout d'une colonne pour le rendement total du portefeuille
        returns['portfolio'] = returns.mean(axis=1)  # Moyenne simple, peut être modifiée pour pondération personnalisée
        
        return returns
    
    def _calculate_statistics(self) -> dict:
        """Calcul des statistiques de performance de la stratégie"""
        
        total_return = (1 + self.returns).prod() - 1                                    # Performance total
        annual_return = ((1 + total_return) ** (252 / len(self.returns))) - 1           # Performance annualisé
        volatility = self.returns.std() * np.sqrt(252)                                  # Volatilité annualisée

        # Ratio de Sharpe
        sharpe_ratio = pd.Series([an_return / vol if vol != 0 else 0 for an_return, vol in zip(annual_return, volatility)], index=annual_return.index)
        
        # Maximum Drawdown
        cumulative_returns = (1 + self.returns).cumprod()
        drawdowns = cumulative_returns / cumulative_returns.cummax() - 1                
        max_drawdown = drawdowns.min()        

        # Ratio de Sortino
        downside_returns = self.returns[self.returns < 0]
        downside_deviation = (np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252) if not downside_returns.empty else 0)
        sortino_ratio = pd.Series([an_return / dev if dev != 0 else np.nan for an_return, dev in zip(annual_return, downside_deviation)], index=annual_return.index)
        
        # VaR (Value at Risk) à 95%
        var_95 = self.returns.apply(lambda col: col.quantile(0.05))

        # CVaR (Conditional Value at Risk) à 95%
        cvar_95 = self.returns[self.returns <= var_95].mean() if len(self.returns[self.returns <= var_95]) > 0 else 0

        # Gain/Pertes moyen (Profit/Loss Ratio)
        avg_gain = self.returns[self.returns > 0].mean() if len(self.returns[self.returns > 0]) > 0 else 0
        avg_loss = self.returns[self.returns < 0].mean() if len(self.returns[self.returns < 0]) > 0 else 0
        profit_loss_ratio = pd.Series([abs(avg_g / avg_l) if avg_l != 0 else np.nan for avg_g, avg_l in zip(avg_gain, avg_loss)], index=avg_gain.index)
        
        # Pourcentage de trades gagnants
        win_rate = self.returns.apply(lambda col: col[col > 0].count() / col[col != 0].count() if col[col != 0].count() > 0 else 0)

        # Facteur de profitabilité
        profit_factor = self.returns.apply(lambda col: abs(col[col > 0].sum() / col[col < 0].sum()) if col[col < 0].sum() != 0 else 1)

        # Nombre de trades
        nb_trade = self.trades['asset'].value_counts()
        nb_trade['portfolio'] = nb_trade.sum()

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
        """
        Visualise les résultats du backtest.
        
        Args:
            backend: 'matplotlib', 'seaborn', ou 'plotly'
            include_costs: Si True, inclut les coûts dans les rendements cumulés
        """

        # Calcul des rendements cumulés en fonction de l'inclusion des coûts
        if include_costs:
            cumulative_returns = (1 + self.returns).cumprod()
        else:
            price_returns = self.data[self.data.columns[0]].pct_change()
            cumulative_returns = (1 + price_returns * self.positions['position'].shift(1)).cumprod()
        
        if backend == 'matplotlib':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            cumulative_returns.plot(ax=ax1)
            ax1.set_title(f'Rendements cumulatifs {name_strat}')
            ax1.grid(True)
            
            self.positions['position'].plot(ax=ax2)
            ax2.set_title('Positions')
            ax2.grid(True)
            
            plt.tight_layout()
            
        elif backend == 'seaborn':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            sns.lineplot(data=cumulative_returns, ax=ax1)
            ax1.set_title(f'Rendements cumulatifs {name_strat}')
            
            sns.lineplot(data=self.positions['position'], ax=ax2)
            ax2.set_title('Positions')
            
            plt.tight_layout()
            
        elif backend == 'plotly':
            # Créer une figure avec deux sous-graphiques
            fig = make_subplots(rows=2, cols=1, 
                                subplot_titles=(f'Rendements cumulatifs {name_strat}', 'Positions'),
                                vertical_spacing=0.15)

            # Rendements cumulatifs
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.values,
                    name='Rendements cumulatifs'
                ),
                row=1, col=1
            )

            # Positions
            fig.add_trace(
                go.Scatter(
                    x=self.positions.index,
                    y=self.positions['position'].values,
                    name='Positions'
                ),
                row=2, col=1
            )

            fig.update_layout(
                height=800,
                showlegend=True,
                title_text=f'Résultats du Backtest {name_strat}'
            )

            fig.show()

        return fig
        
def compare_results(results: dict, liste_sep : list, backend: str = 'matplotlib'):
    """Compare les résultats de plusieurs stratégies."""

    # Statistiques de performance de chaque stratégie
    stats_comparison = pd.DataFrame([r.statistics for r in results.values()], index=results.keys())

    # Statistiques pour chaque dictionnaire
    stats_comparison_sep = stats_comparison[liste_sep]
    stats_comparison_without_sep = stats_comparison.drop(columns=liste_sep)

    if backend == 'matplotlib':
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        # Graphique sur le premier axe Y (ax1)
        stats_comparison_without_sep.plot(kind='bar', ax=ax1)
        stats_comparison_sep.plot(kind='bar', ax=ax2)

        ax1.set_title('Comparaison des stratégies')
        ax1.set_ylabel('Métriques 1')
        ax1.set_xticklabels(results.keys(), rotation=0, ha='right', fontsize=10)
        ax2.set_ylabel('Métriques 2')
        
        plt.tight_layout()
        
    elif backend == 'seaborn':
        fig, ax = plt.subplots(figsize=(12, 6))

        stats_long = stats_comparison.reset_index().melt(
            id_vars='index', 
            var_name='Metric', 
            value_name='Value'
        )
        sns.barplot(
            data=stats_long,
            x='index',
            y='Value',
            hue='Metric',
            ax=ax
        )
        ax.set_title('Comparaison des stratégies')
        plt.tight_layout()
    
    elif backend == 'plotly':
        fig = go.Figure()
        
        for col in stats_comparison.columns:
            fig.add_trace(go.Bar(
                name=col,
                x=strat_name,
                y=stats_comparison[col]
            ))
        
        fig.update_layout(
            title='Comparaison des stratégies',
            barmode='group',
            height=600
        )
        
    return fig