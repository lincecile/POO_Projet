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
        returns['portfolio'] = returns.mean(axis=1)  
        returns_without_cost['portfolio'] = returns_without_cost.mean(axis=1)  

        return returns, returns_without_cost
    
    def _calculate_statistics(self) -> dict:
        """Calcul des statistiques de performance de la stratégie"""
        
        # Performance total
        total_return = (1 + self.returns).prod() - 1 

        # Performance annualisée                                   
        annual_return = ((1 + total_return) ** (252 / len(self.returns))) - 1 if len(self.returns) > 0 else pd.Series(0, index=self.returns.columns)        
        
        # Volatilité annualisée
        volatility = self.returns.std() * np.sqrt(252)                                  

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

        # Vérification du backend entré par l'utilisateur
        if backend not in ['matplotlib', 'seaborn', 'plotly']:
            raise ValueError(f"Backend invalide. Valeurs possibles : {', '.join(['matplotlib', 'seaborn', 'plotly'])}")
        
        # Calcul le return cumulé
        returns_to_use = self.returns if include_costs else self.returns_no_cost
        cumulative_returns = (1 + returns_to_use).cumprod()
        
        titre = 'avec' if include_costs else 'sans'

        if backend == 'matplotlib':
            return cumulative_returns.plot(
                title=f'Rendements cumulatifs {name_strat} {titre} coûts inclus',
                figsize=(12, 6)
            ).get_figure()
        
        elif backend == 'seaborn':
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=cumulative_returns, ax=ax)
            ax.set_title(f'Rendements cumulatifs {name_strat} {titre} coûts inclus')
            plt.tight_layout()
            return fig
        
        elif backend == 'plotly':
            fig = go.Figure()
            # Pour chaque colonne dans les rendements cumulatifs
            if isinstance(cumulative_returns, pd.Series):
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns,
                    mode='lines',
                    name=name_strat
                ))
            else:  # Si c'est un DataFrame avec plusieurs colonnes
                for col in cumulative_returns.columns:
                    fig.add_trace(go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns[col],
                        mode='lines',
                        name=col
                    ))

            # Mise à jour du layout pour correspondre au style matplotlib
            fig.update_layout(
                title={
                    'text': f'Rendements cumulatifs {name_strat} {titre} coûts inclus',
                    'x': 0.5,
                    'y': 0.95
                },
                height=600,
                width=1000,
                xaxis_title="Date",
                yaxis_title="Rendement cumulé",
                template='plotly_white',
                showlegend=True,
                # Placer la légende à droite du graphique
                legend={
                    'orientation': 'v',
                    'yanchor': 'middle',
                    'y': 0.5,
                    'xanchor': 'left',
                    'x': 1.05,
                    'bgcolor': 'rgba(255, 255, 255, 0.8)',  # Fond légèrement transparent
                    'bordercolor': 'rgba(0, 0, 0, 0.2)',    # Bordure légère
                    'borderwidth': 1
                },
                # Ajuster les marges pour faire de la place à la légende
                margin={'l': 50, 'r': 150, 't': 60, 'b': 50}
            )

            return fig


def compare_results(results: dict, backend: str = 'matplotlib', show_plot: bool = True):
    """Compare les résultats de plusieurs stratégies."""

    # Tableau des statistiques calculés
    stats_comparison = pd.DataFrame([r.statistics for r in results.values()], index=results.keys())

    # Liste des métrics avec valeurs très supérieur à 1 
    liste_sep = stats_comparison.loc[:, (stats_comparison > 20).any()].columns

    # On s'assure d'avoir au moins une metrique pour la seconde axe
    liste_sep = ['num_trades'] if len(liste_sep) == 0 else liste_sep

    # Statistiques pour chaque dictionnaire
    stats_sep = stats_comparison[liste_sep]               # séparation des métrics avec valeurs très supérieur à 1  
    stats_main = stats_comparison.drop(columns=liste_sep)

    if backend == 'matplotlib':
        # Définir la colormap
        colormap = plt.cm.get_cmap('tab20c', len(stats_sep.columns))  # Colormap avec des couleurs claires
        # Créer une liste de couleurs claires pour chaque barre
        colors = [colormap(i / len(stats_sep.columns)) for i in range(len(stats_sep.columns))]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Graphique pour les métriques séparées
        stats_sep.plot(kind='bar', ax=ax1, color=colors, alpha=0.5, label=liste_sep)

        ax1.set_xlabel('Stratégies')
        ax1.set_title('Comparaison des Stratégies')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(loc='upper left')

        # Graphique pour les autres métriques
        ax2 = ax1.twinx()
        stats_main.plot(kind='bar', ax=ax2)

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

        # Pas de séparation implémenté puisque graphique dynamique
        for col in stats_comparison.columns:
            fig.add_trace(go.Bar(name=col, x=list(stats_comparison.index), y=stats_comparison[col]))
        fig.update_layout(title='Comparaison des stratégies', barmode='group', height=600)

    return fig
