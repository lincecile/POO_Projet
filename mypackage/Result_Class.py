import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Implémentez une classe Result pour afficher et visualiser les résultats du backtest, 
# avec différentes méthodes pour le plotting et l’affichage des statistiques de performance, incluant (mais non limité à) :
class Result:
    """Classe pour analyser et visualiser les résultats du backtest."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        positions: pd.DataFrame,
        trades: pd.DataFrame
    ):
        self.data = data
        self.positions = positions
        self.trades = trades
        self.returns = self._calculate_returns()
        self.statistics = self._calculate_statistics()
    
    def _calculate_returns(self) -> pd.Series:
        """Calcule les rendements de la stratégie."""
        price_returns = self.data[self.data.columns[0]].pct_change()
        strategy_returns = price_returns * self.positions['position'].shift(1)
        
        if not self.trades.empty:
            strategy_returns -= self.trades['cost']
        
        return strategy_returns.fillna(0)
    
    def _calculate_statistics(self) -> dict:
        """Calcule les statistiques de performance."""
        total_return = (1 + self.returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(self.returns)) - 1
        volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        drawdowns = (self.returns.cumsum() - self.returns.cumsum().cummax())
        max_drawdown = drawdowns.min()
        
        downside_returns = self.returns[self.returns < 0]
        sortino_ratio = (annual_return / (downside_returns.std() * np.sqrt(252))
                        if len(downside_returns) > 0 else 0)
        
        # Performance totale et annualisée
        # Volatilité
        # Ratio de Sharpe
        # Drawdown maximum
        # Ratio de Sortino
        # Nombre de trades
        # Pourcentage de trades gagnants
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino_ratio,
            'num_trades': len(self.trades),
            'win_rate': (self.returns[self.returns > 0].count() /
                        self.returns[self.returns != 0].count()
                        if self.returns[self.returns != 0].count() > 0 else 0)
        }
    
    # Implémentez la possibilité de choisir le backend pour les visualisations (matplotlib par défaut, avec options pour seaborn et plotly).
    def plot(self, backend: str = 'matplotlib'):
        """
        Visualise les résultats du backtest.
        
        Args:
            backend: 'matplotlib', 'seaborn', ou 'plotly'
        """
        cumulative_returns = (1 + self.returns).cumprod()
        
        if backend == 'matplotlib':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            cumulative_returns.plot(ax=ax1)
            ax1.set_title('Rendements cumulatifs')
            ax1.grid(True)
            
            self.positions['position'].plot(ax=ax2)
            ax2.set_title('Positions')
            ax2.grid(True)
            
            plt.tight_layout()
            return fig
            
        elif backend == 'seaborn':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            sns.lineplot(data=cumulative_returns, ax=ax1)
            ax1.set_title('Rendements cumulatifs')
            
            sns.lineplot(data=self.positions['position'], ax=ax2)
            ax2.set_title('Positions')
            
            plt.tight_layout()
            return fig
            
        elif backend == 'plotly':
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                name='Rendements cumulatifs'
            ))
            
            fig.add_trace(go.Scatter(
                x=self.positions.index,
                y=self.positions['position'].values,
                name='Positions',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Résultats du Backtest',
                yaxis2=dict(overlaying='y', side='right'),
                height=600
            )
            
            return fig

# Ajoutez une fonction compare_results(result_1, result_2, ...) pour comparer les résultats de différentes stratégies.
def compare_results(*results: Result, backend: str = 'matplotlib'):
    """Compare les résultats de plusieurs stratégies."""
    stats_comparison = pd.DataFrame([r.statistics for r in results])
    
    if backend == 'matplotlib':
        fig, ax = plt.subplots(figsize=(12, 6))
        stats_comparison.plot(kind='bar', ax=ax)
        ax.set_title('Comparaison des stratégies')
        plt.tight_layout()
        return fig
    
    elif backend == 'plotly':
        fig = go.Figure()
        print(stats_comparison)
        for col in stats_comparison.columns:
            fig.add_trace(go.Bar(
                name=col,
                x=list(range(len(stats_comparison))),
                y=stats_comparison[col]
            ))
        
        fig.update_layout(
            title='Comparaison des stratégies',
            barmode='group',
            height=600
        )
        
        return fig