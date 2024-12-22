import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Result:
    """Classe pour analyser et visualiser les résultats du backtest."""

    def __init__(self, data: pd.DataFrame, positions: pd.DataFrame, trades: pd.DataFrame):
        self.data = data
        self.positions = positions
        self.trades = trades
        self.returns = self._calculate_returns()
        self.statistics = self._calculate_statistics()

    def _calculate_returns(self) -> pd.Series:
        """Calcul des rendements de la stratégie."""
        price_returns = self.data[self.data.columns[0]].pct_change()
        strategy_returns = price_returns * self.positions['position'].shift(1)

        # Prise en compte du coût des transactions
        if not self.trades.empty:
            strategy_returns -= self.trades['cost']

        return strategy_returns.fillna(0)

    def _calculate_statistics(self) -> dict:
        """Calcul des statistiques de performance de la stratégie."""
        total_return = (1 + self.returns).prod() - 1
        annual_return = ((1 + total_return) ** (252 / len(self.returns))) - 1
        volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0

        # Maximum Drawdown
        cumulative_returns = (1 + self.returns).cumprod()
        drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
        max_drawdown = drawdowns.min()

        # Pourcentage de trades gagnants
        win_rate = (self.returns[self.returns > 0].count() /
                    self.returns[self.returns != 0].count() if self.returns[self.returns != 0].count() > 0 else 0)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'win_rate': win_rate
        }


def compare_results_fixed_xaxis(results: dict):
    """
    Compare les résultats des stratégies avec un axe pour 'num_trades' et un autre pour le reste des métriques.

    Args:
        results (dict): Dictionnaire contenant les objets Result pour chaque stratégie.
    """
    # Extraire les statistiques pour chaque stratégie
    stats_comparison = pd.DataFrame([r.statistics for r in results.values()], index=results.keys())

    # Séparer les métriques
    num_trades = stats_comparison['num_trades']
    other_metrics = stats_comparison.drop(columns=['num_trades'])

    # Création du graphique
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Graphe pour 'num_trades' (axe gauche)
    num_trades.plot(kind='bar', ax=ax1, color='skyblue', position=0, width=0.4, label='num_trades')

    ax1.set_ylabel("Nombre de trades", fontsize=12)
    ax1.set_xlabel("Stratégies", fontsize=12)
    ax1.set_title("Comparaison des stratégies", fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc="upper left")

    # Graphe pour les autres métriques (axe droit)
    ax2 = ax1.twinx()
    other_metrics_normalized = other_metrics / other_metrics.abs().max()  # Normalisation des autres métriques
    other_metrics_normalized.plot(kind='bar', ax=ax2, width=0.4, position=1, alpha=0.7)

    ax2.set_ylabel("Métriques normalisées (autres)", fontsize=12)
    ax2.legend(loc="upper right")

    # Mise en page
    plt.tight_layout()
    plt.show()


# Exemple d'appel avec des données fictives
data1 = pd.DataFrame({'price': np.random.rand(100)})
positions1 = pd.DataFrame({'position': np.random.choice([-1, 0, 1], size=100)})
trades1 = pd.DataFrame({'cost': np.random.rand(10)})

data2 = pd.DataFrame({'price': np.random.rand(100)})
positions2 = pd.DataFrame({'position': np.random.choice([-1, 0, 1], size=100)})
trades2 = pd.DataFrame({'cost': np.random.rand(10)})

results = {
    'MA Strat Default': Result(data1, positions1, trades1),
    'Momentum': Result(data2, positions2, trades2)
}

# Comparer les stratégies avec un axe fixe pour les abscisses
compare_results_fixed_xaxis(results)
