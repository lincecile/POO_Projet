import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mypackage import Strategy_Manager, Strategy, Backtester, compare_results, strategy, DataFileReader

# Définir le chemin vers vos fichiers de données
filepath = 'fichier_donnée.csv'

# Initialiser le lecteur de fichiers
reader = DataFileReader(date_format='%d/%m/%Y')
data = reader.read_file(filepath, date_column='Date_Price')

# Création d'une stratégie par héritage
class MovingAverageCrossover(Strategy):
    def __init__(self, short_window=20, long_window=50, rebalancing_frequency='D'):
        super().__init__(rebalancing_frequency=rebalancing_frequency)
        self.short_window = short_window
        self.long_window = long_window

    def get_position(self, historical_data, current_position):
        if len(historical_data) < self.long_window:
            return np.nan

        short_ma = historical_data.iloc[:, 0].rolling(self.short_window).mean()
        long_ma = historical_data.iloc[:, 0].rolling(self.long_window).mean()

        return 1 if short_ma.iloc[-1] > long_ma.iloc[-1] else -1

# Création d'une stratégie simple avec décorateur
@strategy
def momentum_strategy(historical_data, current_position, rebalancing_frequency, chosen_window=20):
    """
    Calcule un signal de trading basé sur le momentum.
    """
    if len(historical_data) < chosen_window:
        return 0

    returns = historical_data.iloc[:, 0].pct_change(chosen_window)
    return 1 if returns.iloc[-1] > 0 else -1

class VolatilityBasedStrategy(Strategy):
    def __init__(self, volatility_threshold=0.02, window_size=10, rebalancing_frequency='D'):
        super().__init__(rebalancing_frequency=rebalancing_frequency)
        self.volatility_threshold = volatility_threshold
        self.window_size = window_size
        self.volatility = None

    def fit(self, data: pd.DataFrame) -> None:
        daily_returns = data.iloc[:, 0].pct_change()
        self.volatility = daily_returns.rolling(window=self.window_size).std()

    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        if self.volatility is None:
            raise ValueError("La méthode fit() doit être appelée avant get_position().")

        current_volatility = self.volatility.iloc[-1]
        if current_volatility is None or pd.isna(current_volatility):
            return 0.0

        if current_volatility > self.volatility_threshold:
            return -1.0
        else:
            return 1.0

class MCOBasedStrategy(Strategy):
    def __init__(self, threshold: float = 0.05, initial_position_cost=0, rebalancing_frequency: str = 'D'):
        super().__init__(rebalancing_frequency)
        self.threshold = threshold
        self.average_cost = None
        self.initial_position_cost = initial_position_cost

    def fit(self, data: pd.DataFrame) -> None:
        self.average_cost = self.initial_position_cost

    def update_average_cost(self, executed_price: float, executed_quantity: float, current_position: float) -> None:
        new_position = current_position + executed_quantity
        if new_position == 0:
            self.average_cost = None
        else:
            self.average_cost = (
                (self.average_cost * current_position + executed_price * executed_quantity) / new_position
            )

    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        if self.average_cost is None:
            raise ValueError("Le coût moyen doit être initialisé avec la méthode fit().")

        if historical_data.empty:
            return 0.0

        current_price = historical_data.iloc[:, 0].iloc[-1]
        if pd.isna(current_price):
            return 0.0

        price_deviation = (current_price - self.average_cost) / self.average_cost

        if price_deviation > self.threshold:
            return -1.0
        elif price_deviation < -self.threshold:
            return 1.0
        else:
            return 0.0

# Ajout des stratégies au Strategy Manager
dico_strat = {
    'ma_strat_default': (MovingAverageCrossover(short_window=20, long_window=50), 0.002, 0.0005),
    'ma_strat_weekly': (MovingAverageCrossover(short_window=20, long_window=50, rebalancing_frequency='W'), 0.01, 0.004),
    'ma_strat_monthly': (MovingAverageCrossover(short_window=20, long_window=50, rebalancing_frequency='M'), 0.005, 0.003),
    'mom_strat_daily': (momentum_strategy(chosen_window=20, rebalancing_frequency='D'), 0.002, 0.0005),
    'mom_strat_weekly': (momentum_strategy(chosen_window=20, rebalancing_frequency='W'), 0.01, 0.004),
    'mom_strat_monthly': (momentum_strategy(chosen_window=20, rebalancing_frequency='M'), 0.005, 0.003),
    'vol_strat_monthly': (VolatilityBasedStrategy(volatility_threshold=0.02, window_size=10, rebalancing_frequency='M'), 0.002, 0.0005),
    'mco_strat_monthly': (MCOBasedStrategy(threshold=0.02, initial_position_cost=0.10, rebalancing_frequency='M'), 0.01, 0.004),
}

manager = Strategy_Manager(data, dico_strat)

# Exécution des backtests
manager.run_backtests()

# Affichage des statistiques
manager.print_statistics()

# Premier graphique : Num Trades Transparent
def plot_num_trades_transparency(manager):
    results = manager.results
    stats = pd.DataFrame([r.statistics for r in results.values()], index=results.keys())

    num_trades = stats['num_trades']
    other_metrics = stats.drop(columns=['num_trades'])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    num_trades.plot(kind='bar', ax=ax1, color='lightblue', alpha=0.5, label='Num Trades')

    ax1.set_ylabel('Nombre de Trades')
    ax1.set_xlabel('Stratégies')
    ax1.set_title('Comparaison des Stratégies avec Num Trades Transparent')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    other_metrics_normalized = other_metrics / other_metrics.abs().max()
    other_metrics_normalized.plot(kind='bar', ax=ax2, alpha=0.7, width=0.8)

    ax2.set_ylabel('Métriques Normalisées')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
    plt.tight_layout()
    plt.show()

# Deuxième graphique : Répartition des rendements cumulés
def plot_cumulative_returns(manager):
    results = manager.results
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, result in results.items():
        cumulative_returns = (1 + result.returns).cumprod()
        cumulative_returns.plot(ax=ax, label=name)

    ax.set_title('Rendements Cumulés des Stratégies')
    ax.set_xlabel('Temps')
    ax.set_ylabel('Rendements Cumulés')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

# Appel des deux fonctions graphiques
plot_num_trades_transparency(manager)
plot_cumulative_returns(manager)
