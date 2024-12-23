import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mypackage import Strategy_Manager, Strategy, Backtester, compare_results, strategy, DataFileReader

# Définir le chemin vers vos fichiers de données
filepath = 'fichier_donnée.csv'

# Initialiser le lecteur de fichiers
reader = DataFileReader(date_format='%d/%m/%Y')
data = reader.read_file(filepath, date_column='Date_Price')
data = data[data.columns.to_list()[:10]]
all_asset = data.columns.to_list()

# Création d'une stratégie par héritage

class MovingAverageCrossover(Strategy):
    """Stratégie de croisement de moyennes mobiles pour plusieurs actifs."""
    
    def __init__(self, assets, short_window=20, long_window=50, rebalancing_frequency='D', 
                 allocation_method='equal'):
        super().__init__(rebalancing_frequency=rebalancing_frequency, assets=assets)
        self.short_window = short_window
        self.long_window = long_window
        self.allocation_method = allocation_method

    def get_position(self, historical_data, current_position):
        if len(historical_data) < self.long_window:
            return {asset: 0 for asset in self.assets}
        
        signals = {}
        for asset in self.assets:
            short_ma = historical_data[asset].rolling(self.short_window).mean()
            long_ma = historical_data[asset].rolling(self.long_window).mean()
            signals[asset] = 1 if short_ma.iloc[-1] > long_ma.iloc[-1] else -1
        
        # Allocation des positions selon la méthode choisie
        if self.allocation_method == 'equal':
            position_size = 1.0 / len(self.assets)
            positions = {asset: signal * position_size for asset, signal in signals.items()}
        else:  # 'signal_weighted'
            total_signals = sum(abs(signal) for signal in signals.values())
            positions = {asset: signal / total_signals for asset, signal in signals.items()}
            
        return positions


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


# Création des instances et exécution des backtests
ma_strat_default = MovingAverageCrossover(assets=all_asset, short_window=20, long_window=50)
ma_strat_weekly = MovingAverageCrossover(assets=all_asset, short_window=20, long_window=50, rebalancing_frequency='W')     # Weekly rebalancing
ma_strat_monthly = MovingAverageCrossover(assets=all_asset, short_window=20, long_window=50, rebalancing_frequency='M')    # Monthly rebalancing

mom_strat_daily = momentum_strategy(chosen_window=20,rebalancing_frequency='D')
mom_strat_weekly = momentum_strategy(chosen_window=20, rebalancing_frequency='W')
mom_strat_monthly = momentum_strategy(chosen_window=20, rebalancing_frequency='M')

vol_strat_monthly = VolatilityBasedStrategy(volatility_threshold=0.02, window_size=10, rebalancing_frequency='M')

mco_strat_monthly = MCOBasedStrategy(threshold=0.02, initial_position_cost=0.10, rebalancing_frequency='M')

dico_strat = {
    'ma_strat_default': (ma_strat_default, None, None),
    'ma_strat_weekly': (ma_strat_weekly, None, None),
    'ma_strat_monthly': (ma_strat_monthly, None, None),
    # 'mom_strat_daily': (mom_strat_daily, 0.002, 0.0005),
    # 'mom_strat_weekly': (mom_strat_weekly, 0.01, 0.004),
    # 'mom_strat_monthly': (mom_strat_monthly, 0.005, 0.003),
    # 'vol_strat_monthly': (vol_strat_monthly, 0.002, 0.0005),
    # 'mco_strat_monthly': (mco_strat_monthly, 0.01, 0.004),
}

manager = Strategy_Manager(data, dico_strat)

# Exécution des backtests
manager.run_backtests()

# Affichage des statistiques
manager.print_statistics()
manager.print_statistics(strategy_name="ma_strat_default")

# Visualize results
backend = 'seaborn' # 'plotly' # 'matplotlib' # 'seaborn'

# Plot individual strategies
manager.plot_all_strategies(backend=backend)

# Plot individual strategies
#manager.plot_strategy(strategy_name="ma_strat_default",backend=backend)

# Compare all strategies
manager.compare_strategies(backend=backend)

plt.show()


exit()

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
plot_cumulative_returns(manager)

