import pandas as pd
import numpy as np
from mypackage import Strategy, Backtester, compare_results, strategy
import matplotlib.pyplot as plt

filepath = 'data.parquet'
#filepath = 'fichier_donnée.csv'
try:
    # Essayer de lire en tant que CSV
    data = pd.read_csv(filepath, sep=';').replace(',', '.', regex=True)
    data['Date_Price'] = pd.to_datetime(data['Date_Price'], format='%d/%m/%Y')
    data.set_index('Date_Price', inplace=True)
except Exception as e_csv:
    try:
        data = pd.read_parquet(filepath)
    except Exception as e_parquet:
        raise ValueError(f"Impossible de lire le fichier : {filepath}. "
                            "Format non supporté ou fichier invalide.")

data = data.astype(float)

# Création d'une stratégie par héritage
class MovingAverageCrossover(Strategy):
    
    def __init__(self, short_window=20, long_window=50, rebalancing_frequency='D'):
        super().__init__(rebalancing_frequency=rebalancing_frequency)
        self.short_window = short_window
        self.long_window = long_window

    def get_position(self, historical_data, current_position):
        if len(historical_data) < self.long_window:
            return 0
        
        short_ma = historical_data[data.columns[0]].rolling(self.short_window).mean()
        long_ma = historical_data[data.columns[0]].rolling(self.long_window).mean()
        
        if short_ma.iloc[-1] > long_ma.iloc[-1]:
            return 1
        else:
            return -1
    
    @property
    def __name__(self):
        return f"{self.__class__.__name__}_{self.rebalancing_frequency}"

# Création d'une stratégie simple avec décorateur
@strategy
def momentum_strategy(historical_data, current_position, rebalancing_frequency='D'):
    if len(historical_data) < 20:
        return 0
    returns = historical_data[data.columns[0]].pct_change(20)
    return 1 if returns.iloc[-1] > 0 else -1

# Création des instances et exécution des backtests
backtester = Backtester(data, transaction_costs=0.001, slippage=0.0005)

ma_strat_default = MovingAverageCrossover(20, 50)
ma_strat_weekly = MovingAverageCrossover(20, 50, rebalancing_frequency='W') # Weekly rebalancing
ma_strat_monthly = MovingAverageCrossover(20, 50, rebalancing_frequency='M') # Monthly rebalancing

mom_strat_daily = momentum_strategy(rebalancing_frequency='D')
mom_strat_weekly = momentum_strategy(rebalancing_frequency='W')
mom_strat_monthly = momentum_strategy(rebalancing_frequency='M')

strat1 = [ma_strat_default, ma_strat_weekly, ma_strat_monthly]
strat2 = [mom_strat_daily, mom_strat_weekly, mom_strat_monthly]
list_strat = strat1 + strat2

nom_strat = [obj.__name__ for obj in list_strat]
type_graph = 'plotly'#'plotly'#'seaborn'#'matplotlib'

results_dict = {}
for strat, name in zip(list_strat, nom_strat):
    # Exécuter le backtest pour chaque stratégie et stocker le résultat dans le dictionnaire
    results_dict[name] = backtester.run(strat)

    # Affichage des statistiques
    print(f"Statistiques de la stratégie {name}:")
    for key, value in results_dict[name].statistics.items():
        print(f"{key}: {value:.4f}")

# Visualisation des résultats
for strat_name, strat in results_dict.items():
    strat.plot(name_strat = strat_name, backend=type_graph)

# Comparaison des stratégies
compare_results(results_dict.values(), strat_name=nom_strat, backend=type_graph).show()

# afin de garder les fenetres graphiques ouvertes sur vscode
plt.show()