import pandas as pd
import numpy as np
from mypackage import Strategy, Backtester, compare_results, strategy

data = pd.read_csv('fichier_donnée.csv', sep=';').replace(',', '.', regex=True)
data['Date_Price'] = pd.to_datetime(data['Date_Price'], format='%d/%m/%Y')
data.set_index('Date_Price', inplace=True)
data = data.astype(float)
print(data)

# Création d'une stratégie par héritage
class MovingAverageCrossover(Strategy):
    def __init__(self, short_window=20, long_window=50):
        super().__init__(rebalancing_frequency='D')
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

# Création d'une stratégie simple avec décorateur
@strategy
def momentum_strategy(historical_data, current_position):
    if len(historical_data) < 20:
        return 0
    
    returns = historical_data[data.columns[0]].pct_change(20)
    return 1 if returns.iloc[-1] > 0 else -1

# Création des instances et exécution des backtests
backtester = Backtester(data, transaction_costs=0.001)

ma_strategy = MovingAverageCrossover(20, 50)
mom_strategy = momentum_strategy

list_strat = [ma_strategy,mom_strategy]
nom_strat = [obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__ for obj in list_strat]

num = -1
results_dict = {}
for strat in list_strat:
    num += 1
    results_dict[f'result_strat_{num}'] = backtester.run(strat)

    # Affichage des statistiques
    print(f"Statistiques de la stratégie {nom_strat[num]}:")
    for key, value in results_dict[f'result_strat_{num}'].statistics.items():
        print(f"{key}: {value:.4f}")

# Visualisation des résultats
for strat in results_dict.values():
    strat.plot(backend='plotly')

# Comparaison des stratégies
compare_results(results_dict.values(), strat_name=nom_strat, backend='plotly').show()