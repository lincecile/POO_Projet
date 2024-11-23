import pandas as pd
import numpy as np
from mypackage import Strategy, Backtester, compare_results, strategy

data = pd.read_csv('fichier_donnée.csv', sep=';')
data['Date_Price'] = pd.to_datetime(data['Date_Price'], format='%d/%m/%Y')
data.set_index('Date_Price', inplace=True)
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

result_ma = backtester.run(ma_strategy)
result_mom = backtester.run(mom_strategy)

# Affichage des statistiques
print("Statistiques de la stratégie Moving Average:")
for key, value in result_ma.statistics.items():
    print(f"{key}: {value:.4f}")

print("\nStatistiques de la stratégie Momentum:")
for key, value in result_mom.statistics.items():
    print(f"{key}: {value:.4f}")

# Visualisation des résultats
result_ma.plot(backend='plotly')
result_mom.plot(backend='plotly')

# Comparaison des stratégies
compare_results(result_ma, result_mom, backend='plotly')