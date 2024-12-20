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

class VolatilityBasedStrategy(Strategy):
    """Stratégie basée sur la volatilité."""
    
    def __init__(self, volatility_threshold=0.02, window_size=10, rebalancing_frequency='D'):
        super().__init__(rebalancing_frequency=rebalancing_frequency)
        self.volatility_threshold = volatility_threshold
        self.window_size = window_size
        self.volatility = None

    def fit(self, data: pd.DataFrame) -> None:
        """
        Calcule la volatilité sur une fenêtre donnée.
        
        Args:
            data: DataFrame contenant les données historiques avec une colonne 'price'.
        """
        # if 'price' not in data.columns:
        #     raise ValueError("Les données historiques doivent contenir une colonne 'price'.")
        
        # Calculer les rendements journaliers
        daily_returns = data[data.columns[0]].pct_change()
        
        # Calculer la volatilité sur la fenêtre définie
        self.volatility = daily_returns.rolling(window=self.window_size).std()
    
    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """
        Détermine la position en fonction de la volatilité.
        
        Args:
            historical_data: DataFrame avec les données historiques.
            current_position: Position actuelle.
        
        Returns:
            float: Position désirée (-1 pour vendre, 1 pour acheter, 0 pour neutre).
        """
        if self.volatility is None:
            raise ValueError("La méthode fit() doit être appelée avant get_position().")
        
        # Vérifier la dernière valeur de volatilité
        current_volatility = self.volatility.iloc[-1]
        if current_volatility is None or pd.isna(current_volatility):
            return 0.0  # Neutre si volatilité non disponible
        
        # Prendre position selon la volatilité
        if current_volatility > self.volatility_threshold:
            return -1.0  # Vendre si la volatilité est élevée (risque élevé)
        else:
            return 1.0  # Acheter si la volatilité est faible (risque faible)


# Création des instances et exécution des backtests
backtester = Backtester(data, transaction_costs=0.001, slippage=0.0005)

ma_strat_default = MovingAverageCrossover(20, 50)
ma_strat_weekly = MovingAverageCrossover(20, 50, rebalancing_frequency='W') # Weekly rebalancing
ma_strat_monthly = MovingAverageCrossover(20, 50, rebalancing_frequency='M') # Monthly rebalancing

mom_strat_daily = momentum_strategy(rebalancing_frequency='D')
mom_strat_weekly = momentum_strategy(rebalancing_frequency='W')
mom_strat_monthly = momentum_strategy(rebalancing_frequency='M')

vol_strat_monthly = VolatilityBasedStrategy(0.02, 10, rebalancing_frequency='M')

strat1 = [ma_strat_default, ma_strat_weekly, ma_strat_monthly]
strat2 = [mom_strat_daily, mom_strat_weekly, mom_strat_monthly]
strat3 = [vol_strat_monthly]
list_strat = strat1 + strat2 + strat3

nom_strat = [obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__ for obj in list_strat]

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