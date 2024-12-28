import pandas as pd
import numpy as np
from mypackage import Strategy, Backtester, compare_results, strategy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

filepath = 'data.parquet'
# filepath = 'fichier_donnee.csv'
try:
    #lire en tant que CSV
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

# Stratégie basée sur les Moindres Carrés Ordinaires (MCO)
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

class OrdinaryLeastSquaresStrategy(Strategy):
    """
    Stratégie utilisant les Moindres Carrés Ordinaires (MCO) pour prédire les tendances des prix.

    Ajuste la position selon l'écart relatif entre le prix actuel et le prix prévu par une régression linéaire.
    """

    def __init__(self, window_size=20, threshold=0.01, rebalancing_frequency='D'):
        """
        Initialise la stratégie avec les paramètres donnés.

        :param window_size: Nombre de périodes pour la fenêtre de régression.
        :param threshold: Seuil pour déclencher une position (en pourcentage).
        :param rebalancing_frequency: Fréquence de rééquilibrage ('D', 'W', etc.).
        """
        super().__init__(rebalancing_frequency=rebalancing_frequency)
        self.window_size = window_size
        self.threshold = threshold

    def get_position(self, historical_data: pd.DataFrame, current_position: dict) -> dict:
        """
        Calcule les positions pour chaque actif en fonction de l'écart entre le prix actuel et le prix prévu.

        :param historical_data: Données historiques des actifs sous forme de DataFrame.
        :param current_position: Positions actuelles pour chaque actif.
        :return: Dictionnaire des nouvelles positions.
        """
        positions = {}

        for asset in historical_data.columns:
            asset_data = historical_data[asset]
            if len(asset_data) < self.window_size:
                # Pas assez de données pour effectuer la régression
                positions[asset] = 0
                continue

            # Préparation des données pour la régression linéaire
            prices = asset_data.iloc[-self.window_size:].values.reshape(-1, 1)
            time = np.arange(self.window_size).reshape(-1, 1)

            # Ajustement du modèle linéaire
            model = LinearRegression()
            model.fit(time, prices)

            # Prédiction du prix actuel
            predicted_price = model.predict([[self.window_size]])[0]
            current_price = asset_data.iloc[-1]

            # Calcul de l'écart relatif
            price_deviation = (current_price - predicted_price) / predicted_price

            # Génération des signaux de position
            if price_deviation > self.threshold:
                positions[asset] = -1  # Position short
            elif price_deviation < -self.threshold:
                positions[asset] = 1  # Position long
            else:
                positions[asset] = 0  # Pas de position

        return positions


class MovingAverageCrossover(Strategy):

# Cette stratégie utilise deux moyennes mobiles (courte et longue).
# Une position long est prise si la moyenne courte dépasse la longue, et short dans le cas inverse.

    def __init__(self, short_window=20, long_window=50, rebalancing_frequency='D'):
        super().__init__(rebalancing_frequency=rebalancing_frequency)
        self.short_window = short_window
        self.long_window = long_window

    def get_position(self, historical_data, current_position):
        if len(historical_data) < self.long_window:
            return 0

        short_ma = historical_data[historical_data.columns[0]].rolling(self.short_window).mean()
        long_ma = historical_data[historical_data.columns[0]].rolling(self.long_window).mean()

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
    returns = historical_data[historical_data.columns[0]].pct_change(20)
    return 1 if returns.iloc[-1] > 0 else -1

class MCOBasedStrategy(Strategy):
    """Stratégie basée sur l'optimisation du coût moyen (MCO)."""

    def __init__(self, threshold: float = 0.05, initial_position_cost= 0, rebalancing_frequency: str = 'D'):
        """
        Initialise la stratégie MCO.

        Args:
            threshold: Seuil relatif pour prendre une décision d'achat ou de vente (en pourcentage, ex: 0.05 pour 5%).
            rebalancing_frequency: Fréquence de rééquilibrage.
        """
        super().__init__(rebalancing_frequency)
        self.threshold = threshold
        self.average_cost = None  # Coût moyen de la position
        self.initial_position_cost = initial_position_cost

    def fit(self, data: pd.DataFrame) -> None:
        """
        Initialise le coût moyen avec un coût existant ou estimé.

        Args:
            data: DataFrame contenant les données historiques avec une colonne 'price'.
            initial_position_cost: Coût moyen initial de la position.
        """
        #if 'price' not in data.columns:
        #    raise ValueError("Les données historiques doivent contenir une colonne 'price'.")

        self.average_cost = self.initial_position_cost

    def update_average_cost(self, executed_price: float, executed_quantity: float, current_position: float) -> None:
        """
        Met à jour le coût moyen en fonction des transactions exécutées.

        Args:
            executed_price: Prix auquel la transaction a été exécutée.
            executed_quantity: Quantité achetée/vendue (positive pour achat, négative pour vente).
            current_position: Position actuelle avant la transaction.
        """
        new_position = current_position + executed_quantity
        if new_position == 0:   
            self.average_cost = None  # Position fermée, pas de coût moyen
        else:
            self.average_cost = (
                (self.average_cost * current_position + executed_price * executed_quantity) / new_position
            )

    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """
        Ajuste la position pour optimiser le coût moyen.

        Args:
            historical_data: DataFrame contenant les données historiques.
            current_position: Position actuelle.

        Returns:
            float: Position désirée (-1 pour vendre, 1 pour acheter, 0 pour neutre).
        """
        if self.average_cost is None:
            raise ValueError("Le coût moyen doit être initialisé avec la méthode fit().")

        if historical_data.empty: # Neutre si pas de données
            return 0.0  

        current_price = historical_data[historical_data.columns[0]].iloc[-1]

        if pd.isna(current_price): # Neutre si les données sont manquantes
            return 0.0  
        
        # Calcul de l'écart relatif par rapport au coût moyen
        price_deviation = (current_price - self.average_cost) / self.average_cost

        
        if price_deviation > self.threshold:        # Si le prix est bien supérieur au coût moyen, on vend
            return -1.0                             # Vendre
        elif price_deviation < -self.threshold:     # Si le prix est bien inférieur au coût moyen, on achète
            return 1.0                              # Acheter
        else:                                       # Neutre si le prix est proche du coût moyen
            return 0.0       

class VolatilityBasedStrategy(Strategy):

    """Stratégie basée sur la volatilité."""

# Cette stratégie identifie les actifs volatils en fonction d'un seuil.
# Les actifs avec une forte volatilité sont short, et ceux avec une faible volatilité sont long.

    def __init__(self, volatility_threshold=0.02, window_size=10, rebalancing_frequency='D'):
        super().__init__(rebalancing_frequency=rebalancing_frequency)
        self.volatility_threshold = volatility_threshold
        self.window_size = window_size
        self.volatility = None

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fonction qui calcule la volatilité sur une fenêtre donnée.

        Args:
            data: DataFrame contenant les données historiques avec une colonne 'price'.
        """
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
ma_strat_default = MovingAverageCrossover(20, 50)
ma_strat_weekly = MovingAverageCrossover(20, 50, rebalancing_frequency='W')
ma_strat_monthly = MovingAverageCrossover(20, 50, rebalancing_frequency='M')

ols_strat = OrdinaryLeastSquaresStrategy(window_size=20, threshold=0.01)
mom_strat_daily = momentum_strategy(rebalancing_frequency='D')

vol_strat_monthly = VolatilityBasedStrategy(0.02, 10, rebalancing_frequency='M')

# Dictionnaire des stratégies avec leurs coûts et liquidités
dico_strat = {
    'ma_strat_default': (ma_strat_default, 0.002, 0.0005),
    'ma_strat_weekly': (ma_strat_weekly, 0.01, 0.004),
    'ma_strat_monthly': (ma_strat_monthly, 0.005, 0.003),
    'ols_strat': (ols_strat, 0.002, 0.0005),
    'mom_strat_daily': (mom_strat_daily, 0.002, 0.0005),
    'vol_strat_monthly': (vol_strat_monthly, 0.002, 0.0005),
}

type_graph = 'matplotlib'  # 'plotly', 'seaborn', or 'matplotlib'

results_dict = {}

for name, (strat, cost, liquidite) in dico_strat.items():
    # Exécuter le backtest pour chaque stratégie et stocker le résultat dans le dictionnaire
    backtester = Backtester(data, transaction_costs=cost, slippage=liquidite)
    results_dict[name] = backtester.exec_backtest(strat)

    # Affichage des statistiques
    print(f"Statistiques de la stratégie {name}:")
    for key, value in results_dict[name].statistics.items():
        print(f"{key}: {value:.4f}")

# Visualisation des résultats
for strat_name, strat in results_dict.items():
    strat.plot(name_strat=strat_name, backend=type_graph)

# Comparaison des stratégies
compare_results(results_dict.values(), strat_name=list(dico_strat.keys()), backend=type_graph).show()

# Afin de garder les fenêtres graphiques ouvertes sur vscode
plt.show()
