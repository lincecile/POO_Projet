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
def momentum_strategy(historical_data, current_position, assets, rebalancing_frequency, chosen_window=20):
    """
    Calcule un signal de trading basé sur le momentum pour plusieurs actifs.
    
    Args:
        historical_data (pd.DataFrame): Données historiques avec les prix des actifs.
        current_position (dict): Positions actuelles sur les actifs.
        assets (list): Liste des actifs à trader.
        rebalancing_frequency (str): Fréquence de rééquilibrage ('D', 'W', etc.).
        chosen_window (int): Fenêtre de temps pour le calcul du momentum (en périodes).
    
    Returns:
        dict: Positions pondérées pour chaque actif.
    """
    # Vérifie si suffisamment de données sont disponibles pour la fenêtre choisie
    if len(historical_data) < chosen_window:
        return {asset: 0 for asset in assets}

    positions = {}
    active_assets = []  # Liste pour enregistrer les actifs ayant des signaux valides
    
    for asset in assets:
        try:
            # Calcul du rendement à la période donnée (momentum)
            returns = historical_data[asset].pct_change(chosen_window)
            
            # Vérifie si le dernier rendement est valide
            if pd.isna(returns.iloc[-1]) or len(returns) == 0:
                positions[asset] = 0
            else:
                # Génère un signal : 1 si momentum positif, -1 si négatif
                signal = 1 if returns.iloc[-1] > 0 else -1
                positions[asset] = signal
                active_assets.append(asset)
        except KeyError:
            # Actif manquant dans les données historiques
            positions[asset] = 0
        except Exception as e:
            # Capture et ignore les erreurs inattendues
            positions[asset] = 0
            print(f"Erreur pour l'actif {asset}: {e}")
    
    # Ajuste les positions uniquement pour les actifs actifs
    if active_assets:
        allocation_size = 1.0 / len(active_assets)  # Allocation égale pour chaque actif actif
        for asset in active_assets:
            positions[asset] *= allocation_size
    
    return positions


class VolatilityBasedStrategy(Strategy):
    """
    Stratégie basée sur la volatilité des actifs.
    
    Si la volatilité d'un actif dépasse un seuil donné, une position short est prise.
    Si la volatilité est inférieure au seuil, une position long est prise.
    """

    def __init__(self, assets, volatility_threshold=0.02, window_size=10, rebalancing_frequency='D', allocation_method='equal'):
        """
        Initialise la stratégie avec les paramètres donnés.

        :param assets: Liste des actifs à trader.
        :param volatility_threshold: Seuil pour décider entre position long/short.
        :param window_size: Fenêtre pour le calcul de la volatilité.
        :param rebalancing_frequency: Fréquence de rééquilibrage ('D', 'W', etc.).
        :param allocation_method: Méthode d'allocation ('equal' ou 'volatility_weighted').
        """
        super().__init__(rebalancing_frequency=rebalancing_frequency, assets=assets)
        self.volatility_threshold = volatility_threshold
        self.window_size = window_size
        self.volatility = None
        self.allocation_method = allocation_method

    def fit(self, data: pd.DataFrame) -> None:
        """
        Calcule la volatilité historique pour chaque actif.

        :param data: Données historiques (prix des actifs).
        """
        self.volatility = {}
        for asset in self.assets:
            daily_returns = data[asset].pct_change()
            self.volatility[asset] = daily_returns.rolling(window=self.window_size).std()

    def get_position(self, historical_data: pd.DataFrame, current_position: dict) -> dict:
        """
        Détermine les positions basées sur la volatilité de chaque actif.

        :param historical_data: Données historiques des actifs.
        :param current_position: Positions actuelles.
        :return: Dictionnaire des positions pour chaque actif.
        """
        if self.volatility is None:
            raise ValueError("La méthode fit() doit être appelée avant get_position().")
        
        positions = {}
        active_assets = []
        
        # Déterminer les signaux pour chaque actif
        for asset in self.assets:
            current_vol = self.volatility[asset].iloc[-1] if asset in self.volatility else None
            
            if current_vol is None or pd.isna(current_vol):
                # Actif sans données valides
                positions[asset] = 0
            else:
                # Décider entre position short ou long
                if current_vol > self.volatility_threshold:
                    positions[asset] = -1  # Position short
                else:
                    positions[asset] = 1  # Position long
                active_assets.append(asset)
        
        # Ajuster les allocations selon la méthode choisie
        if active_assets:
            if self.allocation_method == 'equal':
                position_size = 1.0 / len(active_assets)
                for asset in active_assets:
                    positions[asset] *= position_size
            elif self.allocation_method == 'volatility_weighted':
                total_vol = sum(self.volatility[asset].iloc[-1] for asset in active_assets)
                total_vol = total_vol or 1  # Évite division par zéro
                for asset in active_assets:
                    weight = self.volatility[asset].iloc[-1] / total_vol
                    positions[asset] *= weight
                    
        return positions

class MCOBasedStrategy(Strategy):
    """
    Stratégie basée sur la déviation du prix actuel par rapport au coût moyen des actifs.
    Une forte déviation positive conduit à une position short, une forte déviation négative à une position long.
    """

    def __init__(self, assets, threshold: float = 0.05, initial_position_costs: dict = None, rebalancing_frequency: str = 'D'):
        """
        Initialise la stratégie.

        :param assets: Liste des actifs à trader.
        :param threshold: Seuil de déviation du prix pour prendre une position.
        :param initial_position_costs: Coûts moyens initiaux pour chaque actif.
        :param rebalancing_frequency: Fréquence de rééquilibrage ('D', 'W', etc.).
        """
        super().__init__(rebalancing_frequency=rebalancing_frequency, assets=assets)
        self.threshold = threshold
        self.average_costs = None
        self.initial_position_costs = initial_position_costs or {asset: 0 for asset in assets}

    def fit(self, data: pd.DataFrame) -> None:
        """
        Initialise les coûts moyens pour chaque actif.

        :param data: Données historiques utilisées pour initialiser la stratégie.
        """
        self.average_costs = self.initial_position_costs.copy()

    def update_average_cost(self, asset: str, executed_price: float, executed_quantity: float, current_position: float) -> None:
        """
        Met à jour le coût moyen pour un actif spécifique après une transaction.

        :param asset: Nom de l'actif.
        :param executed_price: Prix d'exécution de la transaction.
        :param executed_quantity: Quantité exécutée.
        :param current_position: Position actuelle avant la transaction.
        """
        new_position = current_position + executed_quantity
        if new_position == 0:
            self.average_costs[asset] = self.initial_position_costs[asset]
        else:
            self.average_costs[asset] = (
                (self.average_costs[asset] * current_position + executed_price * executed_quantity) / new_position
            )

    def get_position(self, historical_data: pd.DataFrame, current_position: dict) -> dict:
        """
        Calcule les positions basées sur la déviation du prix par rapport au coût moyen.

        :param historical_data: Données historiques avec les prix des actifs.
        :param current_position: Positions actuelles des actifs.
        :return: Dictionnaire des positions calculées.
        """
        if self.average_costs is None:
            raise ValueError("Les coûts moyens doivent être initialisés avec la méthode fit().")
        
        positions = {}
        active_assets = []
        
        for asset in self.assets:
            if historical_data[asset].empty:
                positions[asset] = 0
                continue

            current_price = historical_data[asset].iloc[-1]
            if pd.isna(current_price):
                positions[asset] = 0
                continue

            avg_cost = self.average_costs[asset]
            if avg_cost == 0:  # Éviter la division par zéro
                positions[asset] = 0
                continue

            price_deviation = (current_price - avg_cost) / avg_cost

            if price_deviation > self.threshold:
                positions[asset] = -1  # Position short
                active_assets.append(asset)
            elif price_deviation < -self.threshold:
                positions[asset] = 1  # Position long
                active_assets.append(asset)
            else:
                positions[asset] = 0
        
        # Ajuster les positions pour une allocation égale
        if active_assets:
            position_size = 1.0 / len(active_assets)
            for asset in active_assets:
                positions[asset] *= position_size
                
        return positions



# Création des instances et exécution des backtests
ma_strat_default = MovingAverageCrossover(assets=all_asset, short_window=20, long_window=50)
ma_strat_weekly = MovingAverageCrossover(assets=all_asset, short_window=20, long_window=50, rebalancing_frequency='W')     # Weekly rebalancing
ma_strat_monthly = MovingAverageCrossover(assets=all_asset, short_window=20, long_window=50, rebalancing_frequency='M')    # Monthly rebalancing

mom_strat_daily = momentum_strategy(assets=all_asset, chosen_window=10, rebalancing_frequency='D')
mom_strat_weekly = momentum_strategy(assets=all_asset, chosen_window=10, rebalancing_frequency='W')
mom_strat_monthly = momentum_strategy(assets=all_asset, chosen_window=10, rebalancing_frequency='M')

vol_strat_monthly = VolatilityBasedStrategy(
    assets=all_asset,
    volatility_threshold=0.02,
    window_size=10,
    rebalancing_frequency='M',
    allocation_method='equal'
)

initial_costs = {asset: 0.10 for asset in all_asset}
mco_strat_monthly = MCOBasedStrategy(
    assets=all_asset,
    threshold=0.02,
    initial_position_costs=initial_costs,
    rebalancing_frequency='M'
)

transaction_cost = {'AAVEUSDT': 0.5, 'ACMUSDT': 0.2, 'ADAUSDT': 0.4, 'ADXUSDT': 0.6, 'AGLDUSDT': 0.7, 'ALGOUSDT': 0.001, 'ALICEUSDT': 0.8, 'ALPACAUSDT': 2, 'ALPHAUSDT': 0.8, 'AMPUSDT': 0.051}
slippage_cost = {'AAVEUSDT': 0.53, 'ACMUSDT': 0.03, 'ADAUSDT': 0.33, 'ADXUSDT': 0.63, 'AGLDUSDT': 0.73, 'ALGOUSDT': 0.001, 'ALICEUSDT': 0.8, 'ALPACAUSDT': 21, 'ALPHAUSDT': 0.8, 'AMPUSDT': 0.01}

transaction_cost_partiel = {'AAVEUSDT': 5, 'ACMUSDT': 2}

dico_strat = {
    'ma_strat_default': (ma_strat_default, 0.1, 0.2),
    'ma_strat_weekly': (ma_strat_weekly, transaction_cost_partiel, None),
    'ma_strat_monthly': (ma_strat_monthly, transaction_cost, slippage_cost),
    'mom_strat_daily': (mom_strat_daily, 0.002, 0.0005),
    'mom_strat_weekly': (mom_strat_weekly, 0.01, 0.004),
    'mom_strat_monthly': (mom_strat_monthly, 0.005, 0.003),
    'vol_strat_monthly': (vol_strat_monthly, 0.002, 0.0005),
    'mco_strat_monthly': (mco_strat_monthly, 0.01, 0.004),
}

#print(ma_strat_default.get_position(data,1))

manager = Strategy_Manager(data, dico_strat)

# Exécution des backtests
manager.run_backtests()

# Affichage des statistiques
manager.print_statistics()
manager.print_statistics(strategy_name="ma_strat_default")
manager.print_statistics(strategy_name="ma_strat_default",detail=True)
manager.print_statistics(detail=True)


# Visualize results
backend = 'matplotlib' # 'plotly' # 'matplotlib' # 'seaborn'

# Plot individual strategies
manager.plot_all_strategies(backend=backend,include_costs=True)
manager.plot_all_strategies(backend=backend,include_costs=False)

# Plot individual strategies
manager.plot_strategy(strategy_name="ma_strat_default",backend=backend,include_costs=True)
manager.plot_strategy(strategy_name="ma_strat_default",backend=backend,include_costs=False)

# Compare all strategies
manager.compare_strategies(backend=backend)

plt.show()
