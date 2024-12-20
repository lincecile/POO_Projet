from abc import ABC, abstractmethod
import pandas as pd
from typing import Callable

# Créez une classe abstraite Strategy avec les contraintes suivantes 
class Strategy(ABC):
    """Classe abstraite de base pour les stratégies de trading."""
    
    # Permettez de spécifier une fréquence de rééquilibrage pour chaque stratégie.
    def __init__(self, rebalancing_frequency: str = 'D'):
        self.rebalancing_frequency = rebalancing_frequency
    
    # Méthode obligatoire : get_position(historical_data, current_position)
    @abstractmethod
    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """
        Calcule la position désirée basée sur les données historiques.
        
        Args:
            historical_data: DataFrame avec les données historiques
            current_position: Position actuelle
            
        Returns:
            float: Position désirée (-1 à 1)
        """
        pass
    
    # Méthode optionnelle : fit(data) (par défaut, cette méthode ne fait rien si elle n’est pas implémentée)
    def fit(self, data: pd.DataFrame) -> None:
        """Méthode optionnelle"""
        pass

# Permettez la création de stratégies 
#   soit par héritage de la classe abstraite, 
#   soit par un décorateur pour les stratégies simples ne nécessitant que get_position.
def strategy(func):
    """Décorateur pour créer une stratégie simple à partir d'une fonction."""
    
    class WrappedStrategy(Strategy):
        def __init__(self, rebalancing_frequency='D'):
            super().__init__(rebalancing_frequency=rebalancing_frequency)
            
        def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
            return func(historical_data, current_position, self.rebalancing_frequency)
        
        @property
        def __name__(self):
            return f"{func.__name__}_{self.rebalancing_frequency}"
    
    return WrappedStrategy

class VolatilityBasedStrategy(Strategy):
    """Stratégie basée sur la volatilité."""
    
    def __init__(self, volatility_threshold: float = 0.02, window_size: int = 10, rebalancing_frequency: str = 'D'):
        super().__init__(rebalancing_frequency)
        self.volatility_threshold = volatility_threshold
        self.window_size = window_size
        self.volatility = None

    def fit(self, data: pd.DataFrame) -> None:
        """
        Calcule la volatilité sur une fenêtre donnée.
        
        Args:
            data: DataFrame contenant les données historiques avec une colonne 'price'.
        """
        if 'price' not in data.columns:
            raise ValueError("Les données historiques doivent contenir une colonne 'price'.")
        
        # Calculer les rendements journaliers
        daily_returns = data['price'].pct_change()
        
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

class MCOBasedStrategy(Strategy):
    """Stratégie basée sur l'optimisation du coût moyen (MCO)."""

    def __init__(self, threshold: float = 0.05, rebalancing_frequency: str = 'D'):
        """
        Initialise la stratégie MCO.

        Args:
            threshold: Seuil relatif pour prendre une décision d'achat ou de vente (en pourcentage, ex: 0.05 pour 5%).
            rebalancing_frequency: Fréquence de rééquilibrage.
        """
        super().__init__(rebalancing_frequency)
        self.threshold = threshold
        self.average_cost = None  # Coût moyen de la position

    def fit(self, data: pd.DataFrame, initial_position_cost: float) -> None:
        """
        Initialise le coût moyen avec un coût existant ou estimé.

        Args:
            data: DataFrame contenant les données historiques avec une colonne 'price'.
            initial_position_cost: Coût moyen initial de la position.
        """
        if 'price' not in data.columns:
            raise ValueError("Les données historiques doivent contenir une colonne 'price'.")

        self.average_cost = initial_position_cost

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

        if historical_data.empty:
            return 0.0  # Neutre si pas de données

        current_price = historical_data['price'].iloc[-1]

        if pd.isna(current_price):
            return 0.0  # Neutre si les données sont manquantes

        # Calcul de l'écart relatif par rapport au coût moyen
        price_deviation = (current_price - self.average_cost) / self.average_cost

        # Si le prix est bien supérieur au coût moyen, on vend
        if price_deviation > self.threshold:
            return -1.0  # Vendre
        # Si le prix est bien inférieur au coût moyen, on achète
        elif price_deviation < -self.threshold:
            return 1.0  # Acheter
        else:
            return 0.0  # Neutre si le prix est proche du coût moyen

