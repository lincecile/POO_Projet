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


