from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    """Classe abstraite de base pour les stratégies de trading."""
    
    def __init__(self, rebalancing_frequency: str = 'D'):
        self.rebalancing_frequency = rebalancing_frequency      # fréquence de rééquilibrage pour chaque stratégie
    
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
    
    # Cette méthode ne fait rien si elle n’est pas implémentée
    def fit(self, data: pd.DataFrame) -> None:
        """Méthode optionnelle"""
        pass

def strategy(func):
    """Décorateur pour créer une stratégie simple à partir d'une fonction."""
    
    class WrappedStrategy(Strategy):
        def __init__(self, rebalancing_frequency='D', **kwargs):
            super().__init__(rebalancing_frequency=rebalancing_frequency)
            self.kwargs = kwargs
            
        def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
            return func(historical_data, current_position, self.rebalancing_frequency, **self.kwargs)
        
        @property
        def __name__(self):
            return f"{func.__name__}_{self.rebalancing_frequency}"
    
    return WrappedStrategy