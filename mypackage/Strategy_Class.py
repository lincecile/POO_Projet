from abc import ABC, abstractmethod
import pandas as pd
from typing import Callable
#aaa
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
def strategy(func: Callable) -> Strategy:
    """Décorateur pour créer une stratégie simple à partir d'une fonction."""
    class WrappedStrategy(Strategy):
        
        def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
            return func(historical_data, current_position)
    return WrappedStrategy()


