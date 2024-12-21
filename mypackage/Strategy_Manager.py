# StrategyManager.py
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Union
from .Strategy_Class import Strategy, strategy
from .Backtester_Class import Backtester
from .Result_Class import Result, compare_results

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Union, List
from .Strategy_Class import Strategy
from .Backtester_Class import Backtester
from .Result_Class import Result, compare_results

class Strategy_Manager:
    """
    Classe pour gérer et comparer plusieurs stratégies de trading.
    Permet de gérer les backtests, les visualisations et les comparaisons de plusieurs stratégies.
    """
    
    def __init__(self, data: pd.DataFrame, strategies_dict: Dict[str, Tuple[Strategy, float, float]] = None):
        """
        Initialise le Strategy_Manager avec les données de marché et éventuellement un dictionnaire de stratégies.
        
        Args:
            data: DataFrame contenant les données de marché.
            strategies_dict: Dictionnaire des stratégies au format :
                             {nom: (instance_stratégie, coûts_de_transaction, slippage)}
        """
        self.data = data
        self.strategies: Dict[str, Tuple[Strategy, float, float]] = {}
        self.results: Dict[str, Result] = {}
        
        if strategies_dict:
            self.add_strategies_from_dict(strategies_dict)
    
    def add_strategies_from_dict(self, strategies_dict: Dict[str, Tuple[Strategy, float, float]]) -> None:
        """
        Ajouter plusieurs stratégies depuis un dictionnaire.
        
        Args:
            strategies_dict: Dictionnaire au format :
                             {nom: (instance_stratégie, coûts_de_transaction, slippage)}
        """
        for name, (strategy, costs, slip) in strategies_dict.items():
            self.add_strategy(name, strategy, costs, slip)
        
    def add_strategy(self, name: str, strategy: Strategy, transaction_costs: float = 0.001, slippage: float = 0.0005) -> None:
        """
        Ajouter une stratégie au gestionnaire.
        
        Args:
            name: Identifiant unique pour la stratégie.
            strategy: Instance de la stratégie.
            transaction_costs: Coûts de transaction pour la stratégie.
            slippage: Coûts liés au slippage pour la stratégie.
        """
        if name in self.strategies:
            raise ValueError(f"Strategy with name '{name}' already exists")
        
        self.strategies[name] = (strategy, transaction_costs, slippage)
        
    def remove_strategy(self, name: str) -> None:
        """
        Supprimer une stratégie du gestionnaire.
        
        Args:
            name: Nom de la stratégie à supprimer.
        """
        if name in self.strategies:
            del self.strategies[name]
            if name in self.results:
                del self.results[name]
                
    def run_backtests(self) -> None:
        """Exécute les backtests pour toutes les stratégies enregistrées."""
        self.results.clear()
        
        for name, (strategy, costs, slip) in self.strategies.items():
            backtester = Backtester(self.data, transaction_costs=costs, slippage=slip)
            self.results[name] = backtester.run(strategy)
            
    def get_statistics(self, strategy_name: Union[str, None] = None) -> Union[Dict, pd.DataFrame]:
        """
        Obtenir les statistiques pour une stratégie ou pour toutes les stratégies.
        
        Args:
            strategy_name: Nom d'une stratégie spécifique, ou None pour toutes les stratégies.
            
        Returns:
            Dictionnaire des statistiques pour une stratégie ou DataFrame pour toutes les stratégies.
        """
        if not self.results:
            raise ValueError("No results available. Run backtests first.")
            
        if strategy_name is not None:
            if strategy_name not in self.results:
                raise ValueError(f"No results found for strategy '{strategy_name}'")
            return self.results[strategy_name].statistics
            
        return pd.DataFrame([result.statistics for result in self.results.values()],
                          index=self.results.keys())
    
    def plot_strategy(self, strategy_name: str, backend: str = 'matplotlib', include_costs: bool = True) -> None:
        """
        Tracer les résultats pour une stratégie spécifique.
        
        Args:
            strategy_name: Nom de la stratégie à tracer.
            backend: Backend de visualisation ('matplotlib', 'seaborn', ou 'plotly').
            include_costs: Inclure ou non les coûts de transaction dans la visualisation.
        """
        if strategy_name not in self.results:
            raise ValueError(f"No results found for strategy '{strategy_name}'")
            
        self.results[strategy_name].plot(
            name_strat=strategy_name,
            backend=backend,
            include_costs=include_costs
        )
        
    def plot_all_strategies(self, backend: str = 'matplotlib', include_costs: bool = True) -> None:
        """
        Tracer les résultats pour toutes les stratégies.
        
        Args:
            backend: Backend de visualisation ('matplotlib', 'seaborn', ou 'plotly').
            include_costs: Inclure ou non les coûts de transaction dans la visualisation.
        """
        for name in self.results:
            self.plot_strategy(name, backend, include_costs)
            
    def compare_strategies(self, backend: str = 'matplotlib') -> None:
        """
        Comparer toutes les stratégies en utilisant la fonction compare_results.
        
        Args:
            backend: Backend de visualisation ('matplotlib', 'seaborn', ou 'plotly').
        """
        if not self.results:
            raise ValueError("No results available. Run backtests first.")
            
        fig = compare_results(
            list(self.results.values()),
            list(self.strategies.keys()),
            backend=backend
        )
        
        if backend == 'plotly':
            fig.show()
            
    def print_statistics(self, strategy_name: Union[str, None] = None) -> None:
        """
        Afficher les statistiques pour une stratégie ou pour toutes les stratégies.
        
        Args:
            strategy_name: Nom d'une stratégie spécifique, ou None pour toutes les stratégies.
        """
        stats = self.get_statistics(strategy_name)
        
        if isinstance(stats, dict):
            print(f"\nStatistics for strategy '{strategy_name}':")
            for key, value in stats.items():
                print(f"{key}: {value:.4f}")
        else:
            print("\nStatistics for all strategies:")
            print(stats.round(4))
