from mypackage import Strategy_Manager, Strategy
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TestStrategyManager(unittest.TestCase):
    def setUp(self):
        # Création de données fictives
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        self.sample_data = pd.DataFrame({
            'asset1': np.linspace(100, 110, len(dates)),  # Données linéaires croissantes
            'asset2': np.linspace(100, 105, len(dates))   # Données linéaires croissantes différentes
        }, index=dates)

        # Création d'une stratégie de test
        class FakeStrategy(Strategy):
            def __init__(self, rebalancing_frequency='D'):
                super().__init__(rebalancing_frequency=rebalancing_frequency, assets=['asset1','asset2'])
                
            def get_position(self, historical_data, current_position):
                return {'asset1': 1, 'asset2':1}

        self.strategy1 = FakeStrategy()
        self.strategy2 = FakeStrategy()
        
        # Initialisation du Strategy_Manager
        self.manager = Strategy_Manager(self.sample_data)

    def test_initialization(self):
        """Test l'initialisation du Strategy_Manager"""
        self.assertIsInstance(self.manager.data, pd.DataFrame)
        self.assertIsInstance(self.manager.strategies, dict)
        self.assertIsInstance(self.manager.results, dict)

    def test_add_strategy(self):
        """Test l'ajout d'une stratégie"""
        self.manager.add_strategy("test_strat", self.strategy1)
        self.assertIn("test_strat", self.manager.strategies)
        
        # Test d'ajout d'une stratégie avec le même nom
        with self.assertRaises(ValueError):
            self.manager.add_strategy("test_strat", self.strategy2)

    def test_remove_strategy(self):
        """Test la suppression d'une stratégie"""
        self.manager.add_strategy("test_strat", self.strategy1)
        self.manager.remove_strategy("test_strat")
        self.assertNotIn("test_strat", self.manager.strategies)

    def test_run_backtests(self):
        """Test l'exécution des backtests"""
        self.manager.add_strategy("test_strat", self.strategy1)
        self.manager.run_backtests()
        self.assertIn("test_strat", self.manager.results)

    def test_get_statistics(self):
        """Test l'obtention des statistiques"""
        self.manager.add_strategy("test_strat", self.strategy1)
        self.manager.run_backtests()
        
        # Test pour une stratégie spécifique
        stats = self.manager.get_statistics("test_strat")
        self.assertIsInstance(stats, dict)
        
        # Test pour toutes les stratégies
        all_stats = self.manager.get_statistics()
        self.assertIsInstance(all_stats, pd.DataFrame)

    def test_plot_methods(self):
        """Test les méthodes de plotting"""
        self.manager.add_strategy("test_strat", self.strategy1)
        self.manager.run_backtests()

        # Test plot_strategy
        self.manager.plot_strategy("test_strat", backend='matplotlib', show_plot=False)

        # Test plot_all_strategies
        self.manager.plot_all_strategies(backend='matplotlib', show_plot=False)

        # Test compare_strategies
        self.manager.compare_strategies(backend='matplotlib', show_plot=False)

if __name__ == '__main__':
    unittest.main()
