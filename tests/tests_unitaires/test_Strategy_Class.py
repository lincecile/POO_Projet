from mypackage import Strategy, strategy
import unittest
import pandas as pd
import numpy as np

def test_has_method_consigne():
    assert hasattr(Strategy, "fit")
    assert hasattr(Strategy, "get_position")

class TestStrategy(unittest.TestCase):

    # Création data fictive
    def setUp(self):
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        self.sample_data = pd.DataFrame({
            'asset1': np.random.randn(len(dates)).cumsum()+100,
            'asset2': np.random.randn(len(dates)).cumsum()+100
        }, index=dates)
        
    # Vérification que le décorateur fonctionne
    def test_strategy_decorator(self):
        @strategy
        def simple_strategy(historical_data, current_position, assets=None, rebalancing_frequency='D'):
            assets = assets or historical_data.columns  # Use data columns if no assets specified
            return {asset: 1 for asset in assets}
        
        strat = simple_strategy(assets=['asset1', 'asset2'])
        self.assertEqual(strat.rebalancing_frequency, 'D')
        positions = strat.get_position(self.sample_data, {'asset1': 0, 'asset2': 0})
        self.assertEqual(positions, {'asset1': 1, 'asset2': 1})

    # Vérification que la classe Strategy est bien une classe abstraite
    def test_abstract_strategy(self):
        with self.assertRaises(TypeError):
            Strategy()
    
    # Vérification qu'on puisse utiliser l'héritage de la classe Strategy
    def test_custom_strategy(self):
        class CustomStrategy(Strategy):
            def get_position(self, historical_data, current_position):
                return -1
                
        strat = CustomStrategy()
        self.assertEqual(strat.get_position(self.sample_data, 0), -1, "La position doit être -1.")

if __name__ == '__main__':
    unittest.main()

