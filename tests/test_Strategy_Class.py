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
            'price': np.random.randn(len(dates)).cumsum() + 10
        }, index=dates)
        # self.invalid_data = pd.DataFrame({"not_price": [100, 101, 102]}) pas utile?
        self.empty_data = pd.DataFrame(columns=["price"])
        self.nan_data = pd.DataFrame(
            {"price": [100, np.nan, 102, 103, 104]},
            index=pd.date_range(start="2023-01-01", periods=5)
        )

    # Vérification que les NaN dans la data sont gérés ?

    
    # Vérification que le décorateur fonctionne
    def test_strategy_decorator(self):
        @strategy
        def simple_strategy(historical_data, current_position, rebalancing_frequency='D'):
            return 1
            
        strat = simple_strategy()
        self.assertEqual(strat.rebalancing_frequency, 'D')
        self.assertEqual(strat.get_position(self.sample_data, 0), 1, "La position doit être 1.")
    
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

