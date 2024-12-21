from mypackage import Backtester, Result, Strategy, strategy, compare_results
import unittest
import pandas as pd
import numpy as np

# Vérification que la classe possède une méthode pour exécuter le backtest
def test_has_method():
    assert hasattr(Backtester, "exec_backtest")

class TestBacktester(unittest.TestCase):

    # Création data fictive
    def setUp(self):
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        self.sample_data = pd.DataFrame({
            'price': np.random.randn(len(dates)).cumsum() + 100
        }, index=dates)
        # self.invalid_data = pd.DataFrame({"not_price": [100, 101, 102]})
        self.empty_data = pd.DataFrame(columns=["price"])
        self.nan_data = pd.DataFrame(
            {"price": [100, np.nan, 102, 103, 104]},
            index=pd.date_range(start="2023-01-01", periods=5)
        )

        # Création stratégie test
        class FakeStrategy(Strategy):
            def get_position(self, historical_data, current_position):
                return 1 if len(historical_data) != 0 else -1
                
        self.strategy = FakeStrategy()
        self.backtester = Backtester(self.sample_data)
    
    # Vérification que la classe Backtester fonctionne avec les valeurs par défaut
    def test_initialization(self):
        self.assertEqual(self.backtester.transaction_costs, 0.001)
        self.assertEqual(self.backtester.slippage, 0.0005)
        pd.testing.assert_frame_equal(self.backtester.data, self.sample_data)

    # Vérification que la classe Backtester renvoie bien un objet de la classe Result 
    def test_run_backtest(self):
        result = self.backtester.exec_backtest(self.strategy)
        
        self.assertIsInstance(result, Result, "Le résultat doit être une instance de Result.")
        self.assertTrue(hasattr(result, 'positions'))
        self.assertTrue(hasattr(result, 'trades'))
        self.assertTrue(hasattr(result, 'returns'))
    
    # Vérification que les coûts de transaction et de slippage sont pris en compte 
    def test_transaction_costs(self):
        backtester = Backtester(self.sample_data, transaction_costs=0.01, slippage=0.01)
        result = backtester.exec_backtest(self.strategy)
        
        if not result.trades.empty:
            self.assertTrue(all(result.trades['cost'] > 0))

if __name__ == '__main__':
    unittest.main()

