from mypackage import Backtester, Result, Strategy
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
            'asset1': np.random.randn(len(dates)).cumsum(),
            'asset2': np.random.randn(len(dates)).cumsum()+100
        }, index=dates)

        # Création stratégie test
        class FakeStrategy(Strategy):
            def __init__(self):
                super().__init__(rebalancing_frequency='D', assets=['asset1', 'asset2'])
            
            def get_position(self, historical_data, current_position):
                return {asset: 1 for asset in self.assets}
                
        self.strategy = FakeStrategy()
        self.backtester = Backtester(self.sample_data)
    
    # Vérification que la classe Backtester fonctionne avec les valeurs par défaut
    def test_initialization(self):
        self.assertEqual(self.backtester.transaction_costs, {'asset1': 0.001,'asset2': 0.001}, "Le coût de transaction par défaut doit être 0.001 pour chaque actif.")
        self.assertEqual(self.backtester.slippage, {'asset1': 0.0005, 'asset2': 0.0005}, "Le slippage par défaut doit être 0.0005 pour chaque actif.")

    # Vérification que la classe Backtester fonctionne avec les valeurs de l'utilisateur
    def test_initialization_with_custom_costs(self):
        custom_costs = {'asset1': 0.002, 'asset2': 0.003}
        custom_slippage = {'asset1': 0.001, 'asset2': 0.0015}
        backtester = Backtester(
            self.sample_data, 
            transaction_costs=custom_costs, 
            slippage=custom_slippage
        )
        
        self.assertEqual(backtester.transaction_costs, custom_costs)
        self.assertEqual(backtester.slippage, custom_slippage)

    # Vérification que la classe Backtester renvoie bien un objet de la classe Result 
    def test_run_backtest(self):
        result = self.backtester.exec_backtest(self.strategy)
        
        self.assertIsInstance(result, Result, "Le résultat doit être une instance de Result.")
        self.assertTrue(hasattr(result, 'positions'), "L'objet doit contenir l'attribut 'positions'.")
        self.assertTrue(hasattr(result, 'trades'), "L'objet doit contenir l'attribut 'trades'.")
        self.assertTrue(hasattr(result, 'returns'), "L'objet doit contenir l'attribut 'returns'.")
        self.assertTrue(hasattr(result, 'data'), "L'objet doit contenir l'attribut 'data'.")
        self.assertTrue(hasattr(result, 'statistics'), "L'objet doit contenir l'attribut 'statistics'.")

    # Vérification que les coûts de transaction et de slippage sont pris en compte 
    def test_transaction_costs(self):
        backtester = Backtester(self.sample_data, transaction_costs=0.01, slippage=0.01)
        result = backtester.exec_backtest(self.strategy)
        
        if not result.trades.empty:
            self.assertTrue(all(result.trades['cost'] >= 0), "Dans ce cas, les coûts de transaction doivent être positifs pour chaque transaction.")

if __name__ == '__main__':
    unittest.main()

