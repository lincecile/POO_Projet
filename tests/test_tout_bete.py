from mypackage import Backtester, Result, Strategy, strategy, compare_results
import unittest
import pandas as pd
import numpy as np

def test_simple():
    assert 1 + 1 == 2

def test_has_method2():
    assert hasattr(Strategy, "fit")
    assert hasattr(Strategy, "get_position")

def test_has_method4():
    assert hasattr(Backtester, "run")

def test_has_method3():
    assert hasattr(Result, "_calculate_returns")

class TestStrategy(unittest.TestCase):

    # Création data fictive
    def setUp(self):
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        self.sample_data = pd.DataFrame({
            'price': np.random.randn(len(dates)).cumsum() + 10
        }, index=dates)
    
    # Vérification que le décorateur fonctionne
    def test_strategy_decorator(self):
        @strategy
        def simple_strategy(historical_data, current_position, rebalancing_frequency='D'):
            return 1
            
        strat = simple_strategy()
        self.assertEqual(strat.rebalancing_frequency, 'D')
        self.assertEqual(strat.get_position(self.sample_data, 0), 1)
    
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
        self.assertEqual(strat.get_position(self.sample_data, 0), -1)
        
class TestBacktester(unittest.TestCase):

    # Création data fictive
    def setUp(self):
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        self.sample_data = pd.DataFrame({
            'price': np.random.randn(len(dates)).cumsum() + 100
        }, index=dates)
        
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
        result = self.backtester.run(self.strategy)
        
        self.assertIsInstance(result, Result)
        self.assertTrue(hasattr(result, 'positions'))
        self.assertTrue(hasattr(result, 'trades'))
        self.assertTrue(hasattr(result, 'returns'))
    
    # Vérification que les coûts de transaction et de slippage sont pris en compte 
    def test_transaction_costs(self):
        backtester = Backtester(self.sample_data, transaction_costs=0.01, slippage=0.01)
        result = backtester.run(self.strategy)
        
        if not result.trades.empty:
            self.assertTrue(all(result.trades['cost'] > 0))

class TestResult(unittest.TestCase):

    # Création data fictive
    def setUp(self):
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        self.data = pd.DataFrame({
            'price': np.random.randn(len(dates)).cumsum() + 100
        }, index=dates)
        
        self.positions = pd.DataFrame({
            'position': [1] * len(dates)
        }, index=dates)
        
        self.trades = pd.DataFrame({
            'from_pos': [0, 1, -1],
            'to_pos': [1, -1, 0],
            'cost': [0.001] * 3
        }, index=dates[:3])
        
        self.result = Result(self.data, self.positions, self.trades)
    
    # Vérification que la classe Result possède les bons attributs 
    def test_initialization(self):
        self.assertTrue(hasattr(self.result, 'returns'))
        self.assertTrue(hasattr(self.result, 'statistics'))
    
    # Vérification que les metrics sont bien calculés 
    def test_calculate_statistics(self):
        stats = self.result.statistics
        
        required_stats = [
            'total_return', 'annual_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'sortino_ratio', 'VaR_95%', 'CVaR_95%',
            'Profit/Loss_Ratio', 'num_trades', 'win_rate'
        ]
        
        for stat in required_stats:
            self.assertIn(stat, stats)
    
    # Vérification que les options de plotting fonctionnent 
    def test_plotting_backends(self):

        # Test matplotlib 
        fig_mpl = self.result.plot("test_strat", backend='matplotlib')
        self.assertIsNotNone(fig_mpl)
        
        # Test seaborn 
        fig_sns = self.result.plot("test_strat", backend='seaborn')
        self.assertIsNotNone(fig_sns)
        
        # Test plotly 
        fig_plotly = self.result.plot("test_strat", backend='plotly')
        self.assertIsNotNone(fig_plotly)
        
class TestCompareResults(unittest.TestCase):

    # Vérification que de la possibilité de comparaison entre stratégies
    def test_compare_results_different_backends(self):

        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        data = pd.DataFrame({'price': np.random.randn(len(dates)).cumsum() + 100}, index=dates)
        positions = pd.DataFrame({'position': [1] * len(dates)}, index=dates)
        trades = pd.DataFrame()
        
        result1 = Result(data, positions, trades)
        result2 = Result(data, positions, trades)
        
        # Test different backends
        fig_mpl = compare_results([result1, result2], ['Strategy1', 'Strategy2'], backend='matplotlib')
        self.assertIsNotNone(fig_mpl)
        
        fig_sns = compare_results([result1, result2], ['Strategy1', 'Strategy2'], backend='seaborn')
        self.assertIsNotNone(fig_sns)
        
        fig_plotly = compare_results([result1, result2], ['Strategy1', 'Strategy2'], backend='plotly')
        self.assertIsNotNone(fig_plotly)

if __name__ == '__main__':
    unittest.main()

