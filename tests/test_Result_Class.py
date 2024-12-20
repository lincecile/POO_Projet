from mypackage import Backtester, Result, Strategy, strategy, compare_results
import unittest
import pandas as pd
import numpy as np

def test_has_method3():
    assert hasattr(Result, "_calculate_returns")


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
        self.assertIsNotNone(fig_mpl, "La comparaison des résultats doit produire un graphique.")
        
        # Test seaborn 
        fig_sns = self.result.plot("test_strat", backend='seaborn')
        self.assertIsNotNone(fig_sns, "La comparaison des résultats doit produire un graphique.")
        
        # Test plotly 
        fig_plotly = self.result.plot("test_strat", backend='plotly')
        self.assertIsNotNone(fig_plotly, "La comparaison des résultats doit produire un graphique.")
    
 
if __name__ == '__main__':
    unittest.main()

