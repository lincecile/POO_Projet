from mypackage import Backtester, Result, Strategy, strategy, compare_results
import unittest
import pandas as pd
import numpy as np

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
