from mypackage import Result, compare_results
import unittest
import pandas as pd
import numpy as np
import matplotlib.figure
import plotly.graph_objects as go
import seaborn.axisgrid as sns_axisgrid

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
        self.assertIsNotNone(fig_mpl, "La fonction 'compare_results' avec le backend 'matplotlib' ne doit pas retourner None.")
        self.assertIsInstance(fig_mpl, matplotlib.figure.Figure, "Le backend 'matplotlib' doit retourner un objet de type 'matplotlib.figure.Figure'.")

        fig_sns = compare_results([result1, result2], ['Strategy1', 'Strategy2'], backend='seaborn')
        self.assertIsInstance(fig_sns, sns_axisgrid.PairGrid, "Le backend 'matplotlib' doit retourner un objet de type 'matplotlib.figure.Figure'.")

        fig_plotly = compare_results([result1, result2], ['Strategy1', 'Strategy2'], backend='plotly')
        self.assertIsInstance(fig_plotly, go.Figure, "Le backend 'plotly' doit retourner un objet de type 'plotly.graph_objects.Figure'.")

if __name__ == '__main__':
    unittest.main()
