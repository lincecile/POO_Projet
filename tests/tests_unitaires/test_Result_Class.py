import matplotlib
matplotlib.use('Agg')  # backend non-interactif qui ne nécessite pas de serveur X ou d'interface graphique
import matplotlib.pyplot as plt
from mypackage import Result
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
            'asset1': np.linspace(100, 110, len(dates)),  # Données linéaires croissantes
            'asset2': np.linspace(100, 105, len(dates))   # Données linéaires croissantes différentes
        }, index=dates)
        
        self.positions = pd.DataFrame({
            'asset1': [1] * len(dates),
            'asset2': [1] * len(dates)
        }, index=dates)
        
        # Créer des trades fictifs
        self.trades = pd.DataFrame({
            'asset': ['asset1', 'asset2'] * 2,
            'from_pos': [0, 0, 1, 1],
            'to_pos': [1, 1, 0, 0],
            'cost': [0.001] * 4
        }, index=dates[:4])

        self.result = Result(self.data, self.positions, self.trades)
    
    # Vérification que la classe Result possède les bons attributs avec le bon type
    def test_initialization(self):
        self.assertIsInstance(self.result.data, pd.DataFrame)
        self.assertIsInstance(self.result.positions, pd.DataFrame)
        self.assertIsInstance(self.result.trades, pd.DataFrame)
        self.assertIsInstance(self.result.returns, pd.DataFrame)
        self.assertIsInstance(self.result.returns_no_cost, pd.DataFrame)
        self.assertIsInstance(self.result.statistics, dict)
        self.assertIsInstance(self.result.statistics_each_asset, dict)

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
        self.assertIsInstance(fig_mpl, plt.Figure)
        
        # Test seaborn 
        fig_sns = self.result.plot("test_strat", backend='seaborn')
        self.assertIsInstance(fig_sns, plt.Figure)
        
        # Test plotly 
        fig_plotly = self.result.plot("test_strat", backend='plotly')
        self.assertIsNotNone(fig_plotly)


if __name__ == '__main__':
    unittest.main()

