from mypackage import Backtester, Result, Strategy, strategy, compare_results
import unittest
import pandas as pd
import numpy as np

def test_has_method_consigne():
    assert hasattr(Strategy, "fit")
    assert hasattr(Strategy, "get_position")

def test_has_method4():
    assert hasattr(Backtester, "run")

def test_has_method3():
    assert hasattr(Result, "_calculate_returns")

### OUMAIMA 
import unittest
import pandas as pd
import numpy as np

from mypackage import Result, compare_results
from mypackage import MCOBasedStrategy


class MovingAverageStrategy(Strategy):
    """Stratégie simple basée sur un croisement de moyennes mobiles."""
    def __init__(self, short_window=2, long_window=3, target_column='price'):
        super().__init__(rebalancing_frequency='D')
        self.short_window = short_window
        self.long_window = long_window
        self.target_column = target_column

    def validate_data(self, data: pd.DataFrame):
        if data.empty:
            raise ValueError("Les données sont vides.")
        if self.target_column not in data.columns:
            raise KeyError(f"La colonne '{self.target_column}' est manquante.")
        if data.isnull().any().any():
            raise ValueError("Les données contiennent des valeurs nulles.")


    def get_position(self, historical_data: pd.DataFrame, current_position: float) -> float:
        """Calcule la position basée sur un croisement de moyennes mobiles."""
        self.validate_data(historical_data)
        if len(historical_data) < self.long_window:
            return 0  # Pas assez de données

        short_ma = historical_data[self.target_column].rolling(self.short_window).mean()
        long_ma = historical_data[self.target_column].rolling(self.long_window).mean()
        if short_ma.iloc[-1] > long_ma.iloc[-1]:
            return 1  # Long
        elif short_ma.iloc[-1] < long_ma.iloc[-1]:
            return -1  # Short
        return 0  # Neutre
    

class TestMCOBasedStrategy(unittest.TestCase):

    def setUp(self):
        # Données simulées pour le test
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        self.data = pd.DataFrame({
            'price': np.linspace(100, 110, len(dates))  # Une montée linéaire des prix
        }, index=dates)
        
        self.mco_strategy = MCOBasedStrategy(threshold=0.05)
        self.initial_cost = 100  # Coût moyen initial
        self.mco_strategy.fit(self.data, initial_position_cost=self.initial_cost)

    def test_fit_method(self):
        # Vérifier que le coût moyen est bien initialisé
        self.assertEqual(self.mco_strategy.average_cost, self.initial_cost)
    
    def test_get_position_buy_signal(self):
        # Simuler un prix inférieur au coût moyen
        low_price_data = self.data.copy()
        low_price_data.loc[low_price_data.index[-1], 'price'] = 95  # Dernier prix inférieur au coût moyen

        position = self.mco_strategy.get_position(low_price_data, current_position=0)
        self.assertEqual(position, 1)  # Signal d'achat attendu

    def test_get_position_sell_signal(self):
        # Simuler un prix supérieur au coût moyen
        high_price_data = self.data.copy()
        high_price_data.loc[high_price_data.index[-1], 'price'] = 120  # Dernier prix supérieur au coût moyen

        position = self.mco_strategy.get_position(high_price_data, current_position=0)
        self.assertEqual(position, -1)  # Signal de vente attendu

    def test_get_position_neutral_signal(self):
        # Simuler un prix proche du coût moyen
        neutral_price_data = self.data.copy()
        neutral_price_data.loc[neutral_price_data.index[-1], 'price'] = 101  # Proche du coût moyen

        position = self.mco_strategy.get_position(neutral_price_data, current_position=0)
        self.assertEqual(position, 0)  # Signal neutre attendu

class TestMCOIntegrationWithBacktester(unittest.TestCase):

    def setUp(self):
        # Données simulées pour le test
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        self.data = pd.DataFrame({
            'price': np.random.randn(len(dates)).cumsum() + 100  # Série temporelle générée aléatoirement
        }, index=dates)

        self.mco_strategy = MCOBasedStrategy(threshold=0.05)
        self.mco_strategy.fit(self.data, initial_position_cost=100)  # Initialiser avec un coût moyen

        self.backtester = Backtester(self.data)

    def test_backtester_with_mco_strategy(self):
        # Exécuter la stratégie avec le backtester
        result = self.backtester.run(self.mco_strategy)

        # Vérifier que le résultat est de la bonne classe
        self.assertIsInstance(result, Result)
        self.assertTrue(hasattr(result, 'positions'))
        self.assertTrue(hasattr(result, 'trades'))
        self.assertTrue(hasattr(result, 'returns'))

    def test_mco_strategy_trading_logic(self):
        # Exécuter la stratégie pour des données spécifiques
        modified_data = self.data.copy()
        modified_data.loc[modified_data.index[-1], 'price'] = 120  # Simuler une montée des prix
        result = self.backtester.run(self.mco_strategy)

        # Vérifier que la stratégie a émis au moins une transaction
        self.assertFalse(result.trades.empty)


class TestBacktesting(unittest.TestCase):
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

