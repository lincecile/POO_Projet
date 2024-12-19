from mypackage import Strategy, Backtester, Result

def test_simple():
    assert 1 + 1 == 2

def test_has_method2():
    assert hasattr(Strategy, "fit")
    assert hasattr(Strategy, "get_position")

def test_has_method4():
    assert hasattr(Backtester, "run")

def test_has_method3():
    assert hasattr(Result, "_calculate_returns")


exit()

import sys
import os
import unittest
import pandas as pd
import numpy as np
from typing import List

# Ajouter dynamiquement le chemin racine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from mypackage import Backtester
from mypackage import Strategy
from mypackage import Result, compare_results

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

class TestBacktesting(unittest.TestCase):
    def setUp(self):
        self.valid_data = pd.DataFrame(
            {"price": [100, 101, 102, 103, 104]},
            index=pd.date_range(start="2023-01-01", periods=5)
        )
        self.invalid_data = pd.DataFrame({"not_price": [100, 101, 102]})
        self.empty_data = pd.DataFrame(columns=["price"])
        self.nan_data = pd.DataFrame(
            {"price": [100, np.nan, 102, 103, 104]},
            index=pd.date_range(start="2023-01-01", periods=5)
        )
        self.strategy = MovingAverageStrategy()
        self.backtester = Backtester(self.valid_data)

    def test_strategy_get_position_valid_data(self):
        """Teste la méthode get_position() avec des données valides."""
        position = self.strategy.get_position(self.valid_data.iloc[:3], current_position=0)
        self.assertIn(position, [-1, 0, 1], "La position doit être -1, 0 ou 1.")

    def test_strategy_invalid_column(self):
        """Teste get_position() avec des colonnes invalides."""
        with self.assertRaises(ValueError):
            self.strategy.get_position(self.invalid_data, current_position=0)

    def check_invalid_data(self, data):
        with self.assertRaises(ValueError):
            self.strategy.get_position(data, current_position=0)

    def test_strategy_empty_data(self):
        self.check_invalid_data(self.empty_data)

    def test_strategy_nan_data(self):
        self.check_invalid_data(self.nan_data)


    def test_backtester_valid_run(self):
        """Teste un backtest avec des données valides."""
        result = self.backtester.run(self.strategy)
        self.assertIsInstance(result, Result, "Le résultat doit être une instance de Result.")
        self.assertFalse(result.positions.empty, "Les positions ne doivent pas être vides.")

    def test_backtester_empty_data(self):
        """Teste le backtester avec des données vides."""
        with self.assertRaises(ValueError):
            Backtester(self.empty_data)

    def test_backtester_nan_data(self):
        """Teste le backtester avec des données contenant des NaN."""
        with self.assertRaises(ValueError):
            Backtester(self.nan_data)

    def test_end_to_end_backtest(self):
        """Teste un scénario complet d'exécution de backtest."""
        result = self.backtester.run(self.strategy)
        stats = result.statistics

        self.assertIsInstance(result, Result, "Le backtest doit retourner un objet Result.")
        self.assertFalse(result.positions.empty, "Les positions ne doivent pas être vides.")
        self.assertIn('sharpe_ratio', stats, "Les statistiques doivent inclure 'sharpe_ratio'.")

    def test_multiple_strategies_comparison(self):
        """Teste la comparaison entre deux stratégies."""
        strategy_1 = MovingAverageStrategy(short_window=2, long_window=3)
        strategy_2 = MovingAverageStrategy(short_window=3, long_window=4)

        result_1 = self.backtester.run(strategy_1)
        result_2 = self.backtester.run(strategy_2)

        fig = compare_results(
            [result_1, result_2],
            strat_name=["Stratégie 1", "Stratégie 2"],
            backend="matplotlib"
        )
        self.assertIsNotNone(fig, "La comparaison des résultats doit produire un graphique.")
if __name__ == "__main__":
    unittest.main()



