from mypackage import Backtester, Result, Strategy, strategy, compare_results,MCOBasedStrategy
import unittest
import pandas as pd
import numpy as np


def validate_data(self, data: pd.DataFrame):
    if data.empty:
        raise ValueError("Les données sont vides.")
    if self.target_column not in data.columns:
        raise KeyError(f"La colonne '{self.target_column}' est manquante.")
    if data.isnull().any().any():
        raise ValueError("Les données contiennent des valeurs nulles.")

    

class TestBacktesting(unittest.TestCase):

    def test_strategy_invalid_column(self):
        """Teste get_position() avec des colonnes invalides."""
        with self.assertRaises(ValueError):
            self.strategy.get_position(self.invalid_data, current_position=0)

    def test_backtester_valid_run(self):
        """Teste un backtest avec des données valides."""
        result = self.backtester.run(self.strategy)
        self.assertFalse(result.positions.empty, "Les positions ne doivent pas être vides.")

    def test_backtester_empty_data(self):
        """Teste le backtester avec des données vides."""
        with self.assertRaises(ValueError):
            Backtester(self.empty_data)

    def test_backtester_nan_data(self):
        """Teste le backtester avec des données contenant des NaN."""
        with self.assertRaises(ValueError):
            Backtester(self.nan_data)