
import unittest
import pandas as pd
import numpy as np

from mypackage import Result, compare_results

import unittest
import pandas as pd
import numpy as np

from mypackage import Backtester, Result, Strategy, strategy

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

    
if __name__ == "__main__":
    unittest.main()



