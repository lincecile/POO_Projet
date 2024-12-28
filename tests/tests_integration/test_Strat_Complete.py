import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mypackage import Result, compare_results, DataFileReader, Strategy, Backtester, Strategy_Manager

class SimpleTestStrategy(Strategy):
    """Stratégie simple d'exemple pour les tests"""
    def get_position(self, historical_data, current_position):
        # Retourne une position fixe de 0.5 pour chaque actif
        return {asset: 0.5 for asset in self.assets}

class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Préparation des données de test"""
        # Création d'un DataFrame de test
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Simulation de prix pour deux actifs
        data = {
            'AAPL': 100 * (1 + np.random.randn(len(dates)) * 0.02).cumprod(),
            'GOOGL': 150 * (1 + np.random.randn(len(dates)) * 0.02).cumprod()
        }
        cls.test_data = pd.DataFrame(data, index=dates)
        cls.test_data = cls.test_data.astype(float)  # Conversion explicite en float
        
        # Sauvegarde des données dans des fichiers temporaires pour tester le Reader
        cls.csv_path = Path('test_data.csv')
        cls.parquet_path = Path('test_data.parquet')
        
        # Correction: Sauvegarde en CSV avec le format de date correct et index nommé
        cls.test_data.index.name = 'Date'
        csv_data = cls.test_data.copy()
        csv_data.index = csv_data.index.strftime('%d/%m/%Y')
        csv_data.to_csv(cls.csv_path, index=True, date_format='%d/%m/%Y', sep=';')
        
        # Pour Parquet, gardons l'index datetime
        cls.test_data.to_parquet(cls.parquet_path)
        
        # Création d'un objet Result pour les tests avec des données numériques valides
        positions = pd.DataFrame({
            'AAPL': np.full(len(dates), 0.5),  # Utilisation de np.full pour garantir des valeurs numériques
            'GOOGL': np.full(len(dates), 0.5)
        }, index=dates)
        
        # S'assurer que les positions sont en float
        positions = positions.astype(float)
        
        trades = pd.DataFrame({
            'asset': ['AAPL', 'GOOGL', 'AAPL'],
            'from_pos': [0.0, 0.0, 0.5],
            'to_pos': [0.5, 0.5, 0.0],
            'cost': [0.001, 0.001, 0.001]
        }, index=[dates[0], dates[0], dates[-1]])
        
        # S'assurer que les colonnes numériques des trades sont en float
        for col in ['from_pos', 'to_pos', 'cost']:
            trades[col] = trades[col].astype(float)
        
        cls.test_result = Result(cls.test_data, positions, trades)

    def test_data_reader(self):
        """Test de la lecture des données"""
        reader = DataFileReader(date_format='%d/%m/%Y')
        
        # Test lecture CSV avec nom de colonne de date explicite
        csv_data = reader.read_file(self.csv_path, date_column='Date')
        self.assertIsInstance(csv_data, pd.DataFrame)
        self.assertEqual(len(csv_data.columns), len(self.test_data.columns))
        
        # Test lecture Parquet
        parquet_data = reader.read_file(self.parquet_path)
        self.assertIsInstance(parquet_data, pd.DataFrame)
        self.assertEqual(len(parquet_data.columns), len(self.test_data.columns))

    def test_compare_results(self):
        """Test de la fonction compare_results"""
        # Création d'une deuxième stratégie avec des positions différentes
        strategy2 = SimpleTestStrategy(
            rebalancing_frequency='W',
            assets=['AAPL', 'GOOGL']
        )
        
        # S'assurer que les données de test sont au bon format datetime et numériques
        test_data = self.test_data.copy()
        test_data = test_data.astype(float)
        
        backtester = Backtester(test_data)
        result2 = backtester.exec_backtest(strategy2)
        
        # Vérification que les returns sont numériques
        self.assertTrue(pd.api.types.is_numeric_dtype(result2.returns['portfolio']))
        self.assertTrue(not result2.returns.empty)
        self.assertTrue(not result2.returns['portfolio'].isna().all())
        
        # Vérifier les résultats de `self.test_result`
        self.assertTrue(pd.api.types.is_numeric_dtype(self.test_result.returns['portfolio']))
        self.assertTrue(not self.test_result.returns.empty)
        self.assertTrue(not self.test_result.returns['portfolio'].isna().all())
        
        # Dictionnaire de résultats avec des données valides
        results_dict = {
            "Strategy1": self.test_result,
            "Strategy2": result2
        }
        
        # Vérification supplémentaire des données numériques avant la comparaison
        for name, result in results_dict.items():
            self.assertTrue(isinstance(result.statistics['total_return'], (int, float)))
            self.assertTrue(isinstance(result.statistics['sharpe_ratio'], (int, float)))
            print(result.statistics)

        # Test avec différents backends
        for backend in ['matplotlib', 'seaborn', 'plotly']:
            fig = compare_results(results_dict, backend=backend, show_plot=False)
            self.assertIsNotNone(fig)

    def test_integration_workflow(self):
        """Test du workflow complet"""
        
        # 1. Lecture des données avec conversion explicite en float
        data = self.test_data.copy()
        data = data.astype(float)
        
        # 2. Création d'une stratégie
        strategy = SimpleTestStrategy(
            rebalancing_frequency='D',
            assets=['AAPL', 'GOOGL']
        )
        
        # 3. Configuration du gestionnaire
        manager = Strategy_Manager(data)
        manager.add_strategy(
            "Test Strategy",
            strategy,
            transaction_costs={'AAPL': 0.001, 'GOOGL': 0.002},
            slippage={'AAPL': 0.0005, 'GOOGL': 0.001}
        )
        
        # 4. Exécution du backtest
        manager.run_backtests()
        
        # 5. Vérification des résultats et de leur type numérique
        stats = manager.get_statistics("Test Strategy")
        self.assertIsInstance(stats, dict)
        self.assertTrue('total_return' in stats)
        self.assertTrue('sharpe_ratio' in stats)
        self.assertTrue(isinstance(stats['total_return'], (int, float)))
        self.assertTrue(isinstance(stats['sharpe_ratio'], (int, float)))
        
        # Vérification des résultats détaillés
        result = manager.results["Test Strategy"]
        self.assertIsInstance(result, Result)
        self.assertTrue(hasattr(result, 'returns'))
        self.assertTrue(hasattr(result, 'positions'))
        self.assertTrue(hasattr(result, 'trades'))
        
        # Vérifier que les returns contiennent des données numériques
        self.assertTrue(pd.api.types.is_numeric_dtype(result.returns['portfolio']))
        self.assertTrue(not result.returns.empty)
        self.assertTrue(not result.returns['portfolio'].isna().all())
        
        # 6. Test des visualisations
        manager.plot_strategy("Test Strategy", show_plot=False)
        manager.plot_all_strategies(show_plot=False)
        manager.compare_strategies(show_plot=False)

    @classmethod
    def tearDownClass(cls):
        """Nettoyage des fichiers de test"""
        if cls.csv_path.exists():
            cls.csv_path.unlink()
        if cls.parquet_path.exists():
            cls.parquet_path.unlink()

if __name__ == '__main__':
    unittest.main()