
from mypackage import Strategy_Manager, Strategy, Backtester, compare_results, strategy, DataFileReader
import unittest
import pandas as pd
import numpy as np
import tempfile
import os

class TestDataFileReader(unittest.TestCase):
    def setUp(self):
        self.reader = DataFileReader()
        self.temp_dir = tempfile.mkdtemp()
        
        # Création de données fictives
        self.dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        self.sample_data = pd.DataFrame({'date': self.dates, 'price': np.random.randn(len(self.dates)).cumsum() + 100})

    def tearDown(self):
        # Nettoyage des fichiers temporaires
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_read_csv(self):
        """Test la lecture de fichiers CSV"""
        # Création d'un fichier CSV temporaire
        csv_path = os.path.join(self.temp_dir, 'test.csv')
        self.sample_data.to_csv(csv_path, sep=';', index=False)
        
        # Test de lecture
        data = self.reader.read_file(csv_path, date_column='date')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIsInstance(data.index, pd.DatetimeIndex)

    def test_read_parquet(self):
        """Test la lecture de fichiers Parquet"""
        # Création d'un fichier Parquet temporaire
        parquet_path = os.path.join(self.temp_dir, 'test.parquet')
        self.sample_data.to_parquet(parquet_path)
        
        # Test de lecture
        data = self.reader.read_file(parquet_path, date_column='date')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIsInstance(data.index, pd.DatetimeIndex)

    def test_file_not_found(self):
        """Test la gestion des fichiers non trouvés"""
        with self.assertRaises(FileNotFoundError):
            self.reader.read_file('nonexistent.csv')

    def test_unsupported_format(self):
        """Test la gestion des formats non supportés"""
        unsupported_path = os.path.join(self.temp_dir, 'test.txt')
        with open(unsupported_path, 'w') as f:
            f.write('test')
            
        with self.assertRaises(ValueError):
            self.reader.read_file(unsupported_path)

    def test_detect_date_column(self):
        """Test la détection automatique de la colonne de date"""
        csv_path = os.path.join(self.temp_dir, 'test.csv')
        self.sample_data.to_csv(csv_path, sep=';', index=False)
        
        # Test sans spécifier la colonne de date
        data = self.reader.read_file(csv_path)
        self.assertIsInstance(data.index, pd.DatetimeIndex)

    def test_all_nan_data(self):
        """Test la gestion des fichiers ne contenant que des NaN"""
        # Création d'un DataFrame avec uniquement des NaN
        nan_data = pd.DataFrame({'date': self.dates,'price': [np.nan] * len(self.dates)})
        
        # Test avec un fichier CSV contenant que des NaN
        nan_csv_path = os.path.join(self.temp_dir, 'nan.csv')
        nan_data.to_csv(nan_csv_path, sep=';', index=False)
        
        with self.assertRaises(ValueError) as context:
            self.reader.read_file(nan_csv_path, date_column='date')
        self.assertIn("ne contient que des valeurs NaN", str(context.exception))
        
        # Test avec un fichier Parquet contenant que des NaN
        nan_parquet_path = os.path.join(self.temp_dir, 'nan.parquet')
        nan_data.to_parquet(nan_parquet_path)
        
        with self.assertRaises(ValueError) as context:
            self.reader.read_file(nan_parquet_path, date_column='date')
        self.assertIn("ne contient que des valeurs NaN", str(context.exception))


if __name__ == '__main__':
    unittest.main()
