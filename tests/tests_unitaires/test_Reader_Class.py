
from mypackage import DataFileReader
import unittest
import pandas as pd
import numpy as np
import tempfile
import os

class TestDataFileReader(unittest.TestCase):
    def setUp(self):
        self.reader = DataFileReader()
        self.temp_dir = tempfile.mkdtemp()
        
        # Création de dates au format attendu par le reader
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        # Conversion des dates au format attendu par le reader
        formatted_dates = [d.strftime('%d/%m/%Y') for d in dates]
        
        # Création du DataFrame avec les dates au bon format
        self.sample_data = pd.DataFrame({
            'date': formatted_dates,
            'price': np.random.randn(len(dates)).cumsum() + 100
        })
        
    def tearDown(self):
        """Nettoie l'environnement de test"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_read_csv(self):
        """Test la lecture de fichiers CSV"""
        # Création d'un fichier CSV temporaire
        csv_path = os.path.join(self.temp_dir, 'test.csv')
        # Sauvegarde avec l'encodage UTF-8 explicite
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

    def test_all_nan_data(self):
        """Test la gestion des fichiers ne contenant que des NaN"""
        nan_data = self.sample_data.copy()
        nan_data.loc[:, nan_data.columns != 'date'] = np.nan
        
        for fmt in ['csv', 'parquet']:
            with self.subTest(format=fmt):
                file_path = os.path.join(self.temp_dir, f'nan.{fmt}')
                if fmt == 'csv':
                    nan_data.to_csv(file_path, sep=';', index=False)
                else:
                    nan_data.to_parquet(file_path)
                
                with self.assertRaises(ValueError) as context:
                    self.reader.read_file(file_path, date_column='date')
                self.assertIn("nan", str(context.exception).lower())


if __name__ == '__main__':
    unittest.main()
