import pandas as pd
from typing import Union, Optional
from pathlib import Path

class DataFileReader:
    """Une classe pour gérer la lecture de fichiers de données dans différents formats (CSV, Parquet)."""
    
    def __init__(self, date_format: str = '%d/%m/%Y'):
        """
        Initialiser le DataFileReader.
        
        Args:
            date_column: Nom de la colonne contenant les dates.
            date_format: Format des dates dans les fichiers CSV.
        """
        self.date_format = date_format
    
    def read_file(self, filepath: Union[str, Path], date_column: Optional[str] = None) -> pd.DataFrame:
        """
        Lire un fichier de données (CSV ou Parquet) et retourner un DataFrame correctement formaté.
        
        Args:
            filepath: Chemin vers le fichier de données.
            
        Returns:
            pd.DataFrame: DataFrame traité avec les dates comme index.
            
        Raises:
            ValueError: Si le fichier ne peut pas être lu ou s'il est dans un format non supporté.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Le fichier {filepath} n'existe pas.")
            
        try:
            if filepath.suffix.lower() == '.csv':
                return self._read_csv(filepath, date_column)
            elif filepath.suffix.lower() == '.parquet':
                return self._read_parquet(filepath, date_column)
            else:
                raise ValueError(f"Format de fichier non supporté: {filepath.suffix}")
        except Exception as e:
            raise ValueError(f"Impossible de lire le fichier {filepath}: {str(e)}")
    
    def _detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Trouve la colonne de date"""
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                return col
            except:
                continue
        return None
    
    def _read_csv(self, filepath: Path, date_column: Optional[str] = None) -> pd.DataFrame:
        """Lire un fichier CSV."""
        try:
            # Try reading with semicolon separator and replace commas in numbers
            data = pd.read_csv(filepath, sep=';').replace(',', '.', regex=True)
            
            # If date_column is not specified, try to detect it or use first column
            if date_column is None:
                date_column = self._detect_date_column(data)
                if date_column is None:
                    date_column = data.columns[0]
            
            # Convert date column
            try:
                data[date_column] = pd.to_datetime(data[date_column], format=self.date_format)
            except ValueError:
                # If specific format fails, try automatic parsing
                data[date_column] = pd.to_datetime(data[date_column])
            
            # Set date as index and convert to float
            data.set_index(date_column, inplace=True)
            
            # Convert all remaining columns to float
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            return data
            
        except Exception as e:
            raise ValueError(f"Erreur lors de la lecture du CSV: {str(e)}")
    
    def _read_parquet(self, filepath: Path, date_column: Optional[str] = None) -> pd.DataFrame:
        """Lire un fichier Parquet."""
        try:
            data = pd.read_parquet(filepath)
            
            # If date_column is not specified and the index is not already a datetime
            if date_column is not None and not isinstance(data.index, pd.DatetimeIndex):
                if date_column in data.columns:
                    data.set_index(date_column, inplace=True)
                    data.index = pd.to_datetime(data.index)
            
            # Convert all columns to float except datetime columns
            for col in data.columns:
                if not pd.api.types.is_datetime64_any_dtype(data[col]):
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            return data
            
        except Exception as e:
            raise ValueError(f"Erreur lors de la lecture du Parquet: {str(e)}")