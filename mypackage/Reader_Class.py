import pandas as pd
from typing import Union, Optional
from pathlib import Path

class DataFileReader:
    """Une classe pour gérer la lecture de fichiers de données dans différents formats (CSV, Parquet)."""
    
    def __init__(self, date_format: str = '%d/%m/%Y'):
        """
        Initialisation.
        
        Args:
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
        
        # Vérifie que le fichier existe
        if not filepath.exists():
            raise FileNotFoundError(f"Le fichier {filepath} n'existe pas.")
        
        try:
            # Teste si c'est un fichier csv ou un parquet
            if filepath.suffix.lower() == '.csv':
                data = self._read_csv(filepath, date_column)
            elif filepath.suffix.lower() == '.parquet':
                data = self._read_parquet(filepath, date_column)
            else:
                raise ValueError(f"Format de fichier non supporté: {filepath.suffix}")
            
            # Vérifie si le DataFrame ne contient que des NaN
            if data.shape[0] == 0 or data.isna().all().all():
                raise ValueError(f"Le fichier {filepath} ne contient que des valeurs NaN ou est vide après le traitement.")
            
            return data
        
        except Exception as e:
            raise ValueError(f"Impossible de lire le fichier {filepath}: {str(e)}")
    
    def _read_csv(self, filepath: Path, date_column: Optional[str] = None) -> pd.DataFrame:
        """Lire un fichier CSV."""
        try:
            data = pd.read_csv(filepath, sep=';').replace(',', '.', regex=True)
            
            # Si la colonne de date n'est pas indiqué
            if date_column is None:
                raise ValueError(f"Il faut indiquer une colonne de date dans le fichier")
            
            # Formattage de la colonne de date
            try:
                data[date_column] = pd.to_datetime(data[date_column], format=self.date_format)
            except Exception as e:
                # Si le format des dates indiqué par l'utilisateur n'est pas correct
                raise ValueError(f"Le format de date indiqué n'est pas le bon")
            
            # Les dates sont les indices du dataframe
            data.set_index(date_column, inplace=True)
            
            # Conversion des colonnes en float
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            return data
            
        except Exception as e:
            raise ValueError(f"Erreur lors de la lecture du CSV: {str(e)}")
    
    def _read_parquet(self, filepath: Path, date_column: Optional[str] = None) -> pd.DataFrame:
        """Lire un fichier Parquet."""
        try:
            data = pd.read_parquet(filepath)
            
            # Si la colonne de date n'est pas indiqué et que les indices ne sont pas des dates
            if date_column is not None and not isinstance(data.index, pd.DatetimeIndex):
                if date_column in data.columns:
                    data.set_index(date_column, inplace=True)
                    data.index = pd.to_datetime(data.index)
            
            # Conversion des colonnes en float
            for col in data.columns:
                if not pd.api.types.is_datetime64_any_dtype(data[col]):
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            return data
            
        except Exception as e:
            raise ValueError(f"Erreur lors de la lecture du Parquet: {str(e)}")