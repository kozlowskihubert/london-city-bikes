import os
from typing import Dict
import pickle
from sklearn.preprocessing import MinMaxScaler
import polars as pl

from src.paths import SCALER_DIR, AGGREGATED_RIDES_DATA_PATH
from src.const import DATSET_COLUMNS

class ScalerManager:
    """Handles creation and caching of MinMaxScalers for each cluster"""

    def __init__(self, scaler_dir: str = SCALER_DIR):
        self.scaler_dir = scaler_dir
        os.makedirs(scaler_dir, exist_ok=True)
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.scaler_filename_suffix = "_cluster_scaler.pkl"
    
    def get_scaler(self, cluster: str) -> MinMaxScaler:
        """Retrieves a cached scaler or loads from disk"""
        if cluster in self.scalers:
            return self.scalers[cluster]

        scaler_path = os.path.join(self.scaler_dir, f"{cluster}{self.scaler_filename_suffix}")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scalers[cluster] = pickle.load(f)
            return self.scalers[cluster]
        
        raise ValueError(f"Scaler for cluster {cluster} not found. Call create_scaler() first.")

    def create_scaler(self, df: pl.DataFrame, cluster: str) -> None:
        """Creates and saves a scaler for a specific cluster"""
        cluster_data = df.filter(pl.col("Cluster") == cluster).select(DATSET_COLUMNS).to_numpy()
        
        scaler = MinMaxScaler()
        scaler.fit(cluster_data)

        # Save to cache and disk
        self.scalers[cluster] = scaler
        scaler_path = os.path.join(self.scaler_dir, f"{cluster}{self.scaler_filename_suffix}")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    def create_all_scalers(self, df: pl.DataFrame) -> None:
        """Creates scalers for all clusters in the DataFrame"""
        clusters = df["Cluster"].unique().to_list()
        for cluster in clusters:
            self.create_scaler(df, cluster)

    def get_all_scalers(self) -> Dict[int, MinMaxScaler]:
        """
        Returns a dictionary of all scalers saved in SCALER_DIR.
        Keys are cluster names, values are MinMaxScaler objects.
        """
        scalers = {}
        for filename in os.listdir(self.scaler_dir):
            if filename.endswith(self.scaler_filename_suffix):
                # Extract cluster name by removing the suffix
                cluster = int(filename[:-len(self.scaler_filename_suffix)])
                try:
                    scalers[cluster] = self.get_scaler(cluster)
                except ValueError:
                    continue
        return scalers

if __name__ == "__main__":
    df = pl.read_csv(AGGREGATED_RIDES_DATA_PATH)
    
    scaler_manager = ScalerManager()
    scaler_manager.create_all_scalers(df)
    
    cluster = 0 
    scaler = scaler_manager.get_scaler(cluster)
