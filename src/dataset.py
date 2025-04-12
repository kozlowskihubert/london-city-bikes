import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from src.scaler import ScalerManager

SEQ_LENGTH = 7 * 48  # 7 days * 48 half-hour intervals per day
PRED_LENGTH = 48  # Predict 48 steps (24 hours) ahead
TRAIN_RATIO = 0.8  # 80% training, 20% validation
BATCH_SIZE = 128

#AGGREGATED_RIDES_DATA_PATH = "./processed/final_aggregated_rides_example_merged.csv"
AGGREGATED_RIDES_DATA_PATH = "./processed/final_aggregated_rides_merged.csv"

class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, cluster, seq_length, pred_length, scaler=None, is_train=True, train_ratio=0.8):
        self.data_path = data_path
        self.cluster = cluster
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.scaler = scaler
        self.is_train = is_train
        self.train_ratio = train_ratio
        
        self._load_and_preprocess_data()
        
    def _load_and_preprocess_data(self):
        """Load and preprocess data for this cluster"""
        df = pl.scan_csv(self.data_path).filter(pl.col("Cluster") == self.cluster).collect()
        
        self.data = df.select(
            ["Starts", "Ends", "DayOfWeek", "HourOfDay", "Month", "DayOfYear", "IsPeak", 
             "temperature_2m (°C)", "precipitation (mm)", "rain (mm)", "wind_speed_10m (km/h)", "Holiday"]
        ).to_numpy()
        
        if self.scaler:
            self.data = self.scaler.transform(self.data)
        else:
            self.scaler = MinMaxScaler()
            self.data = self.scaler.fit_transform(self.data)

        total_sequences = len(self.data) - self.seq_length - self.pred_length + 1
        train_size = int(total_sequences * self.train_ratio)
        
        if self.is_train:
            self.start_idx = 0
            self.end_idx = train_size
        else:
            self.start_idx = train_size
            self.end_idx = total_sequences
            
    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        # Adjust for train/val split
        actual_idx = self.start_idx + idx
        
        x = self.data[actual_idx:actual_idx + self.seq_length]
        future_feature = self.data[actual_idx + self.seq_length:actual_idx + self.seq_length + self.pred_length, 2:]
        y = self.data[actual_idx + self.seq_length:actual_idx + self.seq_length + self.pred_length, :2]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(future_feature, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

def get_dataloaders():
    """Get memory-efficient dataloaders for each cluster"""
    df_clusters = pl.scan_csv(AGGREGATED_RIDES_DATA_PATH).select("Cluster").unique().collect()
    clusters = df_clusters["Cluster"].to_list()
    
    scalers = ScalerManager().get_all_scalers()
    
    train_dataloaders = {}
    val_dataloaders = {}
    
    for cluster in clusters:
        scaler = scalers[cluster]
        
        train_dataset = TimeSeriesDataset(
            data_path=AGGREGATED_RIDES_DATA_PATH,
            cluster=cluster,
            seq_length=SEQ_LENGTH,
            pred_length=PRED_LENGTH,
            scaler=scaler,
            is_train=True,
            train_ratio=TRAIN_RATIO
        )
        
        val_dataset = TimeSeriesDataset(
            data_path=AGGREGATED_RIDES_DATA_PATH,
            cluster=cluster,
            seq_length=SEQ_LENGTH,
            pred_length=PRED_LENGTH,
            scaler=scaler,
            is_train=False,
            train_ratio=TRAIN_RATIO
        )
        
        train_dataloaders[cluster] = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,  # Only shuffle training data
            num_workers=1,
            pin_memory=True
        )
        
        val_dataloaders[cluster] = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,  # Don't shuffle validation data
            num_workers=1,
            pin_memory=True
        )
    
    return train_dataloaders, val_dataloaders, scalers
