from typing import Optional, Iterable
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from src.scaler import ScalerManager
from src.const import (
    SEQ_LENGTH,
    PRED_LENGTH,
    TRAIN_RATIO,
    REDUCE_DATASET_RATIO,
    BATCH_SIZE,
    DATSET_COLUMNS,
)
from src.paths import AGGREGATED_RIDES_DATA_PATH

class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 data, 
                 scalers, 
                 reduce_dataset_ratio=REDUCE_DATASET_RATIO):
        self.data = data
        self.scalers = scalers
        self.seq_length = SEQ_LENGTH
        self.pred_length = PRED_LENGTH
        self.reduce_dataset_ratio = reduce_dataset_ratio 
        self.samples = self._prepare_samples()
    def _prepare_samples(self):
        all_samples = []
        for cluster, df in self.data.items():
            #TODO Try to filter out these columns earlier
            cluster_data = df.select(
                DATSET_COLUMNS
            ).to_numpy()

            scaler = self.scalers[cluster]
            cluster_data = scaler.transform(cluster_data)
            total_sequences = len(cluster_data) - self.seq_length - self.pred_length + 1
            dataset_size = int(total_sequences)

            start_idx = 0
            end_idx = int(dataset_size * self.reduce_dataset_ratio)
            
            for idx in range(start_idx, end_idx):
                all_samples.append((cluster, cluster_data, idx))

        return all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cluster_id, cluster_data, i = self.samples[idx]

        x = cluster_data[i:i + self.seq_length]
        future_feature = cluster_data[i + self.seq_length:i + self.seq_length + self.pred_length, 2:]
        y = cluster_data[i + self.seq_length:i + self.seq_length + self.pred_length, :2]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(future_feature, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(cluster_id, dtype=torch.long)
        )
    
def filter_years(df_all: pl.DataFrame, 
                 years: list[int]
    ) -> pl.DataFrame:
    df_all = df_all.with_columns(
        pl.col("Interval").str.to_datetime().alias("Interval")
    )
    return df_all.filter(
        (pl.col("Interval").dt.year() >= min(years)) & 
        (pl.col("Interval").dt.year() <= max(years))
    )

def exclude_clusters(df_all: pl.DataFrame,
                     exclude_clusters: Iterable[int]
    ) -> pl.DataFrame:
    exclude_set = set(exclude_clusters)
    return df_all.filter(
        ~pl.col("Cluster").is_in(exclude_set)
    )

def split_train_val(data_by_cluster: dict, 
                    train_ratio: float = 0.8):
    train_data = {}
    val_data = {}
    
    for cluster, df in data_by_cluster.items():
        n_rows = df.height
        train_end = int(n_rows * train_ratio)
        
        train_data[cluster] = df[:train_end]
        val_data[cluster] = df[train_end:]
        
    return train_data, val_data

def get_dataloaders(exclude_clusters: Optional[Iterable[int]]=None,
                    years: Optional[Iterable[int]]=None,
                    batch_size=BATCH_SIZE
    ):
    df_all = pl.scan_csv(AGGREGATED_RIDES_DATA_PATH).collect()

    if years is not None:
        df_all = filter_years(df_all, years)

    if exclude_clusters is not None:
        df_all = exclude_clusters(df_all, exclude_clusters)
    
    scalers = ScalerManager().get_all_scalers()
    clusters = set(sorted(df_all.select("Cluster").unique()["Cluster"].to_list()))
    
    data_by_cluster = {cluster: df_all.filter(pl.col("Cluster") == cluster) for cluster in clusters}
    train_data, val_data = split_train_val(data_by_cluster, TRAIN_RATIO)

    train_dataset = TimeSeriesDataset(train_data, scalers)
    val_dataset = TimeSeriesDataset(val_data, scalers)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader, clusters
