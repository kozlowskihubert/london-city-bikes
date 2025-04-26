from typing import Optional, Iterable
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from src.scaler import ScalerManager

SEQ_LENGTH = 7 * 48  # 7 days * 48 half-hour intervals per day
PRED_LENGTH = 48  # Predict 48 steps (24 hours) ahead
TRAIN_RATIO = 0.8
REDUCE_DATASET_RATIO = 1 #0.02
BATCH_SIZE = 128 * 2


AGGREGATED_RIDES_DATA_PATH = "./processed/final_aggregated_rides_example_merged.csv"
#AGGREGATED_RIDES_DATA_PATH = "./processed/final_aggregated_rides_merged.csv"

class TimeSeriesDataset(Dataset):
    def __init__(self, data, scalers, cluster_ids, use_cluster_embedding=False, is_train=True, train_ratio=TRAIN_RATIO, reduce_dataset_ratio=REDUCE_DATASET_RATIO):
        self.data = data
        self.scalers = scalers
        self.cluster_ids = cluster_ids
        self.use_cluster_embedding = use_cluster_embedding
        self.seq_length = SEQ_LENGTH
        self.pred_length = PRED_LENGTH
        self.is_train = is_train
        self.train_ratio = train_ratio
        self.reduce_dataset_ratio = reduce_dataset_ratio
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        all_samples = []
        for cluster, df in self.data.items():
            cluster_data = df.select(
                ["Starts", "Ends", "DayOfWeek", "HourOfDay", "Month", "DayOfYear", "IsPeak",
                 "temperature_2m (°C)", "precipitation (mm)", "rain (mm)", "wind_speed_10m (km/h)", "Holiday"]
            ).to_numpy()

            scaler = self.scalers[cluster]
            cluster_data = scaler.transform(cluster_data)
            total_sequences = len(cluster_data) - self.seq_length - self.pred_length + 1
            train_size = int(total_sequences * self.train_ratio)

            if self.is_train:
                start_idx = 0
                end_idx = int(train_size * self.reduce_dataset_ratio)
            else:
                start_idx = train_size
                end_idx = int(start_idx + (total_sequences-start_idx) * self.reduce_dataset_ratio)

            for idx in range(start_idx, end_idx):
                all_samples.append((cluster, cluster_data, idx))

        return all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cluster, cluster_data, i = self.samples[idx]
        cluster_id = self.cluster_ids[cluster]

        x = cluster_data[i:i + self.seq_length]
        future_feature = cluster_data[i + self.seq_length:i + self.seq_length + self.pred_length, 2:]
        y = cluster_data[i + self.seq_length:i + self.seq_length + self.pred_length, :2]

        if self.use_cluster_embedding:
            return (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(future_feature, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(cluster_id, dtype=torch.long)
            )
        else:
            return (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(future_feature, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)
            )

def get_dataloaders(use_cluster_embedding_input=True,
                    exclude_clusters: Optional[Iterable[int]]=None,
                    years: Optional[Iterable[int]]=None,
                    batch_size=BATCH_SIZE
    ):
    df_all = pl.scan_csv(AGGREGATED_RIDES_DATA_PATH).collect()

    if years is not None:
        df_all = df_all.with_columns(
            pl.col("Interval").str.to_datetime().alias("Interval")
        )
        df_all = df_all.filter(
            (pl.col("Interval").dt.year() >= min(years)) & (pl.col("Interval").dt.year() <= max(years))
        )

    if exclude_clusters is not None:
        exclude_set = set(exclude_clusters)
        df_all = df_all.filter(
            ~pl.col("Cluster").is_in(exclude_set)
        )

    clusters = sorted(df_all.select("Cluster").unique()["Cluster"].to_list())
    scalers = ScalerManager().get_all_scalers()
    cluster_id_map = {cluster: cluster for cluster in clusters}

    if use_cluster_embedding_input == False:
        train_dataloaders = {}
        val_dataloaders = {}
        for cluster in clusters:
            df_cluster = df_all.filter(pl.col("Cluster") == cluster)
            cluster_data = {cluster: df_cluster}

            train_dataset = TimeSeriesDataset(cluster_data, scalers, cluster_id_map, use_cluster_embedding_input, is_train=True)
            val_dataset = TimeSeriesDataset(cluster_data, scalers, cluster_id_map, use_cluster_embedding_input, is_train=False)

            train_dataloaders[cluster] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
            val_dataloaders[cluster] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        return train_dataloaders, val_dataloaders, scalers, cluster_id_map

    else:
        data_by_cluster = {cluster: df_all.filter(pl.col("Cluster") == cluster) for cluster in clusters}

        train_dataset = TimeSeriesDataset(data_by_cluster, scalers, cluster_id_map, use_cluster_embedding_input, is_train=True)
        val_dataset = TimeSeriesDataset(data_by_cluster, scalers, cluster_id_map, use_cluster_embedding_input, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        return train_loader, val_loader, cluster_id_map
