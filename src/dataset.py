import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

SEQ_LENGTH = 7 * 48  # 7 days * 48 half-hour intervals per day
PRED_LENGTH = 48  # Predict 48 steps (24 hours) ahead
TRAIN_RATIO = 0.8  # 80% training, 20% validation
BATCH_SIZE = 128

AGGREGATED_RIDES_DATA_PATH = "./processed/final_aggregated_rides_example.csv"
WEATHER_DATA_PATH = "./data/open-meteo-51.49N0.16W23m.csv"
HOLIDAYS_DATA_PATH = "./data/bank_holidays.csv"

# Custom Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, future_features, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.future_features = torch.tensor(future_features, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.future_features[idx], self.y[idx]


def load_dataset():
    df = pl.read_csv(AGGREGATED_RIDES_DATA_PATH)
    df_weather_data = pl.read_csv(WEATHER_DATA_PATH)
    df_bank_holidays = pl.read_csv(HOLIDAYS_DATA_PATH)

    df = df.with_columns(
        pl.col("Interval").str.to_datetime().alias("Interval")
    )
    df = df.sort(["Cluster", "Interval"])

    df = df.with_columns(
        pl.col("Interval").dt.weekday().alias("DayOfWeek")
    )
    df = df.with_columns(
        pl.col("Interval").dt.hour().alias("HourOfDay"),
        pl.col("Interval").dt.month().alias("Month"),
        pl.col("Interval").dt.ordinal_day().alias("DayOfYear")
    )
    df = df.with_columns(
        ((((pl.col("HourOfDay") >= 7) & (pl.col("HourOfDay") < 10) | ((pl.col("HourOfDay") >= 17) & (pl.col("HourOfDay") < 19))) & ((pl.col("DayOfWeek") >= 1) & (pl.col("DayOfWeek") <= 5))) |
        (((pl.col("HourOfDay") >= 13) & (pl.col("HourOfDay") <= 19)) & ((pl.col("DayOfWeek") >= 6) & (pl.col("DayOfWeek") <= 7)))
    ).cast(pl.Int8).alias("IsPeak"))

    df_weather_data = df_weather_data.with_columns(
        pl.col("time").str.to_datetime().alias("Interval")
    )

    df_weather_data = df_weather_data.drop(["time"])

    df_half_hour = df_weather_data.with_columns(
        (pl.col("Interval") + pl.duration(minutes=30)).alias("Interval")
    )

    df_weather_data = (
        pl.concat([df_weather_data, df_half_hour])
        .sort("Interval")
    )
    df_weather_data = df_weather_data.filter(
        (pl.col("Interval").dt.year() >= 2018) & (pl.col("Interval").dt.year() <= 2019)
    )

    df = df.join(df_weather_data, on="Interval")

    df_bank_holidays = df_bank_holidays.with_columns(
        pl.col("Date").str.to_date().alias("Date")
    )
    bank_holiday_series = df_bank_holidays["Date"]

    df = df.with_columns(
        (df["Interval"].dt.date().is_in(bank_holiday_series)).cast(pl.Int8).alias("Holiday")
    )

    return df


def create_sequences_with_future_features(data, seq_length, pred_length):
    """
    Create sequences for multi-step forecasting with separate historical and future auxiliary features.
    
    Parameters:
    data: numpy array of shape (num_samples, num_features).
    seq_length: length of the historical input sequence.
    pred_length: number of steps to predict ahead.
    
    Returns:
    X_hist: Historical data including all features [num_samples, seq_length, num_features]
    X_aux: Future auxiliary features (excluding target features) [num_samples, pred_length, aux_features]
    y: Target values [num_samples, pred_length, 2]
    """
    xs, future_features, ys = [], [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        # Historical sequence (all features)
        x = data[i:(i + seq_length)]
        # Future auxiliary features (excluding target variables)
        future_feature = data[(i + seq_length):(i + seq_length + pred_length), 2:]  # All features except Starts and Ends
        # Target values (only Starts and Ends)
        y = data[(i + seq_length):(i + seq_length + pred_length), :2]
        xs.append(x)
        future_features.append(future_feature)
        ys.append(y)
    return np.array(xs), np.array(future_features), np.array(ys)


def train_val_split(data, train_ratio=0.8):
    """
    Split data into training and validation sets.
    data: numpy array of shape (num_samples, num_features).
    train_ratio: fraction of data to use for training.
    """
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:]
    return train_data, val_data


def get_train_val_split_dataset_for_clusters(df):
    clusters_train_data = {}
    clusters_val_data = {}
    scalers = {}

    for cluster in df["Cluster"].unique():
        # Select relevant columns: Starts, Ends, DayOfWeek, HourOfDay, Month, DayOfYear, IsPeak
        cluster_data = df.filter(pl.col("Cluster") == cluster).select(
            ["Starts", "Ends", "DayOfWeek", "HourOfDay", "Month", "DayOfYear", "IsPeak", "temperature_2m (°C)",	
            "precipitation (mm)", "rain (mm)", "wind_speed_10m (km/h)", "Holiday"]
        ).to_numpy()
        
        # Normalize the data
        scaler = MinMaxScaler()
        cluster_data_normalized = scaler.fit_transform(cluster_data)
        scalers[cluster] = scaler  # Save the scaler for this cluster
        
        # Split into training and validation sets
        train_data, val_data = train_val_split(cluster_data_normalized, TRAIN_RATIO)
        
        # Create sequences for training and validation
        X_train, future_features_train, y_train = create_sequences_with_future_features(train_data, SEQ_LENGTH, PRED_LENGTH)
        X_val, future_features_val, y_val = create_sequences_with_future_features(val_data, SEQ_LENGTH, PRED_LENGTH)
        
        clusters_train_data[cluster] = (X_train, future_features_train, y_train)
        clusters_val_data[cluster] = (X_val, future_features_val, y_val)
    
    return clusters_train_data, clusters_val_data, scalers


def get_dataloaders():
    df = load_dataset()
    clusters_train_data, clusters_val_data, scalers = get_train_val_split_dataset_for_clusters(df)

    train_dataloaders = {}
    val_dataloaders = {}

    for cluster, (X_train, future_train, y_train) in clusters_train_data.items():
        train_dataset = TimeSeriesDataset(X_train, future_train, y_train)
        train_dataloaders[cluster] = DataLoader(train_dataset, batch_size=32, shuffle=True)

        val_X, val_future, val_y = clusters_val_data[cluster]

        val_dataset = TimeSeriesDataset(val_X, val_future, val_y)
        val_dataloaders[cluster] = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_dataloaders, val_dataloaders, scalers
