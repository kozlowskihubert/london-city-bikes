import numpy as np
import polars as pl

SEQ_LENGTH = 7 * 48  # 7 days * 48 half-hour intervals per day
PRED_LENGTH = 48  # Predict 48 steps (24 hours) ahead
TRAIN_RATIO = 0.8  # 80% training, 20% validation
BATCH_SIZE = 128

AGGREGATED_RIDES_DATA_PATH = "./processed/final_aggregated_rides.csv"
#AGGREGATED_RIDES_DATA_PATH = "./processed/final_aggregated_rides_example.csv"
WEATHER_DATA_PATH = "./data/open-meteo-51.49N0.16W23m.csv"
HOLIDAYS_DATA_PATH = "./data/bank_holidays.csv"


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
        (pl.col("Interval").dt.year() >= 2016) & (pl.col("Interval").dt.year() <= 2023)
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
