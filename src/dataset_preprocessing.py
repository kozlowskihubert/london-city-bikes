"""
Bike hire data processing module for cleaning, standardizing, and aggregating ride data.
"""

import json
import os
from datetime import timedelta
from typing import Dict, List, Set, Tuple, Optional, Any

import polars as pl

from src.paths import (
    NAME_MAPPING_PATH,
    STATION_LOCATIONS_PATH,
    WEATHER_DATA_PATH,
    HOLIDAYS_DATA_PATH,
    PROCESSED_DATASET_DIR,
    AGGREGATED_RIDES_DATA_PATH
)

from src.const import (
    INTERVAL_MINUTES,
    MIN_YEAR,
    MAX_YEAR,
    WEEKDAY_MORNING_RUSH,
    WEEKDAY_EVENING_RUSH,
    WEEKEND_RUSH
)

from src.mappings import (
    COLUMN_MAPPING
)

COLUMNS_TO_DROP = [
    'Bike Id', 'Bike model', 'Bike number', 'Rental Id',
    'Number', 'Total duration', 'Total duration (ms)',
    'Start station number', 'End station number',
]

REQUIRED_COLUMNS = ["End station", "Start station", "End Date", "Start Date"]

FINAL_COLUMNS = [
    "Start Date", "End Date", "Start station", "End station",
    "Start cluster", "End cluster", "Duration_Seconds"
]

def load_station_mappings() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load station name mappings and cluster assignments."""
    with open(NAME_MAPPING_PATH, "r") as f:
        name_mapping = json.load(f)
    
    station_locations_df = pl.read_csv(STATION_LOCATIONS_PATH)
    cluster_mapping = dict(zip(
        station_locations_df["canonical_name"].to_list(),
        station_locations_df["cluster_label"].to_list()
    ))
    
    return name_mapping, cluster_mapping


def standardize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename columns and drop unnecessary ones."""
    for old_name, new_name in COLUMN_MAPPING.items():
        if old_name in df.columns:
            df = df.rename({old_name: new_name})
    
    existing_columns_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
    return df.drop(existing_columns_to_drop)


def clean_null_data(df: pl.DataFrame) -> Tuple[pl.DataFrame, int]:
    """Remove rows with null values in critical columns."""
    rows_before = df.height
    df_clean = df.drop_nulls(subset=REQUIRED_COLUMNS)
    rows_dropped = rows_before - df_clean.height
    return df_clean, rows_dropped


def map_station_names(df: pl.DataFrame, name_mapping: Dict[str, str]) -> pl.DataFrame:
    """Apply station name mapping to standardize station names."""
    start_stations = [name_mapping.get(station, station) 
                     for station in df["Start station"].to_list()]
    end_stations = [name_mapping.get(station, station) 
                   for station in df["End station"].to_list()]
    
    return df.with_columns([
        pl.Series("Start station", start_stations),
        pl.Series("End station", end_stations)
    ])


def assign_clusters(df: pl.DataFrame, cluster_mapping: Dict[str, str]) -> pl.DataFrame:
    """Assign cluster labels to start and end stations."""
    start_clusters = [cluster_mapping.get(station) 
                     for station in df["Start station"].to_list()]
    end_clusters = [cluster_mapping.get(station) 
                   for station in df["End station"].to_list()]
    
    return df.with_columns([
        pl.Series("Start cluster", start_clusters),
        pl.Series("End cluster", end_clusters)
    ])


def parse_dates(df: pl.DataFrame) -> pl.DataFrame:
    """Parse date columns with multiple format support."""
    def parse_date_column(col_name: str):
        return (
            pl.col(col_name)
            .str.strptime(pl.Datetime, "%d/%m/%Y %H:%M", strict=False)
            .fill_null(pl.col(col_name).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M", strict=False))
        )
    
    return df.with_columns([
        parse_date_column("Start Date").alias("Start Date"),
        parse_date_column("End Date").alias("End Date")
    ])


def calculate_duration(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate ride duration in seconds."""
    return df.with_columns([
        (pl.col("End Date").dt.timestamp() - pl.col("Start Date").dt.timestamp())
        .alias("Duration_Seconds")
    ])


def filter_invalid_rides(df: pl.DataFrame) -> Tuple[pl.DataFrame, int]:
    """Remove rides with duration <= 60 seconds and same start/end station."""
    rows_before = df.height
    df_filtered = df.filter(
        ~((pl.col("Duration_Seconds") <= 60) & (pl.col("End station") == pl.col("Start station")))
    )
    rows_dropped = rows_before - df_filtered.height
    return df_filtered, rows_dropped


def log_processing_stats(filename: str, initial_rows: int, null_drops: int, 
                        filter_drops: int, final_rows: int) -> None:
    """Log processing statistics for a file."""
    total_dropped = initial_rows - final_rows
    drop_percentage = total_dropped / initial_rows * 100 if initial_rows > 0 else 0
    
    print(f"Processing {filename}...")
    print(f"  Rows processed: {initial_rows}")
    print(f"  Rows dropped (nulls): {null_drops}")
    print(f"  Rows dropped (short trips): {filter_drops}")
    print(f"  Total rows dropped: {total_dropped} ({drop_percentage:.2f}%)")
    print(f"  Rows retained: {final_rows}")


def process_file(file_path: str, name_mapping: Dict[str, str], 
                cluster_mapping: Dict[str, str]) -> pl.DataFrame:
    """Process a single CSV file through the complete cleaning pipeline."""
    df = pl.read_csv(file_path, ignore_errors=True)
    initial_rows = df.height
    
    if initial_rows == 0:
        return df
    
    df = standardize_columns(df)
    df, null_drops = clean_null_data(df)
    df = map_station_names(df, name_mapping)
    df = assign_clusters(df, cluster_mapping)
    df = parse_dates(df)
    df = calculate_duration(df)
    df, filter_drops = filter_invalid_rides(df)
    
    log_processing_stats(
        os.path.basename(file_path), initial_rows, null_drops, filter_drops, df.height
    )
    
    return df.select(FINAL_COLUMNS)


def collect_unique_clusters(df: pl.DataFrame) -> Set[str]:
    """Extract unique cluster labels from dataframe."""
    clusters = set()
    if "Start cluster" in df.columns and "End cluster" in df.columns:
        start_clusters = [c for c in df["Start cluster"].unique().to_list() if c is not None]
        end_clusters = [c for c in df["End cluster"].unique().to_list() if c is not None]
        clusters.update(start_clusters)
        clusters.update(end_clusters)
    return clusters


def generate_time_intervals(start_date, end_date, interval_minutes: int = INTERVAL_MINUTES) -> List:
    """Generate all possible time intervals within the date range."""
    intervals = []
    current_date = start_date
    while current_date <= end_date:
        intervals.append(current_date)
        current_date += timedelta(minutes=interval_minutes)
    return intervals


def create_ride_aggregations(df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Create separate aggregations for ride starts and ends."""
    df = df.with_columns([
        pl.col("Start Date").dt.truncate("30m").alias("Start Interval"),
        pl.col("End Date").dt.truncate("30m").alias("End Interval")
    ])
    
    start_df = (df.select(["Start cluster", "Start Interval"])
                 .rename({"Start cluster": "Cluster", "Start Interval": "Interval"}))
    
    end_df = (df.select(["End cluster", "End Interval"])
               .rename({"End cluster": "Cluster", "End Interval": "Interval"}))
    
    start_counts = start_df.group_by(["Cluster", "Interval"]).agg(pl.len().alias("Starts"))
    end_counts = end_df.group_by(["Cluster", "Interval"]).agg(pl.len().alias("Ends"))
    
    return start_counts, end_counts


def create_complete_time_series(clusters: List[str], intervals: List) -> pl.DataFrame:
    """Create a complete time series with all cluster-interval combinations."""
    combinations = [
        {"Cluster": cluster, "Interval": interval}
        for cluster in clusters
        for interval in intervals
    ]
    return pl.DataFrame(combinations)


def aggregate_rides_by_cluster(df: pl.DataFrame, all_clusters: List[str], 
                              all_intervals: List) -> pl.DataFrame:
    """Aggregate ride data into 30-minute intervals by cluster."""
    start_counts, end_counts = create_ride_aggregations(df)
    base_df = create_complete_time_series(all_clusters, all_intervals)
    
    result_df = (base_df
                 .join(start_counts, on=["Cluster", "Interval"], how="left")
                 .join(end_counts, on=["Cluster", "Interval"], how="left")
                 .fill_null(0))
    
    return result_df


def find_date_range(dataframes: List[pl.DataFrame]) -> Tuple[Optional[Any], Optional[Any]]:
    """Find the overall date range across all dataframes."""
    min_date = max_date = None
    
    for df in dataframes:
        if df.height == 0 or "Start Date" not in df.columns:
            continue
            
        current_min = df["Start Date"].min()
        current_max = df["End Date"].max()
        
        if min_date is None or current_min < min_date:
            min_date = current_min
        if max_date is None or current_max > max_date:
            max_date = current_max
    
    return min_date, max_date


def save_results(df: pl.DataFrame) -> None:
    """Save the final aggregated results to CSV."""
    os.makedirs(PROCESSED_DATASET_DIR, exist_ok=True)
    df.write_csv(AGGREGATED_RIDES_DATA_PATH)
    print("✅ Pipeline completed!")


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add time-based features to the dataframe."""
    return df.with_columns([
        pl.col("Interval").dt.weekday().alias("DayOfWeek"),
        pl.col("Interval").dt.hour().alias("HourOfDay"),
        pl.col("Interval").dt.month().alias("Month"),
        pl.col("Interval").dt.ordinal_day().alias("DayOfYear")
    ])


def add_peak_hour_indicator(df: pl.DataFrame) -> pl.DataFrame:
    """Add peak hour indicator based on time and day of week."""
    weekday_condition = (
        (((pl.col("HourOfDay") >= WEEKDAY_MORNING_RUSH[0]) & 
          (pl.col("HourOfDay") < WEEKDAY_MORNING_RUSH[1])) |
         ((pl.col("HourOfDay") >= WEEKDAY_EVENING_RUSH[0]) & 
          (pl.col("HourOfDay") < WEEKDAY_EVENING_RUSH[1]))) &
        ((pl.col("DayOfWeek") >= 1) & (pl.col("DayOfWeek") <= 5))
    )
    
    weekend_condition = (
        ((pl.col("HourOfDay") >= WEEKEND_RUSH[0]) & 
         (pl.col("HourOfDay") <= WEEKEND_RUSH[1])) &
        ((pl.col("DayOfWeek") >= 6) & (pl.col("DayOfWeek") <= 7))
    )
    
    return df.with_columns(
        (weekday_condition | weekend_condition).cast(pl.Int8).alias("IsPeak")
    )


def load_and_prepare_weather_data() -> pl.DataFrame:
    """Load and prepare weather data with proper time intervals."""
    df_weather = pl.read_csv(WEATHER_DATA_PATH)
    
    df_weather = df_weather.with_columns(
        pl.col("time").str.to_datetime().alias("Interval")
    ).drop(["time"])
    
    # Create 30-minute offset data
    df_half_hour = df_weather.with_columns(
        (pl.col("Interval") + pl.duration(minutes=30)).alias("Interval")
    )
    
    # Combine and filter by year range
    df_weather_combined = (
        pl.concat([df_weather, df_half_hour])
        .sort("Interval")
        .filter(
            (pl.col("Interval").dt.year() >= MIN_YEAR) & 
            (pl.col("Interval").dt.year() <= MAX_YEAR)
        )
    )
    
    return df_weather_combined


def load_holiday_data() -> pl.Series:
    """Load and prepare holiday data."""
    df_holidays = pl.read_csv(HOLIDAYS_DATA_PATH)
    return df_holidays.with_columns(
        pl.col("Date").str.to_date().alias("Date")
    )["Date"]


def add_holiday_indicator(df: pl.DataFrame, holiday_dates: pl.Series) -> pl.DataFrame:
    """Add holiday indicator to the dataframe."""
    return df.with_columns(
        (df["Interval"].dt.date().is_in(holiday_dates))
        .cast(pl.Int8)
        .alias("Holiday")
    )


def add_features(aggregated_df: pl.DataFrame) -> pl.DataFrame:
    """
    Merge aggregated ride data with weather and holiday information.
    
    Args:
        aggregated_df: Aggregated ride data DataFrame
        
    Returns:
        Enhanced DataFrame with weather, holiday, and time features
    """
    df = aggregated_df.sort(["Cluster", "Interval"])
    
    df = add_time_features(df)
    df = add_peak_hour_indicator(df)
    
    weather_data = load_and_prepare_weather_data()
    df = df.join(weather_data, on="Interval")
    
    holiday_dates = load_holiday_data()
    df = add_holiday_indicator(df, holiday_dates)
    
    return df


def process_pipeline(csv_files: List[str], return_combined: bool = False) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    """
    Main pipeline function to process multiple CSV files and aggregate bike hire data.
    
    Args:
        csv_files: List of CSV file paths to process
        return_combined: If True, also return the combined raw data before aggregation
        
    Returns:
        Tuple of (aggregated_df, combined_df if requested else None)
    """
    name_mapping, cluster_mapping = load_station_mappings()
    
    processed_dfs = []
    all_clusters = set()
    
    for file_path in csv_files:
        df = process_file(file_path, name_mapping, cluster_mapping)
        
        if df.height == 0:
            print(f"Skipping empty dataframe from {file_path}")
            continue
        
        processed_dfs.append(df)
        all_clusters.update(collect_unique_clusters(df))
    
    if not processed_dfs or not all_clusters:
        print("No valid data processed.")
        return pl.DataFrame(), None
    
    combined_df = pl.concat(processed_dfs)
    min_date, max_date = find_date_range(processed_dfs)
    
    if min_date is None or max_date is None:
        print("Could not determine date range.")
        return pl.DataFrame(), None
    
    all_intervals = generate_time_intervals(min_date, max_date)
    aggregated_df = aggregate_rides_by_cluster(combined_df, list(all_clusters), all_intervals)
    
    save_results(aggregated_df)
    
    return aggregated_df, combined_df if return_combined else None
