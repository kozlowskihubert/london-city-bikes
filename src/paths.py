from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

TF_CYCLING_DATA_DIR = DATA_DIR / "tfl_cycling_data"
NAME_MAPPING_DIR = DATA_DIR / "name_mapping"
STATION_LOCATIONS_DIR = DATA_DIR / "station_locations"
MODELS_DIR = DATA_DIR / "models"
PROCESSED_DATASET_DIR = DATA_DIR / "processed_dataset"

WEATHER_DATA_PATH = DATA_DIR / "open-meteo-51.49N0.16W23m.csv"
HOLIDAYS_DATA_PATH = DATA_DIR / "bank_holidays.csv"
NAME_MAPPING_PATH = NAME_MAPPING_DIR / "name_mapping_2.json"
STATION_LOCATIONS_PATH = STATION_LOCATIONS_DIR / "stations_locations.csv"
#AGGREGATED_RIDES_DATA_PATH = PROCESSED_DATASET_DIR / "final_aggregated_rides_example_merged.csv"
AGGREGATED_RIDES_DATA_PATH = PROCESSED_DATASET_DIR / "final_aggregated_rides_merged.csv"

