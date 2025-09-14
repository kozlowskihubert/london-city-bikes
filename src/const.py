import torch

# DATASET CONSTANTS
SEQ_LENGTH = 7 * 48  # 7 days * 48 half-hour intervals per day
PRED_LENGTH = 48  # Predict 48 steps (24 hours) ahead
REDUCE_DATASET_RATIO = 1 #0.02

INTERVAL_MINUTES = 30
MIN_YEAR = 2016
MAX_YEAR = 2023

DATSET_COLUMNS = ["Starts", 
                  "Ends", 
                  "DayOfWeek", 
                  "HourOfDay", 
                  "Month", 
                  "DayOfYear", 
                  "IsPeak",
                  "temperature_2m (°C)", 
                  "precipitation (mm)", 
                  "rain (mm)", 
                  "wind_speed_10m (km/h)", 
                  "Holiday"]

# MODEL CONSTANTS
TRAIN_RATIO = 0.8  # 80% training, 20% validation
BATCH_SIZE = 128 * 2

INPUT_SIZE = 12
FUTURE_FEATURE_SIZE = 10
HIDDEN_SIZE = 128
OUTPUT_SIZE = 2 * PRED_LENGTH
NUM_LAYERS = 1
NUM_EPOCHS = 5
USE_CLUSTER_EMBEDDING = True

# TRAIN CONSTANTS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROPOUT = 0
LEARNING_RATE = 0.0001



# # Hyperparameters
# FUTURE_FEATURE_SIZE = 10
# HIDDEN_SIZE = 128
# OUTPUT_SIZE = 2 * PRED_LENGTH
# NUM_LAYERS = 1
# NUM_EPOCHS = 5
# DROPOUT = 0
# LEARNING_RATE = 0.0001
# USE_CLUSTER_EMBEDDING = True