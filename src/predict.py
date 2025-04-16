import numpy as np
import torch
from src.scaler import ScalerManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_LENGTH = 48

def predict(model, dataloader, extract_metadata=True, use_cluster_embedding=True):
    model.eval()
    
    all_preds = []
    all_true = []
    metadata = {'DayOfWeek': [], 'Holiday': [], 'IsPeak': []} if extract_metadata else None
    scalers = ScalerManager().get_all_scalers()

    with torch.no_grad():
        for batch in dataloader:
            if use_cluster_embedding:
                batch_X, batch_future, batch_y, cluster_id = batch
                cluster_id = cluster_id.to(DEVICE)
            else:
                batch_X, batch_future, batch_y = batch
                cluster_id = None

            batch_X = batch_X.to(DEVICE)
            batch_future = batch_future.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            predictions = model(batch_X, batch_future, cluster_id).cpu().numpy()
            predictions = predictions.reshape(-1, PRED_LENGTH, 2)

            if extract_metadata:
                for i in range(batch_future.shape[0]):
                    for t in range(batch_future.shape[1]):
                        metadata['DayOfWeek'].append(int(round(batch_future[i, t, 0].cpu().numpy() * 6 + 1)))
                        metadata['Holiday'].append(int(batch_future[i, t, 8].cpu().numpy()))
                        metadata['IsPeak'].append(int(batch_future[i, t, 4].cpu().numpy()))

            # Inverse transform predictions
            original_scale_predictions = [
                inverse_transform_predictions(predictions[i], scalers[cluster_id[i].item()])
                for i in range(predictions.shape[0])
            ]
            original_scale_true = [
                inverse_transform_predictions(batch_y[i].cpu().numpy(), scalers[cluster_id[i].item()])
                for i in range(batch_y.shape[0])
            ]

            all_preds.append(np.array(original_scale_predictions))
            all_true.append(np.array(original_scale_true))
    
    all_true = np.vstack(all_true)
    all_preds = np.vstack(all_preds)

    if extract_metadata:
        for key in metadata:
            metadata[key] = np.array(metadata[key])

    return all_preds, all_true, metadata


def inverse_transform_predictions(predictions, scaler):
    """
    Inverse transform predictions using the scaler for the given cluster.
    predictions: numpy array of shape (num_samples, 2) containing predictions for Starts and Ends.
    cluster: the cluster ID to retrieve the corresponding scaler.
    """
    # Reshape predictions to match the scaler's expected input shape
    predictions = predictions.reshape(-1, 2)  # Ensure predictions are in shape (num_samples, 2)
    
    # Create a dummy array for the other features (excluding Starts and Ends)
    num_samples = predictions.shape[0]
    num_features = scaler.n_features_in_  # Total number of features used during fitting
    dummy_features = np.zeros((num_samples, num_features - 2))  # Exclude Starts and Ends
    
    combined = np.hstack([predictions, dummy_features])
    
    original_scale_predictions = scaler.inverse_transform(combined)
    
    # Extract only the Starts and Ends columns
    original_scale_predictions = original_scale_predictions[:, :2]
    
    return original_scale_predictions