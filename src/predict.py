import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_LENGTH = 48

def predict(model, dataloader, scaler, extract_metadata=True):
    model.eval()
    
    all_preds = []
    all_true = []
    metadata = {'DayOfWeek': [], 'Holiday': [], 'IsPeak': []} if extract_metadata else None

    with torch.no_grad():
        for batch_X, batch_future, batch_y in dataloader:
            batch_X = batch_X.to(DEVICE)
            batch_future = batch_future.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            predictions = model(batch_X, batch_future).cpu().numpy()
            
            predictions = predictions.reshape(-1, PRED_LENGTH, 2)

            if extract_metadata:
                # Extract DayOfWeek, Holiday, IsPeak from future features
                for i in range(batch_future.shape[0]):
                    for t in range(batch_future.shape[1]):
                        # Day of week is at index 0 in future features (after targets)
                        metadata['DayOfWeek'].append(int(round(batch_future[i, t, 0].cpu().numpy() * 6 + 1)))
                        # Holiday is at index 8 in future features 
                        metadata['Holiday'].append(int(batch_future[i, t, 8].cpu().numpy()))
                        # IsPeak is at index 4 in future features
                        metadata['IsPeak'].append(int(batch_future[i, t, 4].cpu().numpy()))
            
            # Inverse transform predictions
            original_scale_predictions = []
            for i in range(predictions.shape[0]):
                pred = inverse_transform_predictions(predictions[i], scaler)
                original_scale_predictions.append(pred)
            original_scale_predictions = np.array(original_scale_predictions)
            
            # Inverse transform the true values (batch_y)
            original_scale_true = []
            for i in range(batch_y.shape[0]):
                true = inverse_transform_predictions(batch_y[i].cpu().numpy(), scaler)
                original_scale_true.append(true)
            original_scale_true = np.array(original_scale_true)

            all_preds.append(original_scale_predictions)
            all_true.append(original_scale_true)
    
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