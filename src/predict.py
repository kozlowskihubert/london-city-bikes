import numpy as np
import torch
from typing import Dict, Optional, Tuple
from src.scaler import ScalerManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_LENGTH = 48

def predict(model, dataloader, extract_metadata=True, use_cluster_embedding=True):
    model.eval()
    scalers = ScalerManager().get_all_scalers()

    predictions_list, true_values_list = [], []
    metadata, current_meta_idx = initialize_metadata(dataloader) if extract_metadata else (None, 0)

    with torch.no_grad():
        for batch in dataloader:
            batch_data = prepare_batch(batch, use_cluster_embedding)
            batch_preds, batch_true = process_batch(model, batch_data, scalers)
            
            predictions_list.append(batch_preds)
            true_values_list.append(batch_true)
            
            if extract_metadata:
                current_meta_idx = update_metadata(
                    metadata, 
                    batch_data['future'],
                    current_meta_idx
                )
    
    all_preds = np.vstack(predictions_list)
    all_true = np.vstack(true_values_list)
    
    return all_preds, all_true, metadata

def initialize_metadata(dataloader: torch.utils.data.DataLoader) -> Dict[str, np.ndarray]:
    total_points = len(dataloader.dataset) * PRED_LENGTH
    metadata = {
        'DayOfWeek': np.empty(total_points, dtype=int),
        'Holiday': np.empty(total_points, dtype=int),
        'IsPeak': np.empty(total_points, dtype=int),
    }
    return metadata, 0

def prepare_batch(batch: Tuple, use_cluster_embedding: bool) -> Dict[str, torch.Tensor]:
    """Move batch data to device and organize into dictionary"""
    if use_cluster_embedding:
        X, future, y, cluster_id = batch
        return {
            'X': X.to(DEVICE),
            'future': future.to(DEVICE),
            'y': y.to(DEVICE),
            'cluster_id': cluster_id.to(DEVICE)
        }
    else:
        X, future, y = batch
        return {
            'X': X.to(DEVICE),
            'future': future.to(DEVICE),
            'y': y.to(DEVICE),
            'cluster_id': None
        }

def process_batch(
    model: torch.nn.Module,
    batch_data: Dict[str, torch.Tensor],
    scalers: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions and inverse transform for a batch"""
    predictions = model(
        batch_data['X'],
        batch_data['future'],
        batch_data['cluster_id']
    )
    
    predictions = predictions.cpu().numpy().reshape(-1, PRED_LENGTH, 2)
    true_values = batch_data['y'].cpu().numpy()
    
    if batch_data['cluster_id'] is not None:
        predictions = inverse_transform_by_cluster(predictions, batch_data['cluster_id'], scalers)
        true_values = inverse_transform_by_cluster(true_values, batch_data['cluster_id'], scalers)
    
    return predictions, true_values
    
def inverse_transform_by_cluster(
    values: np.ndarray,
    cluster_id: torch.Tensor,
    scalers: Dict
) -> np.ndarray:
    """Apply inverse transform grouped by cluster"""
    cluster_ids = cluster_id.cpu().numpy().ravel()
    unique_clusters = np.unique(cluster_ids)
    transformed = np.empty_like(values)
    
    for cluster in unique_clusters:
        mask = (cluster_ids == cluster)
        if np.any(mask):
            transformed[mask] = inverse_transform_batch(
                values[mask].reshape(-1, 2),
                scalers[cluster]
            ).reshape(-1, PRED_LENGTH, 2)
    
    return transformed

def inverse_transform_batch(predictions: np.ndarray, scaler) -> np.ndarray:
    """Vectorized inverse transform for a batch of predictions"""
    num_samples = predictions.shape[0]
    dummy_features = np.zeros((num_samples, scaler.n_features_in_ - 2))
    combined = np.hstack([predictions, dummy_features])
    return scaler.inverse_transform(combined)[:, :2]

def update_metadata(
    metadata: Dict[str, np.ndarray],
    future: torch.Tensor,
    current_idx: int
) -> int:
    """Update metadata arrays with batch data"""
    future_cpu = future.cpu().numpy()
    batch_size = future_cpu.shape[0] * future_cpu.shape[1]
    
    metadata['DayOfWeek'][current_idx:current_idx+batch_size] = \
        (future_cpu[:, :, 0] * 6 + 1).round().astype(int).ravel()
    metadata['Holiday'][current_idx:current_idx+batch_size] = \
        future_cpu[:, :, 8].astype(int).ravel()
    metadata['IsPeak'][current_idx:current_idx+batch_size] = \
        future_cpu[:, :, 4].astype(int).ravel()
    
    return current_idx + batch_size