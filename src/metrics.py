import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.mlflow_logging import (
    log_overall_prediction_metrics,
    log_day_prediction_metrics,
    log_weekend_prediction_metrics,
    log_weekday_prediction_metrics,
    log_peak_prediction_metrics,
    log_no_peak_prediction_metrics,
    log_timepoint_prediction_metric,
)

def calculate_smape(y_true, y_pred, epsilon=1e-10):
    """
    Calculate Symmetric Mean Absolute Percentage Error
    Handles zero values by adding a small epsilon
    """
    return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + epsilon))

def calculate_peak_metrics(y_true, y_pred, is_peak, peak_value=1):
    """
    Calculate metrics specifically for peak periods
    """
    peak_indices = np.where(is_peak == peak_value)[0]
    if len(peak_indices) == 0:
        return None, None, None
    
    peak_true = y_true[peak_indices]
    peak_pred = y_pred[peak_indices]
    
    peak_mae = mean_absolute_error(peak_true, peak_pred)
    peak_rmse = np.sqrt(mean_squared_error(peak_true, peak_pred))
    peak_smape = calculate_smape(peak_true, peak_pred)
    
    return peak_mae, peak_rmse, peak_smape

def evaluate_predictions(all_true, all_preds, metadata=None, timepoint_eval=True):
    """
    Evaluate predictions with multiple metrics and breakdowns
    
    Parameters:
    - all_true: numpy array of true values (samples, pred_length, 2)
    - all_preds: numpy array of predictions (samples, pred_length, 2)
    - metadata: additional information like DayOfWeek, Holiday for segmentation
    - timepoint_eval: whether to evaluate metrics for each timepoint in the prediction horizon
    """
    # Reshape for overall metrics
    all_true_reshaped = all_true.reshape(-1, all_true.shape[-1])  # (samples*pred_length, 2)
    all_preds_reshaped = all_preds.reshape(-1, all_preds.shape[-1])  # (samples*pred_length, 2)
    
    # Calculate overall metrics
    mae = mean_absolute_error(all_true_reshaped, all_preds_reshaped)
    mse = mean_squared_error(all_true_reshaped, all_preds_reshaped)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_true_reshaped, all_preds_reshaped)
    smape = calculate_smape(all_true_reshaped, all_preds_reshaped)
    
    # Log overall metrics
    metrics = {}
    metrics["MAE"] = mae
    metrics["MSE"] = mse
    metrics["RMSE"] = rmse
    metrics["R2"] = r2
    metrics["sMAPE"] = smape
    
    # Calculate metrics separately for Starts and Ends
    starts_true = all_true_reshaped[:, 0]
    starts_pred = all_preds_reshaped[:, 0]
    ends_true = all_true_reshaped[:, 1]
    ends_pred = all_preds_reshaped[:, 1]
    
    # Metrics for Starts
    starts_mae = mean_absolute_error(starts_true, starts_pred)
    starts_mse = mean_squared_error(starts_true, starts_pred)
    starts_rmse = np.sqrt(starts_mse)
    starts_r2 = r2_score(starts_true, starts_pred)
    starts_smape = calculate_smape(starts_true, starts_pred)
    
    # Metrics for Ends
    ends_mae = mean_absolute_error(ends_true, ends_pred)
    ends_mse = mean_squared_error(ends_true, ends_pred)
    ends_rmse = np.sqrt(ends_mse)
    ends_r2 = r2_score(ends_true, ends_pred)
    ends_smape = calculate_smape(ends_true, ends_pred)
    
    # Log metrics for Starts and Ends
    metrics["Starts_MAE"] = starts_mae
    metrics["Starts_RMSE"] = starts_rmse
    metrics["Starts_R2"] = starts_r2
    metrics["Starts_sMAPE"] = starts_smape

    metrics["Ends_MAE"] = ends_mae
    metrics["Ends_RMSE"] = ends_rmse
    metrics["Ends_R2"] = ends_r2
    metrics["Ends_sMAPE"] = ends_smape

    log_overall_prediction_metrics(metrics)

    if metadata is not None:
        for day in range(1,8):
            day_indices = np.where(metadata['DayOfWeek'] == day)[0]
            if len(day_indices) > 0:
                day_true = all_true_reshaped[day_indices]
                day_pred = all_preds_reshaped[day_indices]
                
                day_mae = mean_absolute_error(day_true, day_pred)
                day_rmse = np.sqrt(mean_squared_error(day_true, day_pred))
                day_r2 = r2_score(day_true, day_pred)
                
                log_day_prediction_metrics(day, day_mae, day_rmse, day_r2)
        
        weekend_indices = np.where((metadata['DayOfWeek'] >= 5) | (metadata['Holiday'] == 1))[0]
        weekday_indices = np.where((metadata['DayOfWeek'] < 5) & (metadata['Holiday'] == 0))[0]
        
        if len(weekend_indices) > 0:
            weekend_true = all_true_reshaped[weekend_indices]
            weekend_pred = all_preds_reshaped[weekend_indices]
            
            weekend_mae = mean_absolute_error(weekend_true, weekend_pred)
            weekend_rmse = np.sqrt(mean_squared_error(weekend_true, weekend_pred))
            weekend_r2 = r2_score(weekend_true, weekend_pred)
            
            log_weekend_prediction_metrics(weekend_mae, weekend_rmse, weekend_r2)
        
        if len(weekday_indices) > 0:
            weekday_true = all_true_reshaped[weekday_indices]
            weekday_pred = all_preds_reshaped[weekday_indices]
            
            weekday_mae = mean_absolute_error(weekday_true, weekday_pred)
            weekday_rmse = np.sqrt(mean_squared_error(weekday_true, weekday_pred))
            weekday_r2 = r2_score(weekday_true, weekday_pred)
            
            log_weekday_prediction_metrics(weekday_mae, weekday_rmse, weekday_r2)
        
        if 'IsPeak' in metadata:
            peak_mae, peak_rmse, peak_smape = calculate_peak_metrics(
                all_true_reshaped, all_preds_reshaped, metadata['IsPeak']
            )
            
            if peak_mae is not None:
                log_peak_prediction_metrics(peak_mae, peak_rmse, peak_smape)

            non_peak_mae, non_peak_rmse, non_peak_smape = calculate_peak_metrics(
                all_true_reshaped, all_preds_reshaped, metadata['IsPeak'], 0
            )
            
            if non_peak_mae is not None:
                log_no_peak_prediction_metrics(non_peak_mae, non_peak_rmse, non_peak_smape)
                
    if timepoint_eval:
        for t in range(all_true.shape[1]):
            t_true = all_true[:, t, :]
            t_pred = all_preds[:, t, :]
            
            t_mae = mean_absolute_error(t_true, t_pred)
            t_rmse = np.sqrt(mean_squared_error(t_true, t_pred))
            t_r2 = r2_score(t_true, t_pred)

            log_timepoint_prediction_metric(t_mae, t_rmse, t_r2, t)
  
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'sMAPE': smape
    }
