import mlflow
import mlflow.pytorch

def log_model_architecture(model):
    """
    Log model architecture details to MLflow
    """
    model_summary = str(model)
    mlflow.log_text(model_summary, "model_architecture.txt")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    mlflow.log_param("total_parameters", total_params)
    mlflow.log_param("trainable_parameters", trainable_params)
    mlflow.log_param("hidden_size", model.hidden_size)
    mlflow.log_param("num_layers", model.num_layers)

def log_train_parameters(num_epochs, optimizer, criterion, batch_size):
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", optimizer.param_groups[0]['lr'])
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("criterion", criterion.__class__.__name__)
    mlflow.log_param("optimizer", optimizer.__class__.__name__)

def log_training_metrics(metrics, epoch):
    mlflow.log_metric("train_loss", metrics["train_loss"], step=epoch)
    mlflow.log_metric("train_mae", metrics["train_mae"], step=epoch)
    mlflow.log_metric("val_loss", metrics["val_loss"], step=epoch)
    mlflow.log_metric("val_mae", metrics["val_mae"], step=epoch)
    mlflow.log_metric("val_r2", metrics["val_r2"], step=epoch)

def log_trained_model(model, model_name):
    mlflow.pytorch.log_model(model, model_name)

def log_true_and_predictions_values(true_values, predictions):
    mlflow.log_artifact(true_values)
    mlflow.log_artifact(predictions)

def log_overall_prediction_metrics(metrics):
    mlflow.log_metric("MAE", metrics["MAE"])
    mlflow.log_metric("MSE", metrics["MSE"])
    mlflow.log_metric("RMSE", metrics["RMSE"])
    mlflow.log_metric("R2", metrics["R2"])
    mlflow.log_metric("sMAPE", metrics["sMAPE"])
    
    mlflow.log_metric("Starts_MAE", metrics["Starts_MAE"])
    mlflow.log_metric("Starts_RMSE", metrics["Starts_RMSE"])
    mlflow.log_metric("Starts_R2", metrics["Starts_R2"])
    mlflow.log_metric("Starts_sMAPE", metrics["Starts_sMAPE"])
    
    mlflow.log_metric("Ends_MAE", metrics["Ends_MAE"])
    mlflow.log_metric("Ends_RMSE", metrics["Ends_RMSE"])
    mlflow.log_metric("Ends_R2", metrics["Ends_R2"])
    mlflow.log_metric("Ends_sMAPE", metrics["Ends_sMAPE"])

def log_day_prediction_metrics(day, day_mae, day_rmse, day_r2):
    mlflow.log_metric(f"Day{day}_MAE", day_mae)
    mlflow.log_metric(f"Day{day}_RMSE", day_rmse)
    mlflow.log_metric(f"Day{day}_R2", day_r2)

def log_weekend_prediction_metrics(weekend_mae, weekend_rmse, weekend_r2):
    mlflow.log_metric("Weekend_MAE", weekend_mae)
    mlflow.log_metric("Weekend_RMSE", weekend_rmse)
    mlflow.log_metric("Weekend_R2", weekend_r2)

def log_weekday_prediction_metrics(weekday_mae, weekday_rmse, weekday_r2):
    mlflow.log_metric("Weekday_MAE", weekday_mae)
    mlflow.log_metric("Weekday_RMSE", weekday_rmse)
    mlflow.log_metric("Weekday_R2", weekday_r2)

def log_peak_prediction_metrics(peak_mae, peak_rmse, peak_smape):
    mlflow.log_metric("Peak_MAE", peak_mae)
    mlflow.log_metric("Peak_RMSE", peak_rmse)
    mlflow.log_metric("Peak_sMAPE", peak_smape)

def log_no_peak_prediction_metrics(non_peak_mae, non_peak_rmse, non_peak_smape):
    mlflow.log_metric("No_Peak_MAE", non_peak_mae)
    mlflow.log_metric("No_Peak_RMSE", non_peak_rmse)
    mlflow.log_metric("No_Peak_sMAPE", non_peak_smape)

def log_timepoint_prediction_metric(t_mae, t_rmse, t_r2, step):
    mlflow.log_metric(f"Timepoint_MAE", t_mae, step=step)
    mlflow.log_metric(f"Timepoint_RMSE", t_rmse, step=step)
    mlflow.log_metric(f"Timepoint_R2", t_r2, step=step)