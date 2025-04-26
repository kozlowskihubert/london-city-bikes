import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score
from src.mlflow_logging import log_training_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs=10, use_cluster_embedding=True, patience=5):
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []

    best_val_loss = float('inf')
    best_model_weights = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_mae = 0

        for batch in train_dataloader:
            if use_cluster_embedding:
                batch_X, batch_future, batch_y, cluster_id = batch
                cluster_id = cluster_id.to(DEVICE)
            else:
                batch_X, batch_future, batch_y = batch
                cluster_id = None

            batch_X = batch_X.to(DEVICE)
            batch_future = batch_future.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_X, batch_future, cluster_id)
            loss = criterion(outputs, batch_y.view(batch_y.size(0), -1))
            mae = torch.nn.functional.l1_loss(outputs, batch_y.view(batch_y.size(0), -1), reduction='mean')

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mae += mae.item()

        epoch_loss /= len(train_dataloader)
        epoch_mae /= len(train_dataloader)
        train_losses.append(epoch_loss)
        train_maes.append(epoch_mae)

        model.eval()
        val_loss = 0
        val_mae = 0
        val_outputs_all = []
        val_targets_all = []

        with torch.no_grad():
            for batch in val_dataloader:
                if use_cluster_embedding:
                    batch_X, batch_future, batch_y, cluster_id = batch
                    cluster_id = cluster_id.to(DEVICE)
                else:
                    batch_X, batch_future, batch_y = batch
                    cluster_id = None

                batch_X = batch_X.to(DEVICE)
                batch_future = batch_future.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                outputs = model(batch_X, batch_future, cluster_id)
                loss = criterion(outputs, batch_y.view(batch_y.size(0), -1))
                mae = torch.nn.functional.l1_loss(outputs, batch_y.view(batch_y.size(0), -1), reduction='mean')

                val_loss += loss.item()
                val_mae += mae.item()
                
                val_outputs_all.append(outputs.cpu().numpy())
                val_targets_all.append(batch_y.view(batch_y.size(0), -1).cpu().numpy())

        val_loss /= len(val_dataloader)
        val_mae /= len(val_dataloader)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        val_outputs_concat = np.concatenate(val_outputs_all, axis=0)
        val_targets_concat = np.concatenate(val_targets_all, axis=0)
        val_r2 = r2_score(val_targets_concat, val_outputs_concat)

        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_loss:.4f}, Train MAE: {epoch_mae:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')

        epoch_metrics = {
            "train_loss": epoch_loss,
            "train_mae": epoch_mae,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_r2": val_r2,
        }

        log_training_metrics(epoch_metrics, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if patience and epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f'Best model from epoch {best_epoch+1} with val loss: {best_val_loss:.4f}')
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
    }

    plot_losses(train_losses, val_losses, train_maes, val_maes)

    return model, checkpoint


def plot_losses(train_losses, val_losses, train_maes, val_maes):
    # Plot training & validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()
    
    # Plot training & validation MAE
    plt.figure(figsize=(10, 5))
    plt.plot(train_maes, label="Train MAE")
    plt.plot(val_maes, label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Training and Validation MAE")
    plt.legend()
    plt.show()


def load_checkpoint_and_resume_training(checkpoint_path, model, criterion, optimizer, train_dataloader, val_dataloader, additional_epochs):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    model, checkpoint = train_model(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=additional_epochs,
        use_cluster_embedding=True,
    )
    
    return model, checkpoint
