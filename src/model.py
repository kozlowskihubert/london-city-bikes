import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, future_feature_size, output_size, pred_length, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length
        
        # LSTM layer for historical data
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Linear layer for LSTM output
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        
        # Process future features
        self.future_fc = nn.Linear(future_feature_size * pred_length, hidden_size)
        
        # Combined processing
        self.combined_fc = nn.Linear(hidden_size * 2, hidden_size)
        
        # Final output layer
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Activation
        self.relu = nn.ReLU()

    def forward(self, x, future_features):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Process LSTM output
        lstm_out = out[:, -1, :]  # Take the output of the last time step
        lstm_out = self.relu(self.fc1(lstm_out))
        
        # Process future features
        batch_size = future_features.size(0)
        future_flat = future_features.reshape(batch_size, -1)  # Flatten to (batch_size, pred_length * feature_size)
        future_out = self.relu(self.future_fc(future_flat))
        
        # Combine both sources of information
        combined = torch.cat((lstm_out, future_out), dim=1)
        combined = self.relu(self.combined_fc(combined))
        
        # Final output
        out = self.fc2(combined)
        
        return out
    

class LinearWeightedMSELoss(nn.Module):
    def __init__(self, base_weight=1.0, slope=0.5):
        super().__init__()
        self.base_weight = base_weight 
        self.slope = slope

    def forward(self, y_pred, y_true):
        squared_errors = (y_pred - y_true) ** 2
        
        weights = self.base_weight + self.slope * y_true 
        return torch.mean(weights * squared_errors)
    

class ExponentialWeightedMSELoss(nn.Module):
    def __init__(self, base_weight=1.0, scale=0.05):
        super().__init__()
        self.base_weight = base_weight  # Weight at y_true=0
        self.scale = scale            # Controls exponential steepness

    def forward(self, y_pred, y_true):
        squared_errors = (y_pred - y_true) ** 2
        weights = self.base_weight * torch.exp(self.scale * y_true)  # Exponential scaling
        return torch.mean(weights * squared_errors)