import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, future_feature_size, output_size, pred_length,
                 num_layers=1, dropout=0.2, use_cluster_embedding=False, num_clusters=30, cluster_emb_size=8):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length
        self.use_cluster_embedding = use_cluster_embedding

        # LSTM layer for historical data
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)

        # Future feature processing
        self.future_fc = nn.Linear(future_feature_size * pred_length, hidden_size)

        # Optional cluster embedding
        if use_cluster_embedding:
            self.cluster_embedding = nn.Embedding(num_clusters, cluster_emb_size)
            combined_input_size = hidden_size * 2 + cluster_emb_size
        else:
            combined_input_size = hidden_size * 2

        self.combined_fc = nn.Linear(combined_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x, future_features, cluster_id=None):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        lstm_out = self.relu(self.fc1(out[:, -1, :]))

        batch_size = future_features.size(0)
        future_flat = future_features.reshape(batch_size, -1)
        future_out = self.relu(self.future_fc(future_flat))

        combined = torch.cat((lstm_out, future_out), dim=1)

        if self.use_cluster_embedding and cluster_id is not None:
            cluster_emb = self.cluster_embedding(cluster_id)
            combined = torch.cat((combined, cluster_emb), dim=1)

        combined = self.relu(self.combined_fc(combined))
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