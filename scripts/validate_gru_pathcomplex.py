
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load validation dataset
df = pd.read_csv("val_path_rectangle_8_reverse.csv")
X = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'sin_theta', 'cos_theta']].values
y = df[['vel_x', 'vel_y', 'vel_z']].values

# Normalize input
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Segment into sequences
sequence_length = 50
X_seq, y_seq = [], []
for i in range(0, len(X) - sequence_length, sequence_length):
    X_seq.append(X[i:i+sequence_length])
    y_seq.append(y[i+sequence_length-1])
X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)

# Create DataLoader
test_loader = DataLoader(TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq)), batch_size=1)

# GRU Model definition (same as trained)
class PathGRU(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, output_size=3):
        super(PathGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.norm(out)
        return self.fc(out)

# Load model
model = PathGRU()
model.load_state_dict(torch.load("best_path_gru_model.pth"))
model.eval()

# Predict
predictions = []
targets = []
with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb)
        predictions.append(pred.numpy()[0])
        targets.append(yb.numpy()[0])

predictions = np.array(predictions)
targets = np.array(targets)

# Plot predictions
for i, axis in enumerate(['x', 'y', 'z']):
    plt.figure(figsize=(10, 4))
    plt.plot(predictions[:, i], label=f'Predicted vel_{axis}')
    plt.plot(targets[:, i], label=f'True vel_{axis}')
    plt.title(f'Prediction vs Ground Truth - vel_{axis}')
    plt.xlabel("Sample")
    plt.ylabel(f"vel_{axis}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'path_gru_prediction_vel_{axis}.png')
    plt.show()

# RMSE per component
for i, axis in enumerate(['x', 'y', 'z']):
    rmse = np.sqrt(np.mean((predictions[:, i] - targets[:, i])**2))
    print(f"RMSE vel_{axis}: {rmse:.4f}")
