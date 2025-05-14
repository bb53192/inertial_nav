
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("synthetic_imu_advanced_testset.csv")

# Ensure vel_z exists
if 'vel_z' not in df.columns:
    df['vel_z'] = 0.0

# Parameters
sequence_length = 50
input_size = 6
output_size = 3  # vel_x, vel_y, vel_z

# Prepare sequences
X = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
y = df[['vel_x', 'vel_y', 'vel_z']].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_seq = []
y_seq = []
for i in range(0, len(X) - sequence_length, sequence_length):
    X_seq.append(X[i:i+sequence_length])
    y_seq.append(y[i+sequence_length-1])

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)

# Create DataLoader
dataset = TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq))
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# GRU model
class VelocityGRU(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, output_size=3):
        super(VelocityGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# Load model
model = VelocityGRU()
model.load_state_dict(torch.load("velocity_gru_model.pth"))
model.eval()

# Predict
predictions = []
targets = []

with torch.no_grad():
    for xb, yb in loader:
        preds = model(xb)
        predictions.append(preds.numpy()[0])
        targets.append(yb.numpy()[0])

predictions = np.array(predictions)
targets = np.array(targets)

# Plot results
for i, axis in enumerate(['x', 'y', 'z']):
    plt.figure(figsize=(10, 5))
    plt.plot(predictions[:, i], label=f'Predicted vel_{axis}')
    plt.plot(targets[:, i], label=f'True vel_{axis}')
    plt.title(f"Predikcija brzine u {axis} smjeru")
    plt.legend()
    plt.grid(True)
    plt.show()

# Calculate RMSE
for i, axis in enumerate(['x', 'y', 'z']):
    rmse = np.sqrt(np.mean((predictions[:, i] - targets[:, i])**2))
    print(f"RMSE vel_{axis}: {rmse:.4f}")
