
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("synthetic_imu_advanced_motions.csv")

# Parameters
sequence_length = 50
input_size = 6
output_size = 3

# Prepare sequences
X = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
y = df[['vel_x', 'vel_y', 'vel_z']].values

# Reshape into sequences
X_seq = []
y_seq = []
for i in range(0, len(X) - sequence_length, sequence_length):
    X_seq.append(X[i:i+sequence_length])
    y_seq.append(y[i+sequence_length-1])  # use last velocity in window

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)

# Create DataLoader
dataset = TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq))
loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

# GRU model
class VelocityGRU(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, output_size=3):
        super(VelocityGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # last timestep
        out = self.norm(out)
        out = self.fc(out)
        return out

# Initialize
model = VelocityGRU()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
num_epochs = 40
losses = []
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # OPTIONAL: Validacija ako imaš validacijski loader
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            val_loss += criterion(preds, yb).item()
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")
    """

    # Spremi najbolji model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_velocity_gru_model.pth")
        print("✔️ Model saved.")

# Plot loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()

torch.save(model.state_dict(), "velocity_gru_model.pth")

