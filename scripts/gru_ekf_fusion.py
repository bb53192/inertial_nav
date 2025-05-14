
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === Load dataset ===
df = pd.read_csv("val_path_rectangle_8_reverse.csv")
X = df[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','sin_theta','cos_theta']].values
y = df[['vel_x','vel_y']].values

# === Normalize input ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Segment sequences ===
seq_len = 50
X_seq = []
y_seq = []
for i in range(0, len(X_scaled) - seq_len, seq_len):
    X_seq.append(X_scaled[i:i+seq_len])
    y_seq.append(y[i+seq_len-1])
X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)

# === Load trained GRU model ===
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

model = PathGRU()
model.load_state_dict(torch.load("best_path_gru_model.pth"))
model.eval()

# === Predict velocity from GRU ===
with torch.no_grad():
    vel_preds = model(torch.tensor(X_seq)).numpy()

# === EKF initialization ===
dt = 1.0  # time step
n = len(vel_preds)
x_est = np.zeros((n+1, 4))  # [pos_x, pos_y, vel_x, vel_y]
P = np.eye(4) * 0.1
Q = np.diag([0.01, 0.01, 0.1, 0.1])  # process noise
R = np.diag([0.2, 0.2])              # measurement noise

# === EKF Loop ===
ekf_positions = [x_est[0, :2]]
for t in range(n):
    # === Predict step ===
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    x_pred = F @ x_est[t]
    x_pred[2:] = vel_preds[t, :2]  # use GRU velocity as "known input"

    P = F @ P @ F.T + Q

    # === Update step === (optional: using velocity measurement)
    H = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1]])
    z = y_seq[t]
    y_err = z - H @ x_pred
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_est[t+1] = x_pred + K @ y_err
    P = (np.eye(4) - K @ H) @ P

    ekf_positions.append(x_est[t+1, :2])

# === Ground truth integration ===
gt_positions = [np.array([0.0, 0.0])]
for v in y_seq:
    gt_positions.append(gt_positions[-1] + v * dt)
gt_positions = np.array(gt_positions)
ekf_positions = np.array(ekf_positions)

# === Plot result ===
plt.figure(figsize=(10,6))
plt.plot(gt_positions[:,0], gt_positions[:,1], label='Ground Truth')
plt.plot(ekf_positions[:,0], ekf_positions[:,1], label='GRU + EKF', linestyle='--')
plt.title("Putanja robota: Ground Truth vs GRU + EKF")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.savefig("gru_ekf_fusion_path.png")
plt.show()
