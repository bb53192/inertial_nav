import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Učitaj CSV
df = pd.read_csv("val_path_rectangle_8_reverse.csv")
X = df[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','sin_theta','cos_theta']].values
y = df[['vel_x', 'vel_y', 'vel_z']].values

# 2. Skaliraj
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Segmentacija
sequence_length = 50
X_seq, y_seq = [], []
for i in range(0, len(X_scaled) - sequence_length, sequence_length):
    X_seq.append(X_scaled[i:i+sequence_length])
    y_seq.append(y[i+sequence_length-1])
X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)

# 4. GRU model (isti kao u treningu)
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

# 5. Učitaj model
model = PathGRU()
model.load_state_dict(torch.load("best_path_gru_model.pth"))
model.eval()

# 6. Predikcija
with torch.no_grad():
    predictions = model(torch.tensor(X_seq)).numpy()

# 7. Integracija brzine u putanju
dt = 1.0
gt_pos = [np.array([0, 0])]
pred_pos = [np.array([0, 0])]
for gt_v, pr_v in zip(y_seq, predictions):
    gt_pos.append(gt_pos[-1] + gt_v[:2]*dt)
    pred_pos.append(pred_pos[-1] + pr_v[:2]*dt)
gt_pos = np.array(gt_pos)
pred_pos = np.array(pred_pos)

# 8. Plot
plt.figure(figsize=(10, 6))
plt.plot(gt_pos[:,0], gt_pos[:,1], label="Ground Truth")
plt.plot(pred_pos[:,0], pred_pos[:,1], label="Predicted", linestyle='--')
plt.title("Putanja robota: stvarna vs GRU predikcija")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()
