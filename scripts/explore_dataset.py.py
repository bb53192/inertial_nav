import pandas as pd
import matplotlib.pyplot as plt

# Učitaj podatke
df = pd.read_csv("synthetic_imu_dataset.csv")

# Prikaz prvih 10 redova
print("🔍 Prvih 10 redova podataka:")
print(df.head(10))

# Opis statistika
print("\n📈 Statistički pregled:")
print(df.describe())

# Provjera kolona
print("\n🧱 Kolone u skupu:")
print(df.columns.tolist())

# Npr. jedan prozor od 50 uzastopnih uzoraka
sample_window = df.iloc[0:50]

# Prikaz senzora
plt.figure(figsize=(12, 5))
plt.plot(sample_window["acc_x"], label="acc_x")
plt.plot(sample_window["acc_y"], label="acc_y")
plt.plot(sample_window["acc_z"], label="acc_z")
plt.title("Primjer akcelerometarskih mjerenja")
plt.legend()
plt.grid(True)
plt.show()

# Prikaz brzina
plt.figure(figsize=(12, 5))
plt.plot(sample_window["vel_x"], label="vel_x (ground truth)")
plt.plot(sample_window["vel_y"], label="vel_y (ground truth)")
plt.title("Stvarna brzina tijekom prozora")
plt.legend()
plt.grid(True)
plt.show()
