import pandas as pd
import matplotlib.pyplot as plt

# 평가 결과 CSV 경로
csv_path = "outputs/drift_corrections.csv"

df = pd.read_csv(csv_path)

time = df["time"].values

drift_true = df[["drift_true_e", "drift_true_n", "drift_true_u"]].values
drift_pred = df[["drift_pred_e", "drift_pred_n", "drift_pred_u"]].values

axes_names = ["East (E)", "North (N)", "Up (U)"]

plt.figure(figsize=(12, 8))

for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(time, drift_true[:, i], label="drift_true", linewidth=1.5)
    plt.plot(time, drift_pred[:, i], label="drift_pred", linewidth=1.0, linestyle="--")
    plt.ylabel(f"Drift {axes_names[i]} [m]")
    plt.grid(True, alpha=0.3)
    if i == 0:
        plt.title("INS Drift: True vs Predicted")
    if i == 2:
        plt.xlabel("Time [s]")
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()
