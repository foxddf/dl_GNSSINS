import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

imu_path = "../outputs/synthetic/imu_and_ins.csv"
gnss_path = "../outputs/synthetic/gnss.csv"
ai_path  = "../outputs/artifacts/drift_corrections.csv"

df_imu = pd.read_csv(imu_path)
df_gnss = pd.read_csv(gnss_path)
df_ai  = pd.read_csv(ai_path)

time = df_imu["time"].values
truth = df_imu[["pos_truth_e", "pos_truth_n", "pos_truth_u"]].values
ins_dr = df_imu[["pos_est_e", "pos_est_n", "pos_est_u"]].values

# GNSS (1 Hz)
t_gnss = df_gnss["time"].values
gnss_meas = df_gnss[["gnss_pos_e", "gnss_pos_n", "gnss_pos_u"]].values

# AI 보정 (drift_corrections.csv는 IMU 타임스텝 일부만 존재)
t_ai = df_ai["time"].values
pos_ai = df_ai[["pos_corr_e", "pos_corr_n", "pos_corr_u"]].values

# --- 1) INS DR error ---
err_ins = np.linalg.norm(ins_dr - truth, axis=1)

# --- 2) GNSS/INS LC baseline (아주 단순한 1Hz 업데이트) ---
pos_lc = ins_dr.copy()

# GNSS outage 구간 정의 (Generate와 동일하게)
outage_start, outage_end = 40.0, 60.0

for k, t_g in enumerate(t_gnss):
    if outage_start <= t_g <= outage_end:
        # outage 구간 GNSS는 NaN, 업데이트 안 함
        continue
    # 가장 가까운 IMU index에 GNSS 업데이트 적용
    idx = np.argmin(np.abs(time - t_g))
    if not np.any(np.isnan(gnss_meas[k])):
        pos_lc[idx] = gnss_meas[k]

err_lc = np.linalg.norm(pos_lc - truth, axis=1)

# --- 3) outage 구간에서만 LC + AI drift 보정 ---
pos_lc_ai = pos_lc.copy()

# AI 시각을 IMU 인덱스에 매핑
for k, t_a in enumerate(t_ai):
    if not (outage_start <= t_a <= outage_end):
        continue  # outage가 아닐 땐 굳이 AI를 쓰지 않음
    idx = np.argmin(np.abs(time - t_a))
    pos_lc_ai[idx] = pos_ai[k]

err_lc_ai = np.linalg.norm(pos_lc_ai - truth, axis=1)

# --- outage 마스크 (IMU 타임 기준) ---
outage_mask = (time >= outage_start) & (time <= outage_end)

# RMSE (outage 구간만)
rmse_ins_outage   = np.sqrt(np.mean(err_ins[outage_mask] ** 2))
rmse_lc_outage    = np.sqrt(np.mean(err_lc[outage_mask] ** 2))
rmse_lc_ai_outage = np.sqrt(np.mean(err_lc_ai[outage_mask] ** 2))

print("=== RMSE (GNSS outage 구간에서만) ===")
print(f"INS DR only          : {rmse_ins_outage:.3f} m")
print(f"GNSS/INS LC baseline : {rmse_lc_outage:.3f} m")
print(f"LC + AI drift        : {rmse_lc_ai_outage:.3f} m")

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(time, err_ins,    label="INS DR only")
plt.plot(time, err_lc,     label="GNSS/INS LC baseline", linestyle="--")
plt.plot(time, err_lc_ai,  label="GNSS/INS LC + AI (outage only)", linestyle="-.")

plt.fill_between(time, 0, err_ins.max(),
                 where=outage_mask, alpha=0.1, label="GNSS outage")

plt.xlabel("Time [s]")
plt.ylabel("Position Error Norm [m]")
plt.title("Position Error Comparison (INS vs LC vs LC+AI)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
