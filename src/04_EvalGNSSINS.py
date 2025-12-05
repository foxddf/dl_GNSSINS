import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")   # 또는 "Qt5Agg" 등, 설치된 것 중 하나
import matplotlib.pyplot as plt



# ------------------------------------------------------------
# 설정값 (필요하면 여기만 바꿔서 튜닝)
# ------------------------------------------------------------
IMU_CSV = "../outputs/synthetic/imu_and_ins.csv"
GNSS_CSV = "../outputs/synthetic/gnss.csv"
AI_CSV   = "../outputs/artifacts/drift_corrections.csv"

OUTAGE_START = 40.0   # s, 01_GenerateSyntheticData.py와 일치
OUTAGE_END   = 60.0   # s

# EKF 프로세스/측정 노이즈 (대략적인 값, 튜닝 가능)
SIGMA_V_RW = 0.05     # [m/s]/sqrt(s), 속도 오차 랜덤워크
SIGMA_GNSS = 1.5      # [m], GNSS 위치 측정 노이즈 (SimConfig와 맞춤)


# ------------------------------------------------------------
# 데이터 로드
# ------------------------------------------------------------
df_imu = pd.read_csv(IMU_CSV)
df_gnss = pd.read_csv(GNSS_CSV)
df_ai = pd.read_csv(AI_CSV)

t_imu = df_imu["time"].values          # IMU/INS 시간축 (100 Hz)
truth = df_imu[["pos_truth_e", "pos_truth_n", "pos_truth_u"]].values
ins_dr = df_imu[["pos_est_e", "pos_est_n", "pos_est_u"]].values

t_gnss = df_gnss["time"].values        # GNSS 시간축 (1 Hz)
gnss_meas = df_gnss[["gnss_pos_e", "gnss_pos_n", "gnss_pos_u"]].values

t_ai = df_ai["time"].values            # AI 예측이 존재하는 시간축 (stride 간격)
pos_ai = df_ai[["pos_corr_e", "pos_corr_n", "pos_corr_u"]].values


# ------------------------------------------------------------
# 1) INS DR only 에러
# ------------------------------------------------------------
err_ins = np.linalg.norm(ins_dr - truth, axis=1)


# ------------------------------------------------------------
# 2) GNSS 측정을 IMU 시간축으로 매핑 (nearest-neighbor, outage 고려)
# ------------------------------------------------------------
gnss_up = np.full_like(ins_dr, np.nan, dtype=float)  # (N_imu, 3)

dt_gnss = np.median(np.diff(t_gnss)) if len(t_gnss) > 1 else 1.0
half_dt = 0.5 * dt_gnss

for j, t_g in enumerate(t_gnss):
    meas = gnss_meas[j]
    # outage 구간에서는 이미 NaN 들어있을 것 (simulate_gnss에서 처리)
    if np.any(np.isnan(meas)):
        continue
    # 가장 가까운 IMU index 찾기 (시간 차이가 half_dt 이내인 경우만)
    idx = np.argmin(np.abs(t_imu - t_g))
    if abs(t_imu[idx] - t_g) <= half_dt:
        gnss_up[idx] = meas

# GNSS outage 구간 강제 NaN 처리 (안전장치)
outage_mask = (t_imu >= OUTAGE_START) & (t_imu <= OUTAGE_END)
gnss_up[outage_mask] = np.nan

gnss_available = ~np.isnan(gnss_up[:, 0])


# ------------------------------------------------------------
# 3) EKF 기반 GNSS/INS LC (오차상태: [δp, δv])
# ------------------------------------------------------------
N = len(t_imu)
# 상태: [δp_e, δp_n, δp_u, δv_e, δv_n, δv_u]^T
x = np.zeros(6)
P = np.diag([1.0, 1.0, 1.0,   # 위치 오차 초기 1 m
             0.5, 0.5, 0.5])  # 속도 오차 초기 0.5 m/s

sigma_v_rw = SIGMA_V_RW
Q_v = (sigma_v_rw**2) * np.eye(3)   # 속도 오차 random walk 공분산

R = (SIGMA_GNSS**2) * np.eye(3)     # GNSS 위치 측정 노이즈 공분산

I6 = np.eye(6)
pos_lc = np.zeros_like(ins_dr)

prev_t = t_imu[0]

for k in range(N):
    t = t_imu[k]
    dt = t - prev_t if k > 0 else 0.0
    prev_t = t

    # ---- 예측 단계 (δp, δv) ----
    if k > 0 and dt > 0.0:
        # 상태 예측
        # δp(k+1) = δp(k) + δv(k)*dt
        # δv(k+1) = δv(k) + w_v
        F = np.block([
            [np.eye(3), dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)],
        ])  # 6x6

        x = F @ x

        # 공분산 예측
        # Qd ≈ diag(0, Q_v*dt)
        Qd = np.block([
            [np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), Q_v * dt],
        ])
        P = F @ P @ F.T + Qd

    # ---- 측정 업데이트 (GNSS 위치) ----
    if gnss_available[k]:
        z = gnss_up[k]                 # GNSS 위치 측정
        p_ins = ins_dr[k]              # 현재 INS DR 위치

        # h(x) = p_ins + δp
        h = p_ins + x[:3]
        y_res = z - h                  # residual

        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)           # δp에만 감도

        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x = x + K @ y_res

        # Joseph form으로 공분산 업데이트
        P = (I6 - K @ H) @ P @ (I6 - K @ H).T + K @ R @ K.T

    # EKF 보정 위치 저장
    pos_lc[k] = ins_dr[k] + x[:3]


err_lc = np.linalg.norm(pos_lc - truth, axis=1)


# ------------------------------------------------------------
# 4) outage 구간에서만 AI drift 보정 적용 (LC + AI)
# ------------------------------------------------------------
pos_lc_ai = pos_lc.copy()

# AI 시각을 IMU index로 맵핑해서, outage 구간에서만 덮어쓰기
for j, t_a in enumerate(t_ai):
    if not (OUTAGE_START <= t_a <= OUTAGE_END):
        continue
    idx = np.argmin(np.abs(t_imu - t_a))
    pos_lc_ai[idx] = pos_ai[j]

err_lc_ai = np.linalg.norm(pos_lc_ai - truth, axis=1)


# ------------------------------------------------------------
# 5) outage 구간 RMSE 비교
# ------------------------------------------------------------
rmse_ins_outage   = np.sqrt(np.mean(err_ins[outage_mask] ** 2))
rmse_lc_outage    = np.sqrt(np.mean(err_lc[outage_mask] ** 2))
rmse_lc_ai_outage = np.sqrt(np.mean(err_lc_ai[outage_mask] ** 2))

print("=== RMSE (GNSS outage 구간에서만) ===")
print(f"INS DR only          : {rmse_ins_outage:.3f} m")
print(f"GNSS/INS LC (EKF)    : {rmse_lc_outage:.3f} m")
print(f"LC + AI drift        : {rmse_lc_ai_outage:.3f} m")


# ------------------------------------------------------------
# 6) 플롯
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(t_imu, err_ins,   label="INS DR only")
plt.plot(t_imu, err_lc,    label="GNSS/INS LC (EKF)", linestyle="--")
plt.plot(t_imu, err_lc_ai, label="GNSS/INS LC + AI (outage only)", linestyle="-.")

plt.fill_between(t_imu, 0, np.nanmax(err_ins),
                 where=outage_mask,
                 alpha=0.1, label="GNSS outage")

plt.xlabel("Time [s]")
plt.ylabel("Position Error Norm [m]")
plt.title("Position Error Comparison (INS vs EKF LC vs EKF LC + AI)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
