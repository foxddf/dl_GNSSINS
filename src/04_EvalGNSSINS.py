import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 설정값
# -------------------------------------------------------------------
IMU_CSV = "../outputs/synthetic/imu_and_ins.csv"
GNSS_CSV = "../outputs/synthetic/gnss.csv"
AI_CSV   = "../outputs/artifacts/drift_corrections.csv"

OUTAGE_START = 40.0  # [s] 01_GenerateSyntheticData.py와 동일
OUTAGE_END   = 60.0  # [s]

G = 9.80665  # [m/s^2]

# 프로세스 노이즈 (튜닝 파라미터)
SIGMA_V_RW   = 0.05     # [m/s]/sqrt(s)   δv process noise
SIGMA_PSI_RW = np.deg2rad(0.05)  # [rad]/sqrt(s) δpsi process noise
SIGMA_BG_RW  = np.deg2rad(0.01)  # [rad/s]/sqrt(s) gyro bias RW
SIGMA_BA_RW  = 0.01     # [m/s^2]/sqrt(s) accel bias RW

# GNSS 측정 노이즈 (SimConfig와 맞춰줌)
SIGMA_GNSS_POS = 1.5    # [m]
SIGMA_GNSS_VEL = 0.2    # [m/s]


# -------------------------------------------------------------------
# 기본 수학/회전 관련 함수
# -------------------------------------------------------------------
def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]_x."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_from_delta(delta: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(delta)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = delta / angle
    half = angle / 2
    return np.array([np.cos(half), *(axis * np.sin(half))])


def integrate_quat(q: np.ndarray, omega_b: np.ndarray, dt: float) -> np.ndarray:
    """
    q_{k+1} = q_k ⊗ dq,  dq from small rotation omega_b*dt.
    """
    delta = omega_b * dt
    dq = quat_from_delta(delta)
    q_new = quat_multiply(q, dq)
    return q_new / np.linalg.norm(q_new)


def apply_small_angle_to_dcm(C_nb: np.ndarray, dpsi: np.ndarray) -> np.ndarray:
    """
    C_nb_corrected ≈ (I - [dpsi]_x) C_nb
    """
    return (np.eye(3) - skew(dpsi)) @ C_nb


# -------------------------------------------------------------------
# 데이터 로드
# -------------------------------------------------------------------
df_imu = pd.read_csv(IMU_CSV)
df_gnss = pd.read_csv(GNSS_CSV)
df_ai = pd.read_csv(AI_CSV)

# IMU/Truth
t_imu = df_imu["time"].values
gyro_meas = df_imu[["gyro_x", "gyro_y", "gyro_z"]].values
accel_meas = df_imu[["accel_x", "accel_y", "accel_z"]].values

truth_pos = df_imu[["pos_truth_e", "pos_truth_n", "pos_truth_u"]].values
truth_vel = df_imu[["truth_vel_e", "truth_vel_n", "truth_vel_u"]].values

# GNSS
t_gnss = df_gnss["time"].values
gnss_pos = df_gnss[["gnss_pos_e", "gnss_pos_n", "gnss_pos_u"]].values
gnss_vel = df_gnss[["gnss_vel_e", "gnss_vel_n", "gnss_vel_u"]].values

# AI (drift correction 결과)
t_ai = df_ai["time"].values
pos_ai = df_ai[["pos_corr_e", "pos_corr_n", "pos_corr_u"]].values


# -------------------------------------------------------------------
# 1) 순수 INS mechanization (bias 포함)
# -------------------------------------------------------------------
N = len(t_imu)
dt_imu = np.diff(t_imu, prepend=t_imu[0])

# 초기 상태: truth를 기준으로 잡음 (실제 시스템에서는 alignment 단계 필요)
pos_ins = np.zeros((N, 3))
vel_ins = np.zeros((N, 3))
quat_ins = np.zeros((N, 4))

pos_ins[0] = truth_pos[0]
vel_ins[0] = truth_vel[0]

# 초기 yaw를 truth_vel에서 추정 (단순) / roll, pitch = 0
vx0, vy0, _ = truth_vel[0]
yaw0 = np.arctan2(vx0, vy0)  # ENU 기준: atan2(E, N)
roll0 = 0.0
pitch0 = 0.0
quat_ins[0] = euler_to_quat(roll0, pitch0, yaw0)

# bias 추정 (초기값 0)
b_g = np.zeros(3)
b_a = np.zeros(3)

for k in range(1, N):
    dt = dt_imu[k]
    if dt <= 0.0:
        dt = 1e-3

    # 바이어스 보정된 IMU
    omega_b = gyro_meas[k - 1] - b_g
    accel_b = accel_meas[k - 1] - b_a

    # attitude update
    q_prev = quat_ins[k - 1]
    q_new = integrate_quat(q_prev, omega_b, dt)
    C_nb = quat_to_dcm(q_new)

    # 중력 (local ENU, up positive, so gravity is negative in U)
    g_n = np.array([0.0, 0.0, -G])

    # specific force를 navigation frame으로
    f_nav = C_nb.T @ accel_b

    # 단순 평탄 지구 모델: acc_nav = f_nav + g
    acc_nav = f_nav + g_n

    # 속도, 위치 적분
    vel_ins[k] = vel_ins[k - 1] + acc_nav * dt
    pos_ins[k] = pos_ins[k - 1] + vel_ins[k] * dt

    quat_ins[k] = q_new

# INS DR 에러
err_ins = np.linalg.norm(pos_ins - truth_pos, axis=1)


# -------------------------------------------------------------------
# 2) GNSS/INS 15-state EKF (LC)
#    상태: [δp(3), δv(3), δψ(3), δbg(3), δba(3)]
# -------------------------------------------------------------------
# 초기 error state
x = np.zeros(15)  # δp, δv, δψ, δbg, δba

# 공분산 초기값
P = np.diag(
    [
        10.0, 10.0, 10.0,   # δp [m]
        1.0, 1.0, 1.0,      # δv [m/s]
        np.deg2rad(5.0),
        np.deg2rad(5.0),
        np.deg2rad(5.0),    # δψ [rad]
        np.deg2rad(0.1),
        np.deg2rad(0.1),
        np.deg2rad(0.1),    # δbg [rad/s]
        0.1, 0.1, 0.1,      # δba [m/s^2]
    ]
)

# 프로세스 노이즈 공분산 (연속시간 q -> 이산 Qd = q * dt)
q_v   = SIGMA_V_RW**2
q_psi = SIGMA_PSI_RW**2
q_bg  = SIGMA_BG_RW**2
q_ba  = SIGMA_BA_RW**2

I15 = np.eye(15)

# LC 결과 저장
pos_lc = np.zeros_like(pos_ins)
vel_lc = np.zeros_like(vel_ins)

pos_lc[0] = pos_ins[0]
vel_lc[0] = vel_ins[0]

# GNSS 인덱싱용 포인터
gnss_idx = 0
gnss_len = len(t_gnss)

for k in range(1, N):
    dt = dt_imu[k]
    if dt <= 0.0:
        dt = 1e-3

    # ---------------------------
    # (1) 예측 단계: error-state
    # ---------------------------
    # 현재 attitude (INS DR 결과 사용)
    C_nb = quat_to_dcm(quat_ins[k])

    # 연속시간 error dynamics (아주 단순화된 모델)
    # δp_dot = δv
    # δv_dot ≈ -C_nb^T δba + w_v
    # δψ_dot ≈ -δbg + w_ψ
    F_c = np.zeros((15, 15))
    # δp_dot = δv
    F_c[0:3, 3:6] = np.eye(3)
    # δv_dot = -C_nb^T * δba
    F_c[3:6, 12:15] = -C_nb.T
    # δψ_dot = -δbg
    F_c[6:9, 9:12] = -np.eye(3)

    # 이산시간 근사: F = I + F_c * dt
    F = I15 + F_c * dt

    # 프로세스 노이즈 공분산 (단순 대각 근사)
    Qd = np.zeros((15, 15))
    Qd[3:6, 3:6]   = q_v * dt * np.eye(3)       # δv 노이즈
    Qd[6:9, 6:9]   = q_psi * dt * np.eye(3)     # δψ 노이즈
    Qd[9:12, 9:12] = q_bg * dt * np.eye(3)      # δbg 노이즈
    Qd[12:15, 12:15] = q_ba * dt * np.eye(3)    # δba 노이즈

    # 상태, 공분산 예측
    x = F @ x
    P = F @ P @ F.T + Qd

    # ---------------------------
    # (2) GNSS 측정 업데이트
    # ---------------------------
    # GNSS outage 반영: 40~60 s 구간에는 GNSS 사용 안 함
    t_k = t_imu[k]
    gnss_valid = False
    z = None

    # GNSS 시각과 IMU 시각 매칭 (nearest neighbor)
    while gnss_idx + 1 < gnss_len and t_gnss[gnss_idx + 1] <= t_k:
        gnss_idx += 1

    if gnss_idx < gnss_len:
        if abs(t_gnss[gnss_idx] - t_k) <= 0.5:  # 약 0.5 s 이내면 매칭
            z_pos = gnss_pos[gnss_idx]
            z_vel = gnss_vel[gnss_idx]
            if (
                not np.any(np.isnan(z_pos))
                and not np.any(np.isnan(z_vel))
                and not (OUTAGE_START <= t_gnss[gnss_idx] <= OUTAGE_END)
            ):
                gnss_valid = True
                z = np.concatenate([z_pos, z_vel])

    if gnss_valid and z is not None:
        # 측정 모델:
        # z = [p_gnss; v_gnss]
        # h = [p_ins + δp; v_ins + δv]
        p_ins_k = pos_ins[k]
        v_ins_k = vel_ins[k]

        h = np.concatenate([p_ins_k + x[0:3], v_ins_k + x[3:6]])
        y = z - h  # residual (6,)

        # H: 6 x 15
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)  # δp
        H[3:6, 3:6] = np.eye(3)  # δv

        R = np.zeros((6, 6))
        R[0:3, 0:3] = (SIGMA_GNSS_POS**2) * np.eye(3)
        R[3:6, 3:6] = (SIGMA_GNSS_VEL**2) * np.eye(3)

        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x = x + K @ y

        # Joseph form
        I = np.eye(15)
        P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

        # ---------------------------
        # (2-1) 폐루프 피드백
        # ---------------------------
        delta_p  = x[0:3]
        delta_v  = x[3:6]
        delta_psi = x[6:9]
        delta_bg  = x[9:12]
        delta_ba  = x[12:15]

        # 위치, 속도 보정
        pos_ins[k] = pos_ins[k] + delta_p
        vel_ins[k] = vel_ins[k] + delta_v

        # attitude 보정
        C_nb = quat_to_dcm(quat_ins[k])
        C_nb_corr = apply_small_angle_to_dcm(C_nb, delta_psi)
        # C_nb_corr에서 쿼터니언 재구성
        # (간단히 eigen분해 대신, 소위 DCM->quat 변환 구현)
        # 여기서는 수식 간단화를 위해 대략적인 변환 사용
        # 참고: numerically stable DCM->quat 구현은 필요시 개선
        tr = np.trace(C_nb_corr)
        if tr > 0.0:
            S = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * S
            qx = (C_nb_corr[2, 1] - C_nb_corr[1, 2]) / S
            qy = (C_nb_corr[0, 2] - C_nb_corr[2, 0]) / S
            qz = (C_nb_corr[1, 0] - C_nb_corr[0, 1]) / S
            quat_ins[k] = np.array([qw, qx, qy, qz]) / np.linalg.norm(
                np.array([qw, qx, qy, qz])
            )
        else:
            # fallback 간단 구현
            quat_ins[k] = quat_ins[k]  # 필요시 개선

        # bias 보정
        b_g = b_g + delta_bg
        b_a = b_a + delta_ba

        # error-state를 리셋 (δp, δv, δψ, δbg, δba → 0)
        x[:] = 0.0
        # 공분산은 그대로 두거나, 일부 블록만 줄이는 방식도 있음 (여기서는 그대로 둠)

    # LC 결과 저장 (보정된 INS 상태 기준)
    pos_lc[k] = pos_ins[k]
    vel_lc[k] = vel_ins[k]


# LC 에러
err_lc = np.linalg.norm(pos_lc - truth_pos, axis=1)


# -------------------------------------------------------------------
# 3) GNSS outage 구간에서만 AI 보정 적용 (LC + AI)
# -------------------------------------------------------------------
pos_lc_ai = pos_lc.copy()

# outage 구간 마스크 (IMU 시간 기준)
outage_mask = (t_imu >= OUTAGE_START) & (t_imu <= OUTAGE_END)

for j, t_a in enumerate(t_ai):
    if not (OUTAGE_START <= t_a <= OUTAGE_END):
        continue
    idx = np.argmin(np.abs(t_imu - t_a))
    pos_lc_ai[idx] = pos_ai[j]

err_lc_ai = np.linalg.norm(pos_lc_ai - truth_pos, axis=1)


# -------------------------------------------------------------------
# 4) outage 구간 RMSE 비교
# -------------------------------------------------------------------
rmse_ins_outage   = np.sqrt(np.mean(err_ins[outage_mask] ** 2))
rmse_lc_outage    = np.sqrt(np.mean(err_lc[outage_mask] ** 2))
rmse_lc_ai_outage = np.sqrt(np.mean(err_lc_ai[outage_mask] ** 2))

print("=== RMSE (GNSS outage 구간에서만) ===")
print(f"INS DR only          : {rmse_ins_outage:.3f} m")
print(f"GNSS/INS LC (15-state EKF) : {rmse_lc_outage:.3f} m")
print(f"LC + AI drift        : {rmse_lc_ai_outage:.3f} m")


# -------------------------------------------------------------------
# 5) 플롯
# -------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(t_imu, err_ins,   label="INS DR only")
plt.plot(t_imu, err_lc,    label="GNSS/INS LC (15-state EKF)", linestyle="--")
plt.plot(t_imu, err_lc_ai, label="LC + AI (outage only)", linestyle="-.")

plt.fill_between(t_imu, 0, np.nanmax(err_ins),
                 where=outage_mask,
                 alpha=0.1, label="GNSS outage")

plt.xlabel("Time [s]")
plt.ylabel("Position Error Norm [m]")
plt.title("INS vs 15-state EKF LC vs LC+AI (GNSS outage)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
