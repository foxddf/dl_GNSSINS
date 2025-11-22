import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")   # 또는 "Qt5Agg" 등
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# 설정값
# -------------------------------------------------------------------
IMU_CSV = "../outputs/synthetic/imu_and_ins.csv"
GNSS_CSV = "../outputs/synthetic/gnss.csv"
AI_CSV   = "../outputs/artifacts/drift_corrections.csv"

OUTAGE_START = 40.0  # [s]
OUTAGE_END   = 60.0  # [s]

G = 9.80665  # [m/s^2]

# 프로세스 노이즈 (튜닝 파라미터)
SIGMA_V_RW   = 0.05                      # [m/s]/sqrt(s)
SIGMA_PSI_RW = np.deg2rad(0.05)          # [rad]/sqrt(s)
SIGMA_BG_RW  = np.deg2rad(0.01)          # [rad/s]/sqrt(s)
SIGMA_BA_RW  = 0.01                      # [m/s^2]/sqrt(s)

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

def dcm_to_euler(C: np.ndarray):
    """
    ZYX (roll-pitch-yaw) 기준 Euler angle.
    nav = C * body  형태 DCM이라고 가정.
    """
    # pitch = -asin(C[2,0]) 부분에서 수치오차 방지
    pitch = -np.arcsin(np.clip(C[2, 0], -1.0, 1.0))
    roll  = np.arctan2(C[2, 1], C[2, 2])
    yaw   = np.arctan2(C[1, 0], C[0, 0])
    return roll, pitch, yaw

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


def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    """DCM -> quat 변환 (trace 기반, 안정한 버전)."""
    tr = np.trace(C)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (C[2, 1] - C[1, 2]) / S
        qy = (C[0, 2] - C[2, 0]) / S
        qz = (C[1, 0] - C[0, 1]) / S
    else:
        # 주대각 원소 중 가장 큰 것 기준으로 계산
        if (C[0, 0] > C[1, 1]) and (C[0, 0] > C[2, 2]):
            S = np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2]) * 2.0
            qw = (C[2, 1] - C[1, 2]) / S
            qx = 0.25 * S
            qy = (C[0, 1] + C[1, 0]) / S
            qz = (C[0, 2] + C[2, 0]) / S
        elif C[1, 1] > C[2, 2]:
            S = np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2]) * 2.0
            qw = (C[0, 2] - C[2, 0]) / S
            qx = (C[0, 1] + C[1, 0]) / S
            qy = 0.25 * S
            qz = (C[1, 2] + C[2, 1]) / S
        else:
            S = np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1]) * 2.0
            qw = (C[1, 0] - C[0, 1]) / S
            qx = (C[0, 2] + C[2, 0]) / S
            qy = (C[1, 2] + C[2, 1]) / S
            qz = 0.25 * S
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)


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
truth_vel = df_imu[["vel_truth_e", "vel_truth_n", "vel_truth_u"]].values

# GNSS
t_gnss = df_gnss["time"].values
gnss_pos = df_gnss[["gnss_pos_e", "gnss_pos_n", "gnss_pos_u"]].values
gnss_vel = df_gnss[["gnss_vel_e", "gnss_vel_n", "gnss_vel_u"]].values

# AI (drift correction 결과)
t_ai = df_ai["time"].values
pos_ai = df_ai[["pos_corr_e", "pos_corr_n", "pos_corr_u"]].values

N = len(t_imu)
dt_imu = np.diff(t_imu, prepend=t_imu[0])
g_n = np.array([0.0, 0.0, -G])  # ENU up 기준


# -------------------------------------------------------------------
# 1) 순수 INS mechanization (DR only, EKF 없이) - 비교용
# -------------------------------------------------------------------
pos_dr = np.zeros((N, 3))
vel_dr = np.zeros((N, 3))
quat_dr = np.zeros((N, 4))

# 초기 상태: truth 기준
pos_dr[0] = truth_pos[0]
vel_dr[0] = truth_vel[0]

vx0, vy0, _ = truth_vel[0]
yaw0 = np.arctan2(vx0, vy0)  # ENU 기준: atan2(E, N)
roll0 = 0.0
pitch0 = 0.0
quat_dr[0] = euler_to_quat(roll0, pitch0, yaw0)

for k in range(1, N):
    dt = dt_imu[k]
    if dt <= 0.0:
        dt = 1e-3

    q_prev = quat_dr[k - 1]
    omega_b = gyro_meas[k - 1]  # bias 보정 없음
    accel_b = accel_meas[k - 1]

    # attitude update
    q_pred = integrate_quat(q_prev, omega_b, dt)
    C_nb = quat_to_dcm(q_pred)

    # acceleration in nav frame
    f_nav = C_nb.T @ accel_b
    acc_nav = f_nav + g_n

    vel_dr[k] = vel_dr[k - 1] + acc_nav * dt
    pos_dr[k] = pos_dr[k - 1] + vel_dr[k] * dt
    quat_dr[k] = q_pred

err_ins = np.linalg.norm(pos_dr - truth_pos, axis=1)


# -------------------------------------------------------------------
# 2) GNSS/INS 15-state EKF (약결합, 통합된 mechanization + EKF)
#    상태: x = [δp(3), δv(3), δψ(3), δbg(3), δba(3)]
# -------------------------------------------------------------------
# nominal 상태 저장 (EKF 적용된 INS)
pos_lc = np.zeros((N, 3))
vel_lc = np.zeros((N, 3))
quat_lc = np.zeros((N, 4))

# 초기 nominal 상태: truth 기준
pos_lc[0] = truth_pos[0]
vel_lc[0] = truth_vel[0]
quat_lc[0] = euler_to_quat(roll0, pitch0, yaw0)

# gyro/accel bias 초기값
b_g = np.zeros(3)
b_a = np.zeros(3)

# error state 초기값
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

# 프로세스 노이즈 공분산 계수
q_v   = SIGMA_V_RW**2
q_psi = SIGMA_PSI_RW**2
q_bg  = SIGMA_BG_RW**2
q_ba  = SIGMA_BA_RW**2

I15 = np.eye(15)

# GNSS 인덱싱용
gnss_idx = 0
gnss_len = len(t_gnss)

first_gnss_after_outage_done = False
prev_gnss_valid = False

for k in range(1, N):
    dt = dt_imu[k]
    if dt <= 0.0:
        dt = 1e-3

    t_k = t_imu[k]
    in_outage = (OUTAGE_START <= t_k <= OUTAGE_END)

    # ---------------------------
    # (1) bias 보정된 IMU로 nominal INS propagate
    # ---------------------------
    q_prev = quat_lc[k - 1]
    p_prev = pos_lc[k - 1]
    v_prev = vel_lc[k - 1]

    omega_b = gyro_meas[k - 1] - b_g
    accel_b = accel_meas[k - 1] - b_a

    # attitude update
    q_pred = integrate_quat(q_prev, omega_b, dt)
    C_nb = quat_to_dcm(q_pred)

    # nav frame acceleration
    f_nav = C_nb.T @ accel_b
    acc_nav = f_nav + g_n

    v_pred = v_prev + acc_nav * dt
    p_pred = p_prev + v_pred * dt

    # ---------------------------
    # (2) error-state 예측
    # ---------------------------
    F_c = np.zeros((15, 15))
    # δp_dot = δv
    F_c[0:3, 3:6] = np.eye(3)
    # δv_dot ≈ -C_nb^T δba
    F_c[3:6, 12:15] = -C_nb.T
    # δψ_dot ≈ -δbg
    F_c[6:9, 9:12] = -np.eye(3)

    F = I15 + F_c * dt

    Qd = np.zeros((15, 15))
    Qd[3:6,   3:6]   = q_v   * dt * np.eye(3)
    Qd[6:9,   6:9]   = q_psi * dt * np.eye(3)
    Qd[9:12,  9:12]  = q_bg  * dt * np.eye(3)
    Qd[12:15, 12:15] = q_ba  * dt * np.eye(3)

    x = F @ x
    P = F @ P @ F.T + Qd

    # ---------------------------
    # (3) GNSS 측정 매칭
    # ---------------------------
    gnss_valid = False
    z = None
    z_pos = None
    z_vel = None

    # GNSS 시각과 IMU 시각 매칭 (nearest neighbor)
    while gnss_idx + 1 < gnss_len and t_gnss[gnss_idx + 1] <= t_k:
        gnss_idx += 1

    if gnss_idx < gnss_len:
        if abs(t_gnss[gnss_idx] - t_k) <= 0.5:
            z_pos = gnss_pos[gnss_idx]
            z_vel = gnss_vel[gnss_idx]
            if (
                not np.any(np.isnan(z_pos))
                and not np.any(np.isnan(z_vel))
                and not (OUTAGE_START <= t_gnss[gnss_idx] <= OUTAGE_END)
            ):
                gnss_valid = True
                z = np.concatenate([z_pos, z_vel])

    # ---------------------------
    # (3-1) outage 이후 "첫 GNSS"인지 체크 → hard reset
    #   조건: 
    #   - gnss_valid 이고
    #   - 이 GNSS 시각은 outage 범위 밖이고
    #   - 바로 직전 GNSS 시각은 outage 범위 안(또는 outage 종료 직전)
    # ---------------------------
    if gnss_valid and (not first_gnss_after_outage_done):
        t_g = t_gnss[gnss_idx]

        # 이 GNSS 샘플은 outage 바깥
        if t_g > OUTAGE_END:
            # 직전 GNSS가 있으면, 그건 outage 구간(또는 그 직전)이라고 가정
            # (strict 하게 보려면 t_gnss[gnss_idx-1] >= OUTAGE_START 조건까지 넣을 수도 있음)
            first_gnss_after_outage_done = True

            # 1) 위치/속도는 GNSS로 강제 동기화
            pos_lc[k] = z_pos.copy()
            vel_lc[k] = z_vel.copy()

            # 2) yaw는 GNSS 속도 방향으로 재정렬 (roll/pitch는 기존 유지)
            vE, vN, _ = vel_lc[k]
            if np.hypot(vE, vN) > 1e-3:
                yaw_align = np.arctan2(vE, vN)
            else:
                # 속도가 너무 작으면 기존 yaw 유지
                roll_tmp, pitch_tmp, yaw_tmp = dcm_to_euler(C_nb)
                yaw_align = yaw_tmp

            roll_tmp, pitch_tmp, _ = dcm_to_euler(C_nb)
            quat_lc[k] = euler_to_quat(roll_tmp, pitch_tmp, yaw_align)

            # 3) error state, 공분산 리셋
            x[:] = 0.0
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

            # 이 샘플은 EKF update로 쓰지 않고 바로 다음 step으로
            continue

    # ---------------------------
    # (3-2) 일반 GNSS update (outage 구간 밖, reset 이후)
    # ---------------------------
    if gnss_valid and z is not None:
        # 측정 모델:
        # z = [p_gnss; v_gnss]
        # h = [p_pred + δp; v_pred + δv]
        h = np.concatenate([p_pred + x[0:3], v_pred + x[3:6]])
        y = z - h  # residual (6,)

        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)  # δp
        H[3:6, 3:6] = np.eye(3)  # δv

        R = np.zeros((6, 6))
        R[0:3, 0:3] = (SIGMA_GNSS_POS**2) * np.eye(3)
        R[3:6, 3:6] = (SIGMA_GNSS_VEL**2) * np.eye(3)

        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x = x + K @ y

        I = np.eye(15)
        P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

        # ---- 폐루프 피드백 (closed-loop) ----
        delta_p   = x[0:3]
        delta_v   = x[3:6]
        delta_psi = x[6:9]
        delta_bg  = x[9:12]
        delta_ba  = x[12:15]

        p_corr = p_pred + delta_p
        v_corr = v_pred + delta_v
        C_nb_corr = apply_small_angle_to_dcm(C_nb, delta_psi)
        q_corr = dcm_to_quat(C_nb_corr)

        b_g = b_g + delta_bg
        b_a = b_a + delta_ba

        # error-state 리셋
        x[:] = 0.0

        pos_lc[k] = p_corr
        vel_lc[k] = v_corr
        quat_lc[k] = q_corr
    else:
        # GNSS 없음 (outage 포함): DR + 예측값만 사용
        pos_lc[k] = p_pred
        vel_lc[k] = v_pred
        quat_lc[k] = q_pred


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
print(f"INS DR only                : {rmse_ins_outage:.3f} m")
print(f"GNSS/INS LC (15-state EKF) : {rmse_lc_outage:.3f} m")
print(f"LC + AI drift              : {rmse_lc_ai_outage:.3f} m")


# -------------------------------------------------------------------
# 5) 플롯
# -------------------------------------------------------------------
plt.figure(figsize=(12, 6))
# plt.plot(t_imu, err_ins,   label="INS DR only")
plt.plot(t_imu, err_lc,    label="GNSS/INS LC (15-state EKF)", linestyle="--")
# plt.plot(t_imu, err_lc_ai, label="LC + AI (outage only)", linestyle="-.")

# plt.fill_between(
#     t_imu,
#     0,
#     np.nanmax(err_ins),
#     where=outage_mask,
#     alpha=0.1,
#     label="GNSS outage",
# )

plt.fill_between(
    t_imu,
    0,
    np.nanmax(err_lc),
    where=outage_mask,
    alpha=0.1,
    label="GNSS outage",
)

plt.xlabel("Time [s]")
plt.ylabel("Position Error Norm [m]")
plt.title("INS vs 15-state EKF LC vs LC+AI (GNSS outage)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
