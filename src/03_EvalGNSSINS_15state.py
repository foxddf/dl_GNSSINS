import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")   # 또는 "Qt5Agg" 등
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Sequence, Dict, Tuple, List


# -------------------------------------------------------------------
# 설정값
# -------------------------------------------------------------------
scenario_num = "scenario_011"
IMU_CSV   = "../outputs/synthetic_multi/" + scenario_num + "/imu_and_ins.csv"
GNSS_CSV  = "../outputs/synthetic_multi/" + scenario_num + "/gnss.csv"

# GRU가 학습에 사용한 bias_training.csv & 체크포인트
BIAS_CSV        = "../outputs/synthetic_multi/" + scenario_num + "/bias_training.csv"
CHECKPOINT_PATH = "../outputs/artifacts/drift_gru.pt"

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
# GRU용 Dataset / Model 정의 (02_TrainINSDrift.py와 동일 구조)
# -------------------------------------------------------------------
class DriftSequenceDataset(Dataset):
    """Create sliding-window IMU+INS+GNSS sequences and bias labels."""

    def __init__(
        self,
        csv_path: Path | str,
        window_size: int,
        stride: int = 1,
        feature_columns: Optional[Sequence[str]] = None,
        include_velocity: bool = True,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.window_size = int(window_size)
        self.stride = int(stride)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        df = pd.read_csv(self.csv_path)
        if "time" not in df.columns:
            raise ValueError("CSV must contain a 'time' column")

        # ------------------------------------------------------------------
        # 1) 기본 feature 구성
        #    - IMU: gyro, accel
        #    - INS: position, velocity, quaternion
        #    - GNSS: pos/vel
        #    - residual: pos_res, vel_res
        #    - 기타: gnss_valid, time_since_gnss, gyro_norm, accel_norm
        # ------------------------------------------------------------------
        default_features: List[str] = [
            # IMU
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "accel_x",
            "accel_y",
            "accel_z",
            # INS position
            "ins_pos_e",
            "ins_pos_n",
            "ins_pos_u",
        ]

        if include_velocity:
            default_features.extend(
                [
                    "ins_vel_e",
                    "ins_vel_n",
                    "ins_vel_u",
                ]
            )

        # 추가 feature 묶음들 (Train 코드와 동일)
        extra_feature_groups = [
            # INS attitude (quaternion)
            ["ins_q_w", "ins_q_x", "ins_q_y", "ins_q_z"],
            # GNSS interpolated measurements
            ["gnss_pos_e", "gnss_pos_n", "gnss_pos_u"],
            ["gnss_vel_e", "gnss_vel_n", "gnss_vel_u"],
            # INS - GNSS residuals
            ["pos_res_e", "pos_res_n", "pos_res_u"],
            ["vel_res_e", "vel_res_n", "vel_res_u"],
            # IMU norms
            ["gyro_norm", "accel_norm"],
            # GNSS 상태
            ["gnss_valid", "time_since_gnss"],
        ]

        # 실제 CSV에 존재하는 컬럼만 추가
        for group in extra_feature_groups:
            for col in group:
                if col in df.columns and col not in default_features:
                    default_features.append(col)

        # 사용자가 feature_columns를 직접 주면 그것을 우선 사용
        self.feature_columns = list(feature_columns) if feature_columns else default_features
        for col in self.feature_columns:
            if col not in df.columns:
                raise ValueError(f"Missing feature column '{col}' in {self.csv_path}")

        # ------------------------------------------------------------------
        # 2) 타깃 라벨 (gyro/accel bias 6개)
        # ------------------------------------------------------------------
        bias_cols = [
            "gyro_bias_x",
            "gyro_bias_y",
            "gyro_bias_z",
            "accel_bias_x",
            "accel_bias_y",
            "accel_bias_z",
        ]
        for col in bias_cols:
            if col not in df.columns:
                raise ValueError(f"Missing bias column '{col}' in {self.csv_path}")

        self.times = df["time"].to_numpy(dtype=np.float64)
        self.labels = df[bias_cols].to_numpy(dtype=np.float32)

        # ------------------------------------------------------------------
        # 3) 피처 행렬 구성 및 정규화 (Train과 동일 로직)
        # ------------------------------------------------------------------
        feature_matrix = df[self.feature_columns].to_numpy(dtype=np.float32)

        # mean/std 계산 (NaN 무시)
        if feature_mean is None or feature_std is None:
            self.feature_mean = np.nanmean(feature_matrix, axis=0).astype(np.float32)
            self.feature_std = np.nanstd(feature_matrix, axis=0).astype(np.float32) + 1e-6
            # 전체 NaN 컬럼 방어
            self.feature_mean = np.nan_to_num(self.feature_mean, nan=0.0)
            self.feature_std = np.nan_to_num(self.feature_std, nan=1.0)
        else:
            self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
            self.feature_std = np.asarray(feature_std, dtype=np.float32)

        if self.feature_mean.shape[0] != feature_matrix.shape[1]:
            raise ValueError("Feature mean/std dimension mismatch")

        # NaN을 column mean으로 대체 (→ outage 구간은 평균값, 정규화 후 0 근처)
        nan_mask = np.isnan(feature_matrix)
        if np.any(nan_mask):
            feature_matrix = np.where(
                nan_mask,
                self.feature_mean[None, :],
                feature_matrix,
            )

        # 정규화
        normalized_features = (feature_matrix - self.feature_mean) / self.feature_std
        self.features = normalized_features.astype(np.float32)

        # ------------------------------------------------------------------
        # 4) 슬라이딩 윈도우 인덱스
        # ------------------------------------------------------------------
        total_steps = len(df)
        if total_steps < self.window_size:
            raise ValueError("Not enough samples for the requested window size")
        self.indices = list(range(0, total_steps - self.window_size + 1, self.stride))
        if not self.indices:
            raise ValueError("No windows could be created with the given stride")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        start = self.indices[idx]
        end = start + self.window_size
        x = torch.from_numpy(self.features[start:end])   # (T, D)
        y = torch.from_numpy(self.labels[end - 1])       # (6,)

        meta = {
            "time": torch.tensor(self.times[end - 1], dtype=torch.float32),
            "gyro_bias_true": torch.from_numpy(self.labels[end - 1, 0:3]),
            "accel_bias_true": torch.from_numpy(self.labels[end - 1, 3:6]),
        }
        return x, y, meta

    @property
    def input_dim(self) -> int:
        return self.features.shape[1]

    @property
    def label_dim(self) -> int:
        return self.labels.shape[1]



class DriftGRUModel(nn.Module):
    """GRU encoder followed by MLP regression head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 6,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]  # (B, H)
        return self.head(last_hidden)


# -------------------------------------------------------------------
# 기본 수학/회전 관련 함수 (기존 코드 그대로)
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
# 2) GNSS/INS 15-state EKF (약결합)
#    - use_ai_bias=False : EKF only (baseline)
#    - use_ai_bias=True  : EKF + GRU bias (outage 구간에서만 보정)
# -------------------------------------------------------------------
def run_ekf(
    use_ai_bias: bool,
    outage_mask_imu: np.ndarray,
    outage_recovery_time: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    return:
        pos_lc, vel_lc, quat_lc  (모두 shape: (N, 3 또는 4))
    """
    pos_lc = np.zeros((N, 3))
    vel_lc = np.zeros((N, 3))
    quat_lc = np.zeros((N, 4))

    # 초기 nominal 상태: truth 기준
    pos_lc[0] = truth_pos[0]
    vel_lc[0] = truth_vel[0]
    quat_lc[0] = euler_to_quat(0.0, 0.0, np.arctan2(truth_vel[0, 0], truth_vel[0, 1]))

    # EKF 내부 bias 추정값
    b_g = np.zeros(3)
    b_a = np.zeros(3)

    # error state
    x = np.zeros(15)

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

    q_v   = SIGMA_V_RW**2
    q_psi = SIGMA_PSI_RW**2
    q_bg  = SIGMA_BG_RW**2
    q_ba  = SIGMA_BA_RW**2

    I15 = np.eye(15)

    gnss_idx = 0
    gnss_len = len(t_gnss)

    first_gnss_after_outage_done = False

    for k in range(1, N):
        dt = dt_imu[k]
        if dt <= 0.0:
            dt = 1e-3

        t_k = t_imu[k]
        in_outage = bool(outage_mask_imu[k])


        # ---------- (1) nominal INS propagate ----------
        q_prev = quat_lc[k - 1]
        p_prev = pos_lc[k - 1]
        v_prev = vel_lc[k - 1]

        # use_ai_bias=True 이고 outage 구간이면 GRU bias 사용
        if use_ai_bias and in_outage:
            omega_b = gyro_meas[k - 1] - (b_g + bg_ai[k - 1])
            accel_b = accel_meas[k - 1] - (b_a + ba_ai[k - 1])
        else:
            omega_b = gyro_meas[k - 1] - b_g
            accel_b = accel_meas[k - 1] - b_a

        q_pred = integrate_quat(q_prev, omega_b, dt)
        C_nb = quat_to_dcm(q_pred)

        f_nav = C_nb.T @ accel_b
        acc_nav = f_nav + g_n

        v_pred = v_prev + acc_nav * dt
        p_pred = p_prev + v_pred * dt

        # ---------- (2) error-state 예측 ----------
        F_c = np.zeros((15, 15))
        F_c[0:3, 3:6] = np.eye(3)
        F_c[3:6, 12:15] = -C_nb.T
        F_c[6:9, 9:12]  = -np.eye(3)

        F = I15 + F_c * dt

        Qd = np.zeros((15, 15))
        Qd[3:6,   3:6]   = q_v   * dt * np.eye(3)
        Qd[6:9,   6:9]   = q_psi * dt * np.eye(3)
        Qd[9:12,  9:12]  = q_bg  * dt * np.eye(3)
        Qd[12:15, 12:15] = q_ba  * dt * np.eye(3)

        x = F @ x
        P = F @ P @ F.T + Qd

        # ---------- (3) GNSS 매칭 ----------
        gnss_valid = False
        z = None
        z_pos = None
        z_vel = None

        while gnss_idx + 1 < gnss_len and t_gnss[gnss_idx + 1] <= t_k:
            gnss_idx += 1

        if gnss_idx < gnss_len:
            if abs(t_gnss[gnss_idx] - t_k) <= 0.5:
                z_pos = gnss_pos[gnss_idx]
                z_vel = gnss_vel[gnss_idx]
                if (
                    not np.any(np.isnan(z_pos))
                    and not np.any(np.isnan(z_vel))
                ):
                    gnss_valid = True
                    z = np.concatenate([z_pos, z_vel])

        if gnss_valid and (not first_gnss_after_outage_done) and (outage_recovery_time is not None):
            t_g = t_gnss[gnss_idx]
            if t_g >= outage_recovery_time:
                first_gnss_after_outage_done = True

                pos_lc[k] = z_pos.copy()
                vel_lc[k] = z_vel.copy()

                vE, vN, _ = vel_lc[k]
                if np.hypot(vE, vN) > 1e-3:
                    yaw_align = np.arctan2(vE, vN)
                else:
                    roll_tmp, pitch_tmp, yaw_tmp = dcm_to_euler(C_nb)
                    yaw_align = yaw_tmp
                roll_tmp, pitch_tmp, _ = dcm_to_euler(C_nb)
                quat_lc[k] = euler_to_quat(roll_tmp, pitch_tmp, yaw_align)

                x[:] = 0.0
                P = np.diag(
                    [
                        10.0, 10.0, 10.0,
                        1.0, 1.0, 1.0,
                        np.deg2rad(5.0),
                        np.deg2rad(5.0),
                        np.deg2rad(5.0),
                        np.deg2rad(0.1),
                        np.deg2rad(0.1),
                        np.deg2rad(0.1),
                        0.1, 0.1, 0.1,
                    ]
                )
                continue


        # 일반 GNSS update
        if gnss_valid and z is not None:
            h = np.concatenate([p_pred + x[0:3], v_pred + x[3:6]])
            y = z - h

            H = np.zeros((6, 15))
            H[0:3, 0:3] = np.eye(3)
            H[3:6, 3:6] = np.eye(3)

            R = np.zeros((6, 6))
            R[0:3, 0:3] = (SIGMA_GNSS_POS**2) * np.eye(3)
            R[3:6, 3:6] = (SIGMA_GNSS_VEL**2) * np.eye(3)

            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)

            x = x + K @ y

            I = np.eye(15)
            P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

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

            x[:] = 0.0

            pos_lc[k] = p_corr
            vel_lc[k] = v_corr
            quat_lc[k] = q_corr
        else:
            # GNSS 없음: DR + 예측값만 사용
            pos_lc[k] = p_pred
            vel_lc[k] = v_pred
            quat_lc[k] = q_pred

    return pos_lc, vel_lc, quat_lc


# -------------------------------------------------------------------
# 데이터 로드 (IMU/Truth/GNSS)
# -------------------------------------------------------------------
df_imu = pd.read_csv(IMU_CSV)
df_gnss = pd.read_csv(GNSS_CSV)

# IMU/Truth
t_imu = df_imu["time"].values
gyro_meas = df_imu[["gyro_x", "gyro_y", "gyro_z"]].values
accel_meas = df_imu[["accel_x", "accel_y", "accel_z"]].values

truth_pos = df_imu[["pos_truth_e", "pos_truth_n", "pos_truth_u"]].values
truth_vel = df_imu[["vel_truth_e", "vel_truth_n", "vel_truth_u"]].values

gyro_bias_true = df_imu[["gyro_bias_x", "gyro_bias_y", "gyro_bias_z"]].values
accel_bias_true = df_imu[["accel_bias_x", "accel_bias_y", "accel_bias_z"]].values

# GNSS
t_gnss = df_gnss["time"].values
gnss_pos = df_gnss[["gnss_pos_e", "gnss_pos_n", "gnss_pos_u"]].values
gnss_vel = df_gnss[["gnss_vel_e", "gnss_vel_n", "gnss_vel_u"]].values

N = len(t_imu)
dt_imu = np.diff(t_imu, prepend=t_imu[0])
g_n = np.array([0.0, 0.0, -G])  # ENU up 기준


# -------------------------------------------------------------------
# 1-a) bias_training.csv 기반 GNSS outage 자동 검출
# -------------------------------------------------------------------
df_bias = pd.read_csv(BIAS_CSV)
t_bias = df_bias["time"].to_numpy()
gnss_valid_bias = df_bias["gnss_valid"].to_numpy().astype(float)

# IMU 시간축으로 보간 → 0이면 outage, 1이면 정상
gnss_valid_imu = np.interp(t_imu, t_bias, gnss_valid_bias)
outage_mask_imu = gnss_valid_imu < 0.5

# outage → 정상으로 넘어가는 첫 시점(복구 시간) 찾기
recovery_edges = np.where(outage_mask_imu[:-1] & (~outage_mask_imu[1:]))[0]
outage_recovery_time = None
if len(recovery_edges) > 0:
    outage_recovery_index = recovery_edges[0] + 1
    outage_recovery_time = t_imu[outage_recovery_index]

print(f"[outage detection] recovery_time = {outage_recovery_time}")


# -------------------------------------------------------------------
# 0) GRU 체크포인트 로드 & bias 예측 → IMU 시간축으로 정렬
# -------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

model_args = ckpt["model_args"]
model = DriftGRUModel(**model_args).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

dataset = DriftSequenceDataset(
    csv_path=BIAS_CSV,
    window_size=200,        # 학습 때 사용한 window_size와 동일하게 맞추는 게 좋음
    stride=1,
    include_velocity=True,
    feature_mean=ckpt["feature_mean"],
    feature_std=ckpt["feature_std"],
)
loader = DataLoader(dataset, batch_size=256, shuffle=False)

# IMU 시간축 기준 AI bias 추정값
bg_ai = np.zeros((N, 3))
ba_ai = np.zeros((N, 3))
count_ai = np.zeros(N, dtype=int)

with torch.no_grad():
    for inputs, _, meta in loader:
        inputs = inputs.to(device)
        preds = model(inputs).cpu().numpy()  # (B, 6)
        times = meta["time"].cpu().numpy()   # (B,)

        for i in range(preds.shape[0]):
            t = times[i]
            # IMU 시간과 매칭 (가장 가까운 인덱스)
            idx = int(np.argmin(np.abs(t_imu - t)))
            bg_ai[idx] += preds[i, 0:3]
            ba_ai[idx] += preds[i, 3:6]
            count_ai[idx] += 1

# 여러 window가 같은 time을 덮을 수 있으므로 평균
valid_idx = count_ai > 0
bg_ai[valid_idx] /= count_ai[valid_idx, None]
ba_ai[valid_idx] /= count_ai[valid_idx, None]

# bias RMSE (전체 구간, 참고용)
rmse_bg = np.sqrt(np.mean((bg_ai - gyro_bias_true) ** 2))
rmse_ba = np.sqrt(np.mean((ba_ai - accel_bias_true) ** 2))
print(f"[GRU bias prediction] gyro bias RMSE: {rmse_bg:.6e} [rad/s]")
print(f"[GRU bias prediction] accel bias RMSE: {rmse_ba:.6e} [m/s^2]")


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

# EKF baseline (GRU 미사용)
pos_lc_base, vel_lc_base, quat_lc_base = run_ekf(
    use_ai_bias=False,
    outage_mask_imu=outage_mask_imu,
    outage_recovery_time=outage_recovery_time,
)
err_lc_base = np.linalg.norm(pos_lc_base - truth_pos, axis=1)

# EKF + GRU bias
pos_lc_ai, vel_lc_ai, quat_lc_ai = run_ekf(
    use_ai_bias=True,
    outage_mask_imu=outage_mask_imu,
    outage_recovery_time=outage_recovery_time,
)
err_lc_ai = np.linalg.norm(pos_lc_ai - truth_pos, axis=1)

# outage 구간 마스크 (자동 검출 결과)
outage_mask = outage_mask_imu


rmse_ins_outage    = np.sqrt(np.mean(err_ins[outage_mask] ** 2))
rmse_lc_base_out   = np.sqrt(np.mean(err_lc_base[outage_mask] ** 2))
rmse_lc_ai_out     = np.sqrt(np.mean(err_lc_ai[outage_mask] ** 2))

print("=== RMSE (GNSS outage 구간에서만) ===")
print(f"INS DR only                  : {rmse_ins_outage:.3f} m")
print(f"GNSS/INS LC (EKF only)       : {rmse_lc_base_out:.3f} m")
print(f"GNSS/INS LC + GRU bias (AI)  : {rmse_lc_ai_out:.3f} m")


plt.figure(figsize=(12, 6))
plt.plot(t_imu, err_lc_base, label="GNSS/INS LC (EKF only)", linestyle="--")
plt.plot(t_imu, err_lc_ai,   label="GNSS/INS LC + GRU bias")

plt.fill_between(
    t_imu,
    0,
    np.nanmax(np.concatenate([err_lc_base, err_lc_ai])),
    where=outage_mask,
    alpha=0.1,
    label="GNSS outage",
)

# -------------------------------------------------------------------
# 플롯
# -------------------------------------------------------------------
plt.xlabel("Time [s]")
plt.ylabel("Position Error Norm [m]")
plt.title("GNSS/INS LC: EKF only vs EKF + GRU bias (GNSS outage)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
