"""GNSS/INS synthetic data simulator."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# WGS-84 constants
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
OMEGA_EARTH = 7.292115e-5  # rad/s


@dataclass
class TrajectoryConfig:
    total_time: float = 600.0
    imu_dt: float = 0.01
    speed_mps: float = 25.0
    radius_m: float = 150.0
    altitude_m: float = 50.0
    pattern: str = "circle"


@dataclass
class IMUErrorModel:
    gyro_bias_rw: float = 1e-5
    gyro_noise_std: float = 5e-4
    accel_bias_rw: float = 5e-5
    accel_noise_std: float = 1e-3
    gyro_bias_init_std: float = 5e-4
    accel_bias_init_std: float = 5e-4


@dataclass
class GNSSNoiseModel:
    dt: float = 1.0
    pos_noise_std: float = 1.0
    vel_noise_std: float = 0.1
    outage_intervals: Tuple[Tuple[float, float], ...] = ()


@dataclass
class SimConfig:
    origin_lat_deg: float = 37.0
    origin_lon_deg: float = -122.0
    origin_height_m: float = 10.0
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    imu_errors: IMUErrorModel = field(default_factory=IMUErrorModel)
    gnss: GNSSNoiseModel = field(default_factory=GNSSNoiseModel)
    gravity: float = 9.80665
    seed: int = 42
    output_dir: Path = Path("../outputs/synthetic")


def generate_time_axes(traj_cfg: TrajectoryConfig, gnss_dt: float) -> Tuple[np.ndarray, np.ndarray]:
    imu_times = np.arange(0.0, traj_cfg.total_time + traj_cfg.imu_dt, traj_cfg.imu_dt)
    gnss_times = np.arange(0.0, traj_cfg.total_time + gnss_dt, gnss_dt)
    return imu_times, gnss_times


def ecef_to_enu_matrix(lat_rad: float, lon_rad: float) -> np.ndarray:
    s_lat = math.sin(lat_rad)
    c_lat = math.cos(lat_rad)
    s_lon = math.sin(lon_rad)
    c_lon = math.cos(lon_rad)
    return np.array(
        [
            [-s_lon, c_lon, 0.0],
            [-s_lat * c_lon, -s_lat * s_lon, c_lat],
            [c_lat * c_lon, c_lat * s_lon, s_lat],
        ]
    )


def latlon_to_ecef(lat_rad: float, lon_rad: float, h: float) -> np.ndarray:
    s_lat = math.sin(lat_rad)
    c_lat = math.cos(lat_rad)
    s_lon = math.sin(lon_rad)
    c_lon = math.cos(lon_rad)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * s_lat * s_lat)
    x = (N + h) * c_lat * c_lon
    y = (N + h) * c_lat * s_lon
    z = (N * (1 - WGS84_E2) + h) * s_lat
    return np.array([x, y, z])


def ecef_to_latlonh(ecef: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = ecef
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1 - WGS84_E2))
    for _ in range(5):
        s_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * s_lat * s_lat)
        h = p / math.cos(lat) - N
        lat = math.atan2(z, p * (1 - WGS84_E2 * N / (N + h)))
    s_lat = math.sin(lat)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * s_lat * s_lat)
    h = p / math.cos(lat) - N
    return lat, lon, h


def enu_to_ecef(enu: np.ndarray, origin_lat: float, origin_lon: float, origin_h: float) -> np.ndarray:
    origin_ecef = latlon_to_ecef(origin_lat, origin_lon, origin_h)
    R = ecef_to_enu_matrix(origin_lat, origin_lon)
    return origin_ecef + R.T @ enu


def enu_to_geodetic(enu: np.ndarray, origin_lat: float, origin_lon: float, origin_h: float) -> Tuple[float, float, float]:
    ecef = enu_to_ecef(enu, origin_lat, origin_lon, origin_h)
    return ecef_to_latlonh(ecef)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def quat_normalize(q: np.ndarray) -> np.ndarray:
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


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_from_delta(delta: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(delta)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = delta / angle
    half = angle / 2
    return np.array([math.cos(half), *(axis * math.sin(half))])


def integrate_quat(q: np.ndarray, omega_b: np.ndarray, dt: float) -> np.ndarray:
    delta = omega_b * dt
    dq = quat_from_delta(delta)
    q_new = quat_multiply(q, dq)
    return quat_normalize(q_new)


def generate_trajectory(times: np.ndarray, traj_cfg: TrajectoryConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pattern = getattr(traj_cfg, "pattern", "circle")
    if pattern != "multi_segment":
        omega = traj_cfg.speed_mps / traj_cfg.radius_m
        east = traj_cfg.radius_m * np.cos(omega * times)
        north = traj_cfg.radius_m * np.sin(omega * times)
        up = np.full_like(times, traj_cfg.altitude_m)
        pos = np.column_stack((east, north, up))
        vel_e = -traj_cfg.radius_m * omega * np.sin(omega * times)
        vel_n = traj_cfg.radius_m * omega * np.cos(omega * times)
        vel_u = np.zeros_like(times)
        vel = np.column_stack((vel_e, vel_n, vel_u))
    else:
        total_time = traj_cfg.total_time
        speed = traj_cfg.speed_mps
        radius = max(traj_cfg.radius_m, 1e-6)
        altitude = traj_cfg.altitude_m
        t1 = 0.25 * total_time
        t2 = 0.5 * total_time
        t3 = 0.75 * total_time
        t4 = total_time
        east = np.zeros_like(times)
        north = np.zeros_like(times)
        up = np.zeros_like(times)
        vel_e = np.zeros_like(times)
        vel_n = np.zeros_like(times)
        vel_u = np.zeros_like(times)

        # Segment 1: accelerate eastward
        mask1 = times <= t1
        tau1 = times[mask1]
        accel_slope = (speed - 10.0) / t1 if t1 > 0 else 0.0
        east[mask1] = 10.0 * tau1 + 0.5 * accel_slope * tau1 * tau1
        north[mask1] = 0.0
        up[mask1] = altitude
        vel_e[mask1] = 10.0 + accel_slope * tau1
        vel_n[mask1] = 0.0
        vel_u[mask1] = 0.0

        # Segment 2: constant-speed left turn
        mask2 = (times > t1) & (times <= t2)
        tau2 = times[mask2] - t1
        east_t1 = 10.0 * t1 + 0.5 * accel_slope * t1 * t1
        north_t1 = 0.0
        center_e = east_t1
        center_n = north_t1 + radius
        omega = speed / radius
        theta2 = omega * tau2
        east[mask2] = center_e + radius * np.sin(theta2)
        north[mask2] = center_n - radius * np.cos(theta2)
        up[mask2] = altitude
        vel_e[mask2] = speed * np.cos(theta2)
        vel_n[mask2] = speed * np.sin(theta2)
        vel_u[mask2] = 0.0

        # Segment 3: northbound climb
        mask3 = (times > t2) & (times <= t3)
        tau3 = times[mask3] - t2
        theta_end = omega * (t2 - t1)
        east_t2 = center_e + radius * np.sin(theta_end)
        north_t2 = center_n - radius * np.cos(theta_end)
        alt_slope = 30.0 / (t3 - t2) if (t3 - t2) > 0 else 0.0
        east[mask3] = east_t2
        north[mask3] = north_t2 + speed * tau3
        up[mask3] = altitude + alt_slope * tau3
        vel_e[mask3] = 0.0
        vel_n[mask3] = speed
        vel_u[mask3] = alt_slope

        # Segment 4: right turn descending back
        mask4 = times > t3
        tau4 = times[mask4] - t3
        north_t3 = north_t2 + speed * (t3 - t2)
        alt_t3 = altitude + 30.0
        speed4 = 0.8 * speed
        omega4 = speed4 / radius
        center_e4 = east_t2 + radius
        center_n4 = north_t3
        theta4 = omega4 * tau4
        east[mask4] = center_e4 - radius * np.cos(theta4)
        north[mask4] = center_n4 + radius * np.sin(theta4)
        alt_desc = -30.0 / (t4 - t3) if (t4 - t3) > 0 else 0.0
        up[mask4] = alt_t3 + alt_desc * tau4
        vel_e[mask4] = speed4 * np.sin(theta4)
        vel_n[mask4] = speed4 * np.cos(theta4)
        vel_u[mask4] = alt_desc

        pos = np.column_stack((east, north, up))
        vel = np.column_stack((vel_e, vel_n, vel_u))

    yaw = np.unwrap(np.arctan2(vel[:, 0], vel[:, 1]))
    roll = np.zeros_like(yaw)
    pitch = np.zeros_like(yaw)
    quats = np.array([euler_to_quat(r, p, y) for r, p, y in zip(roll, pitch, yaw)])
    return pos, vel, quats


def normal_gravity(lat: float, h: float) -> float:
    sin_lat = math.sin(lat)
    g = 9.7803253359 * (1 + 0.00193185265241 * sin_lat * sin_lat) / math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    g -= 3.086e-6 * h
    return g


def compute_angular_rates(quats: np.ndarray, dt: float) -> np.ndarray:
    omegas = np.zeros_like(quats[:, :3])
    for k in range(1, len(quats)):
        dq = quat_multiply(quat_conjugate(quats[k - 1]), quats[k])
        dq = quat_normalize(dq)
        angle = 2 * math.acos(np.clip(dq[0], -1.0, 1.0))
        if angle < 1e-12:
            rot_vec = np.zeros(3)
        else:
            axis = dq[1:] / math.sin(angle / 2)
            rot_vec = axis * angle
        omegas[k - 1] = rot_vec / dt
    omegas[-1] = omegas[-2]
    return omegas


def compute_ideal_imu(
    times: np.ndarray,
    pos_enu: np.ndarray,
    vel_enu: np.ndarray,
    quats: np.ndarray,
    origin_lat_deg: float,
    origin_lon_deg: float,
    origin_h: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = times[1] - times[0]
    acc = np.column_stack(
        (
            np.gradient(vel_enu[:, 0], times),
            np.gradient(vel_enu[:, 1], times),
            np.gradient(vel_enu[:, 2], times),
        )
    )
    latitudes = np.zeros(len(times))
    longitudes = np.zeros(len(times))
    heights = np.zeros(len(times))
    omega_ie_n = np.zeros_like(pos_enu)
    g_n = np.zeros_like(pos_enu)
    origin_lat = math.radians(origin_lat_deg)
    origin_lon = math.radians(origin_lon_deg)
    for k, pos in enumerate(pos_enu):
        lat, lon, h = enu_to_geodetic(pos, origin_lat, origin_lon, origin_h)
        latitudes[k] = lat
        longitudes[k] = lon
        heights[k] = h
        R = ecef_to_enu_matrix(lat, lon)
        omega_ie_n[k] = R @ np.array([0.0, 0.0, OMEGA_EARTH])
        g_mag = normal_gravity(lat, h)
        g_n[k] = np.array([0.0, 0.0, -g_mag])
    omega_nb_b = compute_angular_rates(quats, dt)
    gyro = np.zeros_like(omega_nb_b)
    accel = np.zeros_like(acc)
    for k in range(len(times)):
        C_nb = quat_to_dcm(quats[k])
        gyro[k] = omega_nb_b[k] + C_nb @ omega_ie_n[k]
        f_nav = acc[k] - g_n[k] + 2 * np.cross(omega_ie_n[k], vel_enu[k])
        accel[k] = C_nb @ f_nav
    return gyro, accel, latitudes, longitudes, heights


def add_imu_errors(
    gyro_true: np.ndarray,
    accel_true: np.ndarray,
    imu_cfg: IMUErrorModel,
    dt: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(gyro_true)
    gyro_bias = rng.normal(0.0, imu_cfg.gyro_bias_init_std, size=3)
    accel_bias = rng.normal(0.0, imu_cfg.accel_bias_init_std, size=3)
    gyro_meas = np.zeros_like(gyro_true)
    accel_meas = np.zeros_like(accel_true)
    gyro_bias_hist = np.zeros_like(gyro_true)
    accel_bias_hist = np.zeros_like(accel_true)
    for k in range(n):
        gyro_bias += imu_cfg.gyro_bias_rw * math.sqrt(dt) * rng.normal(size=3)
        accel_bias += imu_cfg.accel_bias_rw * math.sqrt(dt) * rng.normal(size=3)
        gyro_noise = (imu_cfg.gyro_noise_std / math.sqrt(dt)) * rng.normal(size=3)
        accel_noise = (imu_cfg.accel_noise_std / math.sqrt(dt)) * rng.normal(size=3)
        gyro_meas[k] = gyro_true[k] + gyro_bias + gyro_noise
        accel_meas[k] = accel_true[k] + accel_bias + accel_noise
        gyro_bias_hist[k] = gyro_bias
        accel_bias_hist[k] = accel_bias
    return gyro_meas, accel_meas, gyro_bias_hist, accel_bias_hist


def ins_mechanization(
    times: np.ndarray,
    gyro_meas: np.ndarray,
    accel_meas: np.ndarray,
    origin_lat_deg: float,
    origin_lon_deg: float,
    origin_h: float,
    init_pos: np.ndarray,
    init_vel: np.ndarray,
    init_quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(times)
    pos_est = np.zeros_like(gyro_meas)
    vel_est = np.zeros_like(gyro_meas)
    quats = np.zeros((n, 4))
    pos = init_pos.copy()
    vel = init_vel.copy()
    quat = init_quat.copy()
    pos_est[0] = pos
    vel_est[0] = vel
    quats[0] = quat
    origin_lat = math.radians(origin_lat_deg)
    origin_lon = math.radians(origin_lon_deg)
    for k in range(1, n):
        dt = times[k] - times[k - 1]
        lat, lon, h = enu_to_geodetic(pos, origin_lat, origin_lon, origin_h)
        R = ecef_to_enu_matrix(lat, lon)
        omega_ie_n = R @ np.array([0.0, 0.0, OMEGA_EARTH])
        g_mag = normal_gravity(lat, h)
        g_n = np.array([0.0, 0.0, -g_mag])
        C_nb = quat_to_dcm(quat)
        omega_nb_b = gyro_meas[k - 1] - C_nb @ omega_ie_n
        quat = integrate_quat(quat, omega_nb_b, dt)
        C_nb = quat_to_dcm(quat)
        f_nav = C_nb.T @ accel_meas[k - 1]
        acc_nav = f_nav + g_n - 2 * np.cross(omega_ie_n, vel)
        vel = vel + acc_nav * dt
        pos = pos + vel * dt
        pos_est[k] = pos
        vel_est[k] = vel
        quats[k] = quat
    return pos_est, vel_est, quats


def simulate_gnss(
    gnss_times: np.ndarray,
    imu_times: np.ndarray,
    truth_pos: np.ndarray,
    truth_vel: np.ndarray,
    gnss_cfg: GNSSNoiseModel,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos_truth = np.column_stack([np.interp(gnss_times, imu_times, truth_pos[:, i]) for i in range(3)])
    vel_truth = np.column_stack([np.interp(gnss_times, imu_times, truth_vel[:, i]) for i in range(3)])
    pos_meas = pos_truth + rng.normal(loc=0.0, scale=gnss_cfg.pos_noise_std, size=pos_truth.shape)
    vel_meas = vel_truth + rng.normal(loc=0.0, scale=gnss_cfg.vel_noise_std, size=vel_truth.shape)
    for start, end in gnss_cfg.outage_intervals:
        mask = (gnss_times >= start) & (gnss_times <= end)
        pos_meas[mask] = np.nan
        vel_meas[mask] = np.nan
    return pos_meas, vel_meas, pos_truth, vel_truth


def run_simulation(config: SimConfig) -> None:
    rng = np.random.default_rng(config.seed)
    imu_times, gnss_times = generate_time_axes(config.trajectory, config.gnss.dt)
    pos_truth, vel_truth, quats_truth = generate_trajectory(imu_times, config.trajectory)
    gyro_true, accel_true, lat, lon, h = compute_ideal_imu(
        imu_times,
        pos_truth,
        vel_truth,
        quats_truth,
        config.origin_lat_deg,
        config.origin_lon_deg,
        config.origin_height_m,
    )
    (
        gyro_meas,
        accel_meas,
        gyro_bias_hist,
        accel_bias_hist,
    ) = add_imu_errors(gyro_true, accel_true, config.imu_errors, config.trajectory.imu_dt, rng)
    pos_est, vel_est, quats_est = ins_mechanization(
        imu_times,
        gyro_meas,
        accel_meas,
        config.origin_lat_deg,
        config.origin_lon_deg,
        config.origin_height_m,
        pos_truth[0],
        vel_truth[0],
        quats_truth[0],
    )
    gnss_pos_meas, gnss_vel_meas, gnss_pos_truth, gnss_vel_truth = simulate_gnss(
        gnss_times,
        imu_times,
        pos_truth,
        vel_truth,
        config.gnss,
        rng,
    )
    gnss_pos_interp = np.full_like(pos_truth, np.nan)
    gnss_vel_interp = np.full_like(vel_truth, np.nan)
    gnss_nan_mask = np.any(np.isnan(gnss_pos_meas), axis=1) | np.any(np.isnan(gnss_vel_meas), axis=1)
    outage_intervals: list[Tuple[float, float]] = []
    if np.any(gnss_nan_mask):
        nan_indices = np.where(gnss_nan_mask)[0]
        start_idx = nan_indices[0]
        prev_idx = nan_indices[0]
        for idx in nan_indices[1:]:
            if idx == prev_idx + 1:
                prev_idx = idx
                continue
            outage_intervals.append((gnss_times[start_idx], gnss_times[prev_idx]))
            start_idx = idx
            prev_idx = idx
        outage_intervals.append((gnss_times[start_idx], gnss_times[prev_idx]))
    imu_outage_mask = np.zeros_like(imu_times, dtype=bool)
    for start, end in outage_intervals:
        imu_outage_mask |= (imu_times >= start) & (imu_times <= end)
    for i in range(3):
        valid_pos = ~np.isnan(gnss_pos_meas[:, i])
        if np.any(valid_pos):
            gnss_pos_interp[:, i] = np.interp(
                imu_times,
                gnss_times[valid_pos],
                gnss_pos_meas[valid_pos, i],
                left=np.nan,
                right=np.nan,
            )
        valid_vel = ~np.isnan(gnss_vel_meas[:, i])
        if np.any(valid_vel):
            gnss_vel_interp[:, i] = np.interp(
                imu_times,
                gnss_times[valid_vel],
                gnss_vel_meas[valid_vel, i],
                left=np.nan,
                right=np.nan,
            )
    gnss_pos_interp[imu_outage_mask] = np.nan
    gnss_vel_interp[imu_outage_mask] = np.nan
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    imu_df = pd.DataFrame(
        {
            "time": imu_times,
            "gyro_x": gyro_meas[:, 0],
            "gyro_y": gyro_meas[:, 1],
            "gyro_z": gyro_meas[:, 2],
            "accel_x": accel_meas[:, 0],
            "accel_y": accel_meas[:, 1],
            "accel_z": accel_meas[:, 2],
            "pos_truth_e": pos_truth[:, 0],
            "pos_truth_n": pos_truth[:, 1],
            "pos_truth_u": pos_truth[:, 2],
            "vel_truth_e": vel_truth[:, 0],
            "vel_truth_n": vel_truth[:, 1],
            "vel_truth_u": vel_truth[:, 2],
            "pos_est_e": pos_est[:, 0],
            "pos_est_n": pos_est[:, 1],
            "pos_est_u": pos_est[:, 2],
            "vel_est_e": vel_est[:, 0],
            "vel_est_n": vel_est[:, 1],
            "vel_est_u": vel_est[:, 2],
            "ins_q_w": quats_est[:, 0],
            "ins_q_x": quats_est[:, 1],
            "ins_q_y": quats_est[:, 2],
            "ins_q_z": quats_est[:, 3],
            "gyro_bias_x": gyro_bias_hist[:, 0],
            "gyro_bias_y": gyro_bias_hist[:, 1],
            "gyro_bias_z": gyro_bias_hist[:, 2],
            "accel_bias_x": accel_bias_hist[:, 0],
            "accel_bias_y": accel_bias_hist[:, 1],
            "accel_bias_z": accel_bias_hist[:, 2],
        }
    )

    imu_path = output_dir / "imu_and_ins.csv"
    imu_df.to_csv(imu_path, index=False)
    gnss_df = pd.DataFrame(
        {
            "time": gnss_times,
            "gnss_pos_e": gnss_pos_meas[:, 0],
            "gnss_pos_n": gnss_pos_meas[:, 1],
            "gnss_pos_u": gnss_pos_meas[:, 2],
            "gnss_vel_e": gnss_vel_meas[:, 0],
            "gnss_vel_n": gnss_vel_meas[:, 1],
            "gnss_vel_u": gnss_vel_meas[:, 2],
            "truth_pos_e": gnss_pos_truth[:, 0],
            "truth_pos_n": gnss_pos_truth[:, 1],
            "truth_pos_u": gnss_pos_truth[:, 2],
            "truth_vel_e": gnss_vel_truth[:, 0],
            "truth_vel_n": gnss_vel_truth[:, 1],
            "truth_vel_u": gnss_vel_truth[:, 2],
        }
    )
    gnss_path = output_dir / "gnss.csv"
    gnss_df.to_csv(gnss_path, index=False)
    bias_df = pd.DataFrame(
        {
            "time": imu_times,
            "gyro_x": gyro_meas[:, 0],
            "gyro_y": gyro_meas[:, 1],
            "gyro_z": gyro_meas[:, 2],
            "accel_x": accel_meas[:, 0],
            "accel_y": accel_meas[:, 1],
            "accel_z": accel_meas[:, 2],
            "ins_pos_e": pos_est[:, 0],
            "ins_pos_n": pos_est[:, 1],
            "ins_pos_u": pos_est[:, 2],
            "ins_vel_e": vel_est[:, 0],
            "ins_vel_n": vel_est[:, 1],
            "ins_vel_u": vel_est[:, 2],
            "gnss_pos_e": gnss_pos_interp[:, 0],
            "gnss_pos_n": gnss_pos_interp[:, 1],
            "gnss_pos_u": gnss_pos_interp[:, 2],
            "gnss_vel_e": gnss_vel_interp[:, 0],
            "gnss_vel_n": gnss_vel_interp[:, 1],
            "gnss_vel_u": gnss_vel_interp[:, 2],
            "truth_pos_e": pos_truth[:, 0],
            "truth_pos_n": pos_truth[:, 1],
            "truth_pos_u": pos_truth[:, 2],
            "truth_vel_e": vel_truth[:, 0],
            "truth_vel_n": vel_truth[:, 1],
            "truth_vel_u": vel_truth[:, 2],
            "gyro_bias_x": gyro_bias_hist[:, 0],
            "gyro_bias_y": gyro_bias_hist[:, 1],
            "gyro_bias_z": gyro_bias_hist[:, 2],
            "accel_bias_x": accel_bias_hist[:, 0],
            "accel_bias_y": accel_bias_hist[:, 1],
            "accel_bias_z": accel_bias_hist[:, 2],
        }
    )
    bias_path = output_dir / "bias_training.csv"
    bias_df.to_csv(bias_path, index=False)
    print(f"Saved IMU+INS data to {imu_path}")
    print(f"Saved GNSS data to {gnss_path}")
    print(f"Saved bias training data to {bias_path}")


if __name__ == "__main__":
    sim_cfg = SimConfig(
        trajectory=TrajectoryConfig(
            total_time=120.0,
            imu_dt=0.01,
            speed_mps=20.0,
            radius_m=120.0,
            altitude_m=30.0,
            pattern="multi_segment",
        ),
        imu_errors=IMUErrorModel(
            gyro_bias_rw=5e-6,
            gyro_noise_std=5e-4,
            accel_bias_rw=2e-5,
            accel_noise_std=1e-3,
        ),
        gnss=GNSSNoiseModel(dt=1.0, pos_noise_std=1.5, vel_noise_std=0.2, outage_intervals=((40.0, 60.0),)),
        output_dir=Path("../outputs/synthetic"),
    )
    run_simulation(sim_cfg)
