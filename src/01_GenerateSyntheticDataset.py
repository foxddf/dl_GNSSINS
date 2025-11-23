# 03_generate_datasets.py
from pathlib import Path
import numpy as np

from GenerateSyntheticData import (  # <= 여기 이름을 실제 파일명에 맞게 바꿔줘 (예: synthetic_sim)
    SimConfig,
    TrajectoryConfig,
    IMUErrorModel,
    GNSSNoiseModel,
    run_simulation,
)

def random_outages(total_time: float, rng: np.random.Generator, max_outages: int = 2):
    """총 비행시간 안에서 랜덤 GNSS outage interval들을 만들어준다."""
    n_out = rng.integers(0, max_outages + 1)  # 0, 1, 2개 중 하나
    n_out = 1 # 1개 고정
    intervals = []
    for _ in range(n_out):
        start = rng.uniform(0.1 * total_time, 0.8 * total_time)
        dur = rng.uniform(5.0, 30.0)  # 5~30초 outage
        end = min(start + dur, total_time)
        intervals.append((start, end))
    # 시간 순으로 정렬
    intervals.sort(key=lambda x: x[0])
    return tuple(intervals)

def main():
    base_output = Path("../outputs/synthetic_multi")
    base_output.mkdir(parents=True, exist_ok=True)

    base_seed = 42
    n_datasets = 20  # 만들고 싶은 데이터셋 개수

    rng = np.random.default_rng(base_seed)

    for idx in range(n_datasets):
        # 1) 다양한 궤적/속도/반경 랜덤 선택
        pattern = rng.choice(["circle", "multi_segment"])
        speed = float(rng.uniform(15.0, 35.0))      # 15~35 m/s
        radius = float(rng.uniform(80.0, 200.0))    # 80~200 m
        altitude = float(rng.uniform(20.0, 80.0))   # 20~80 m
        total_time = float(rng.uniform(80.0, 180.0))  # 80~180초

        traj_cfg = TrajectoryConfig(
            total_time=total_time,
            imu_dt=0.01,
            speed_mps=speed,
            radius_m=radius,
            altitude_m=altitude,
            pattern=pattern,
        )

        # 2) IMU 노이즈/바이어스 수준을 약간씩 랜덤화
        imu_cfg = IMUErrorModel(
            gyro_bias_rw=5e-6,
            gyro_noise_std=5e-4,
            accel_bias_rw=2e-5,
            accel_noise_std=1e-3,
            gyro_bias_init_std=5e-4,
            accel_bias_init_std=5e-4,
        )

        # 3) GNSS 노이즈 + outage 패턴 랜덤
        outages = random_outages(total_time, rng, max_outages=2)
        gnss_cfg = GNSSNoiseModel(
            dt=1.0,
            pos_noise_std=1.5, 
            vel_noise_std=0.2,
            outage_intervals=outages,
        )

        # 4) 출력 디렉토리와 seed 설정
        out_dir = base_output / f"scenario_{idx:03d}"
        cfg = SimConfig(
            origin_lat_deg=37.0,
            origin_lon_deg=127.0,
            origin_height_m=50.0,
            trajectory=traj_cfg,
            imu_errors=imu_cfg,
            gnss=gnss_cfg,
            gravity=9.80665,
            seed=base_seed + idx,
            output_dir=out_dir,
        )

        print(f"=== Running scenario {idx:03d} ===")
        print(f" pattern={pattern}, speed={speed:.1f}, radius={radius:.1f}, "
              f"alt={altitude:.1f}, T={total_time:.1f}, outages={outages}")
        run_simulation(cfg)

if __name__ == "__main__":
    main()
