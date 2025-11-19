"""PyTorch pipeline for learning INS drift from synthetic GNSS/INS sequences."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed common RNGs for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Dataset definition
# -----------------------------------------------------------------------------


class DriftSequenceDataset(Dataset):
    """Create sliding-window IMU+INS sequences and drift labels."""

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

        # Prepare default feature set if none provided
        default_features = [
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "accel_x",
            "accel_y",
            "accel_z",
            "pos_est_e",
            "pos_est_n",
            "pos_est_u",
        ]
        if include_velocity:
            default_features.extend([
                "vel_est_e",
                "vel_est_n",
                "vel_est_u",
            ])
        self.feature_columns = list(feature_columns) if feature_columns else default_features
        for col in self.feature_columns:
            if col not in df.columns:
                raise ValueError(f"Missing feature column '{col}' in {self.csv_path}")

        required_truth = ["pos_truth_e", "pos_truth_n", "pos_truth_u"]
        required_est = ["pos_est_e", "pos_est_n", "pos_est_u"]
        for col in required_truth + required_est:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {self.csv_path}")

        self.times = df["time"].to_numpy(dtype=np.float64)
        self.pos_truth = df[required_truth].to_numpy(dtype=np.float32)
        self.pos_est = df[required_est].to_numpy(dtype=np.float32)

        feature_matrix = df[self.feature_columns].to_numpy(dtype=np.float32)
        if feature_mean is None or feature_std is None:
            self.feature_mean = feature_matrix.mean(axis=0)
            self.feature_std = feature_matrix.std(axis=0) + 1e-6
        else:
            self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
            self.feature_std = np.asarray(feature_std, dtype=np.float32)
        if self.feature_mean.shape[0] != feature_matrix.shape[1]:
            raise ValueError("Feature mean/std dimension mismatch")
        normalized_features = (feature_matrix - self.feature_mean) / self.feature_std
        self.features = normalized_features.astype(np.float32)

        self.drift = (self.pos_truth - self.pos_est).astype(np.float32)
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
        x = torch.from_numpy(self.features[start:end])  # (T, D)
        y = torch.from_numpy(self.drift[end - 1])  # (3,)
        meta = {
            "time": torch.tensor(self.times[end - 1], dtype=torch.float32),
            "pos_est": torch.from_numpy(self.pos_est[end - 1]),
            "pos_truth": torch.from_numpy(self.pos_truth[end - 1]),
        }
        return x, y, meta

    @property
    def input_dim(self) -> int:
        return self.features.shape[1]


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------


class DriftGRUModel(nn.Module):
    """GRU encoder followed by MLP regression head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
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
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]  # (B, H)
        return self.head(last_hidden)


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    csv_path: Path
    output_path: Path
    window_size: int = 200
    stride: int = 5
    batch_size: int = 64
    val_ratio: float = 0.2
    epochs: int = 30
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-4
    grad_clip: float = 1.0
    include_velocity: bool = True
    seed: int = 42


@dataclass
class EvalConfig:
    csv_path: Path
    checkpoint_path: Path
    output_csv: Path
    window_size: int
    stride: int
    batch_size: int = 128
    include_velocity: bool = True
    device: str = "cpu"


def build_dataloaders(cfg: TrainConfig) -> Tuple[Dataset, Dataset, DataLoader, Optional[DataLoader]]:
    dataset = DriftSequenceDataset(
        csv_path=cfg.csv_path,
        window_size=cfg.window_size,
        stride=cfg.stride,
        include_velocity=cfg.include_velocity,
    )
    val_len = max(1, int(len(dataset) * cfg.val_ratio))
    train_len = len(dataset) - val_len
    if train_len <= 0:
        raise ValueError("Validation split too large for available samples")
    generator = torch.Generator().manual_seed(cfg.seed)
    train_subset, val_subset = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    return dataset, train_subset, train_loader, val_loader


def train_model(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    full_dataset, _, train_loader, val_loader = build_dataloaders(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DriftGRUModel(
        input_dim=full_dataset.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val_loss = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses: List[float] = []
        for batch in train_loader:
            inputs, targets, _ = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, _ = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state": model.state_dict(),
                "feature_mean": full_dataset.feature_mean,
                "feature_std": full_dataset.feature_std,
                "model_args": {
                    "input_dim": full_dataset.input_dim,
                    "hidden_dim": cfg.hidden_dim,
                    "num_layers": cfg.num_layers,
                    "dropout": cfg.dropout,
                },
            }
        print(f"Epoch {epoch:03d}/{cfg.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state")

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, cfg.output_path)

    meta_path = cfg.output_path.with_suffix(".json")

    cfg_dict = asdict(cfg)
    # Path 객체들을 문자열로 변환
    for k, v in cfg_dict.items():
        if isinstance(v, Path):
            cfg_dict[k] = str(v)

    meta = {"best_val_loss": best_val_loss, **cfg_dict}

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved best model with val loss {best_val_loss:.6f} to {cfg.output_path}")


# -----------------------------------------------------------------------------
# Evaluation utilities
# -----------------------------------------------------------------------------


def run_evaluation(cfg: EvalConfig) -> None:
    device = torch.device(cfg.device)
    checkpoint = torch.load(
    cfg.checkpoint_path,
    map_location=device,
    weights_only=False,  # PyTorch 2.6 이상에서 필요
    )
    model_args = checkpoint["model_args"]
    model = DriftGRUModel(**model_args).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = DriftSequenceDataset(
        csv_path=cfg.csv_path,
        window_size=cfg.window_size,
        stride=cfg.stride,
        include_velocity=cfg.include_velocity,
        feature_mean=checkpoint["feature_mean"],
        feature_std=checkpoint["feature_std"],
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for inputs, _, meta in loader:
            inputs = inputs.to(device)
            preds = model(inputs).cpu().numpy()
            times = meta["time"].cpu().numpy()
            pos_est = meta["pos_est"].cpu().numpy()
            pos_truth = meta["pos_truth"].cpu().numpy()
            drift_true = pos_truth - pos_est
            pos_corr = pos_est + preds
            for i in range(preds.shape[0]):
                rows.append(
                    {
                        "time": float(times[i]),
                        "pos_est_e": float(pos_est[i, 0]),
                        "pos_est_n": float(pos_est[i, 1]),
                        "pos_est_u": float(pos_est[i, 2]),
                        "drift_pred_e": float(preds[i, 0]),
                        "drift_pred_n": float(preds[i, 1]),
                        "drift_pred_u": float(preds[i, 2]),
                        "pos_corr_e": float(pos_corr[i, 0]),
                        "pos_corr_n": float(pos_corr[i, 1]),
                        "pos_corr_u": float(pos_corr[i, 2]),
                        "drift_true_e": float(drift_true[i, 0]),
                        "drift_true_n": float(drift_true[i, 1]),
                        "drift_true_u": float(drift_true[i, 2]),
                    }
                )

    output_df = pd.DataFrame(rows)
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(cfg.output_csv, index=False)
    print(f"Saved drift-corrected positions to {cfg.output_csv}")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate an INS drift predictor.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a GRU drift model")
    train_parser.add_argument("--csv", type=Path, required=True, help="Path to imu_and_ins.csv")
    train_parser.add_argument("--output", type=Path, default=Path("checkpoints/drift_gru.pt"), help="Where to save the trained model")
    train_parser.add_argument("--window-size", type=int, default=200)
    train_parser.add_argument("--stride", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--val-ratio", type=float, default=0.2)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--num-layers", type=int, default=2)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--no-velocity", action="store_true", help="Exclude velocity estimates from inputs")
    train_parser.add_argument("--seed", type=int, default=42)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model and export corrected positions")
    eval_parser.add_argument("--csv", type=Path, required=True, help="Path to imu_and_ins.csv for evaluation")
    eval_parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model checkpoint")
    eval_parser.add_argument("--output-csv", type=Path, default=Path("outputs/drift_corrections.csv"))
    eval_parser.add_argument("--window-size", type=int, default=200)
    eval_parser.add_argument("--stride", type=int, default=1)
    eval_parser.add_argument("--batch-size", type=int, default=128)
    eval_parser.add_argument("--no-velocity", action="store_true")
    eval_parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        cfg = TrainConfig(
            csv_path=args.csv,
            output_path=args.output,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            grad_clip=args.grad_clip,
            include_velocity=not args.no_velocity,
            seed=args.seed,
        )
        train_model(cfg)
    elif args.command == "eval":
        cfg = EvalConfig(
            csv_path=args.csv,
            checkpoint_path=args.checkpoint,
            output_csv=args.output_csv,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            include_velocity=not args.no_velocity,
            device=args.device,
        )
        run_evaluation(cfg)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
