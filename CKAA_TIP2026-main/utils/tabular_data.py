import csv
import json
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


FAULT_ORDER = ("A", "B", "B+G", "C", "E", "G", "H", "H+N", "N")
RECORD_RE = re.compile(r"^电流-(.+)-S1-(\d+)\.csv$")


def _read_csv_prefix(path: str, nrows: int, usecols: list[int] | None = None) -> np.ndarray:
    import pandas as pd

    frame = pd.read_csv(
        path,
        header=None,
        nrows=nrows,
        usecols=usecols,
        engine="c",
        on_bad_lines="skip",
    )
    # Empty CSV fields represent sensor zeros, not missing measurements.
    values = (
        frame.apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64, copy=False)
    )
    if values.ndim == 1:
        values = values[:, None]
    return values


def _recording_key(path: Path) -> tuple[str, str] | None:
    match = RECORD_RE.match(path.name)
    if match is None:
        return None
    return match.group(1), match.group(2)


def _window_count(num_samples: int, window_length: int, stride: int) -> int:
    if num_samples < window_length:
        return 0
    return 1 + (num_samples - window_length) // stride


def build_tabular_dataset(
    raw_root: str,
    output_root: str,
    window_length: int = 1568,
    stride: int = 1568,
    max_common_samples: int = 614400,
    train_ratio: float = 0.8,
    overwrite: bool = False,
) -> dict:
    """Convert vibration and pressure-XYZ CSV streams into memmap-backed windows."""
    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root).resolve()
    vibration_root = raw_root / "data_1_S1"
    pressure_root = raw_root / "data_2_S1"

    if not vibration_root.is_dir() or not pressure_root.is_dir():
        raise FileNotFoundError(
            f"Expected data_1_S1 and data_2_S1 under {raw_root}"
        )
    if window_length <= 0 or stride <= 0 or window_length % 32:
        raise ValueError("window_length must be positive and divisible by 32")
    if not 0.5 <= train_ratio < 1.0:
        raise ValueError("train_ratio must be in [0.5, 1.0)")

    output_root.mkdir(parents=True, exist_ok=True)
    data_paths = {
        "train": {
            "windows": output_root / "train_windows.npy",
            "labels": output_root / "train_labels.npy",
            "index": output_root / "train_index.csv",
        },
        "eval": {
            "windows": output_root / "eval_windows.npy",
            "labels": output_root / "eval_labels.npy",
            "index": output_root / "eval_index.csv",
        },
    }
    meta_path = output_root / "meta.json"
    existing = [p for split in data_paths.values() for p in split.values()]
    if meta_path.exists() and all(p.exists() for p in existing) and not overwrite:
        with meta_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    files: dict[tuple[str, str], tuple[Path, Path]] = {}
    for vibration_path in sorted(vibration_root.glob("*.csv")):
        key = _recording_key(vibration_path)
        if key is None:
            continue
        pressure_path = pressure_root / vibration_path.name
        if pressure_path.exists():
            files[key] = (vibration_path, pressure_path)

    labels = [name for name in FAULT_ORDER if any(key[0] == name for key in files)]
    if not labels:
        raise RuntimeError(f"No recognized recordings found in {vibration_root}")
    label_to_index = {name: index for index, name in enumerate(labels)}

    train_end = int(max_common_samples * train_ratio)
    vibration_sum = 0.0
    vibration_sumsq = 0.0
    vibration_count = 0
    pressure_sum = np.zeros(3, dtype=np.float64)
    pressure_sumsq = np.zeros(3, dtype=np.float64)
    pressure_count = np.zeros(3, dtype=np.float64)

    for (fault, condition), (vibration_path, pressure_path) in sorted(files.items()):
        vibration = _read_csv_prefix(str(vibration_path), max_common_samples, usecols=[0])[:, 0]
        vibration = vibration[:train_end]
        finite = np.isfinite(vibration)
        vibration_sum += float(vibration[finite].sum())
        vibration_sumsq += float(np.square(vibration[finite]).sum())
        vibration_count += int(finite.sum())

        # Pressure XYZ exists only in the -01 recordings. Their statistics are shared
        # by all fault classes so zero-filled -03 pressure channels stay comparable.
        if condition == "01":
            pressure = _read_csv_prefix(str(pressure_path), max_common_samples, usecols=[1, 2, 3])
            pressure = pressure[:train_end]
            finite = np.isfinite(pressure)
            pressure_sum += np.where(finite, pressure, 0.0).sum(axis=0)
            pressure_sumsq += np.where(finite, np.square(pressure), 0.0).sum(axis=0)
            pressure_count += finite.sum(axis=0)

    vibration_mean = vibration_sum / max(vibration_count, 1)
    vibration_var = vibration_sumsq / max(vibration_count, 1) - vibration_mean**2
    vibration_std = max(float(np.sqrt(max(vibration_var, 0.0))), 1e-8)

    pressure_count = np.maximum(pressure_count, 1)
    pressure_mean = pressure_sum / pressure_count
    pressure_var = pressure_sumsq / pressure_count - np.square(pressure_mean)
    pressure_std = np.maximum(np.sqrt(np.maximum(pressure_var, 0.0)), 1e-8)
    means = np.concatenate([[vibration_mean], pressure_mean]).astype(np.float32)
    stds = np.concatenate([[vibration_std], pressure_std]).astype(np.float32)

    windows: dict[str, list[np.ndarray]] = {"train": [], "eval": []}
    window_labels: dict[str, list[int]] = {"train": [], "eval": []}
    index_rows: dict[str, list[dict]] = {"train": [], "eval": []}

    for (fault, condition), (vibration_path, pressure_path) in sorted(files.items()):
        vibration = _read_csv_prefix(str(vibration_path), max_common_samples, usecols=[0])[:, 0]
        pressure = _read_csv_prefix(str(pressure_path), max_common_samples, usecols=[1, 2, 3])
        signal = np.concatenate([vibration[:, None], pressure], axis=1)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        signal = (signal - means) / stds
        signal = np.clip(signal, -12.0, 12.0).astype(np.float32)

        num_windows = _window_count(len(signal), window_length, stride)
        split_point = int(num_windows * train_ratio)
        for window_index in range(num_windows):
            start = window_index * stride
            split = "train" if window_index < split_point else "eval"
            # Keep channel-major storage; the tabular tokenizer interleaves channels.
            window = signal[start : start + window_length].T.copy()
            windows[split].append(window.astype(np.float16))
            window_labels[split].append(label_to_index[fault])
            index_rows[split].append(
                {
                    "label": label_to_index[fault],
                    "fault": fault,
                    "condition": condition,
                    "source": vibration_path.name,
                    "window": window_index,
                    "start": start,
                }
            )

    for split in ("train", "eval"):
        if not windows[split]:
            raise RuntimeError(f"No {split} windows were generated")
        np.save(data_paths[split]["windows"], np.stack(windows[split]).astype(np.float16))
        np.save(data_paths[split]["labels"], np.asarray(window_labels[split], dtype=np.int64))
        with data_paths[split]["index"].open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=index_rows[split][0].keys())
            writer.writeheader()
            writer.writerows(index_rows[split])

    meta = {
        "format": "CKAA tabular windows",
        "raw_root": str(raw_root),
        "window_length": window_length,
        "stride": stride,
        "max_common_samples": max_common_samples,
        "train_ratio": train_ratio,
        "channels": ["vibration", "pressure_x", "pressure_y", "pressure_z"],
        "labels": labels,
        "label_to_index": label_to_index,
        "normalization": {
            "mean": means.tolist(),
            "std": stds.tolist(),
        },
        "num_patches": window_length * 4 // 32,
        "files": {
            split: {
                "windows": str(data_paths[split]["windows"]),
                "labels": str(data_paths[split]["labels"]),
                "index": str(data_paths[split]["index"]),
                "num_samples": len(window_labels[split]),
                "class_counts": {
                    label: int(window_labels[split].count(label_to_index[label]))
                    for label in labels
                },
            }
            for split in ("train", "eval")
        },
    }
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
    return meta


class TabularPathDataset:
    def __init__(self, root_dir: str, train: bool):
        self.root_dir = Path(root_dir)
        with (self.root_dir / "meta.json").open("r", encoding="utf-8") as file:
            self.meta = json.load(file)
        split_name = "train" if train else "eval"
        split = self.meta["files"][split_name]
        self.split_path = str(self.root_dir)
        self.windows = np.load(split["windows"], mmap_mode="r")
        self.labels = np.load(split["labels"], mmap_mode="r")
        self.class_list = list(range(len(self.meta["labels"])))
        self.class_int_str_map = {
            index: name for index, name in enumerate(self.meta["labels"])
        }
        self.class_window_indices = {
            label: np.flatnonzero(self.labels == label).tolist()
            for label in self.class_list
        }

    def __getitem__(self, label: int):
        return self.class_window_indices[label]

    def __len__(self):
        return len(self.class_list)

    @property
    def num_classes(self):
        return len(self)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(root_dir={self.root_dir!r}, "
            f"samples={len(self.labels)}, classes={len(self.class_list)})"
        )


class TabularIncrementalDataset(Dataset):
    def __init__(
        self,
        path_dataset: TabularPathDataset,
        task_classes: list[int],
        label_map_g2l: dict[int, tuple[int, int, int]],
        training: bool,
        target_map_to_local: bool,
        expand_times: int = 1,
        return_index: bool = True,
    ):
        self.path_dataset = path_dataset
        self.training = training
        self.target_map_to_local = target_map_to_local
        self.label_map_g2l = deepcopy(label_map_g2l)
        self.expand_times = max(int(expand_times), 1)
        self.return_index = return_index
        self.samples = [
            index
            for label in task_classes
            for index in path_dataset.class_window_indices[label]
        ]
        self.labels = [
            path_dataset.labels[index]
            for index in self.samples
        ]
        self.num_samples = len(self.samples)

    def _target(self, global_label: int) -> int:
        task_id, local_label, mapped_label = self.label_map_g2l[int(global_label)]
        return local_label if self.target_map_to_local else mapped_label

    def __getitem__(self, index: int):
        index %= self.num_samples
        window_index = self.samples[index]
        window = torch.from_numpy(
            np.asarray(self.path_dataset.windows[window_index], dtype=np.float32)
        ).clone()

        if self.training:
            max_shift = 8
            shift = int(torch.randint(-max_shift, max_shift + 1, ()).item())
            if shift:
                window = torch.roll(window, shifts=shift, dims=1)
            scale = 1.0 + (torch.rand(window.shape[0], 1) - 0.5) * 0.02
            window = window * scale
            window = window + torch.randn_like(window) * 0.01

        label = self._target(int(self.path_dataset.labels[window_index]))
        if self.return_index:
            return window, label, index
        return window, label

    def __len__(self):
        return self.num_samples * self.expand_times


def define_tabular_dataset(
    GVM,
    task_classes: list[int],
    training: bool,
    target_map_to_local: bool = True,
    expand_times: int = 1,
) -> TabularIncrementalDataset:
    split = "train" if training else "eval"
    return TabularIncrementalDataset(
        GVM.path_data_dict[split],
        task_classes,
        GVM.label_map_g2l,
        training=training,
        target_map_to_local=target_map_to_local,
        expand_times=expand_times,
    )
