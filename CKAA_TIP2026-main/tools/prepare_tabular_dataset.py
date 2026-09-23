import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tabular_data import build_tabular_dataset


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert CKAA vibration and pressure XYZ CSV streams to tabular windows."
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=str(ROOT.parent),
        help="Directory containing data_1_S1 and data_2_S1.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(ROOT / "A_CLData" / "tabular_ckaa"),
    )
    parser.add_argument("--window-length", type=int, default=1568)
    parser.add_argument("--stride", type=int, default=1568)
    parser.add_argument("--max-common-samples", type=int, default=614400)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    meta = build_tabular_dataset(
        raw_root=args.raw_root,
        output_root=args.output_root,
        window_length=args.window_length,
        stride=args.stride,
        max_common_samples=args.max_common_samples,
        train_ratio=args.train_ratio,
        overwrite=args.overwrite,
    )
    print(f"Processed dataset: {args.output_root}")
    for split in ("train", "eval"):
        print(f"{split}: {meta['files'][split]['num_samples']} windows")
    print(f"labels: {meta['labels']}")


if __name__ == "__main__":
    main()
