import argparse
import glob
import os

import numpy as np
import pandas as pd

SPLITS = {
    "train": ["S1-ADL1", "S1-ADL2", "S1-ADL3", "S1-ADL4", "S1-ADL5", "S1-Drill",
              "S2-ADL1", "S2-ADL2", "S2-Drill",
              "S3-ADL1", "S3-ADL2", "S3-Drill"],
    "val": ["S2-ADL3", "S3-ADL3"],
    "test": ["S2-ADL4", "S2-ADL5", "S3-ADL4", "S3-ADL5"],
}

# 0-indexed columns of the 250-column .dat files that are NOT used:
# IMU quaternions, object/ambient sensors, and the other label tracks.
_DROP = np.concatenate([
    np.arange(46, 50), np.arange(59, 63), np.arange(72, 76),
    np.arange(85, 89), np.arange(98, 102),
    np.arange(134, 243), np.arange(244, 249),
])

LOCOMOTION_MAP = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4}
GESTURE_MAP = {0: 0,
               406516: 1, 406517: 2, 404516: 3, 404517: 4, 406520: 5, 404520: 6,
               406505: 7, 404505: 8, 406519: 9, 404519: 10, 406511: 11, 404511: 12,
               406508: 13, 404508: 14, 408512: 15, 407521: 16, 405506: 17}


def find_file(raw_dir, name):
    hits = glob.glob(os.path.join(raw_dir, "**", f"{name}.dat"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name}.dat not found under {raw_dir}. Did you run download_data.py?")
    return hits[0]


def load_recording(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, na_values=["NaN"], engine="c")
    if df.shape[1] != 250:
        raise ValueError(f"{path}: expected 250 columns, found {df.shape[1]}")
    data = np.delete(df.to_numpy(dtype=np.float64), _DROP, axis=1)
    x = pd.DataFrame(data[:, 1:114])
    x = x.interpolate(method="linear", axis=0, limit_direction="both").fillna(0.0)
    y_loco = np.vectorize(LOCOMOTION_MAP.get)(data[:, 114].astype(int))
    y_gest = np.vectorize(GESTURE_MAP.get)(data[:, 115].astype(int))
    return x.to_numpy(np.float32), y_loco.astype(np.int64), y_gest.astype(np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/opportunity.npz")
    args = p.parse_args()

    out = {}
    recs = {}
    for split, names in SPLITS.items():
        recs[split] = []
        for n in names:
            path = find_file(args.raw, n)
            x, yl, yg = load_recording(path)
            print(f"{split:5s} {n:9s} {x.shape[0]:7d} samples")
            recs[split].append((x, yl, yg))

    train_x = np.concatenate([r[0] for r in recs["train"]])
    lo, hi = train_x.min(0), train_x.max(0)
    rng = np.where(hi - lo > 1e-8, hi - lo, 1.0)

    for split, rs in recs.items():
        xs = [np.clip((x - lo) / rng, 0.0, 1.0).astype(np.float32) for x, _, _ in rs]
        out[f"{split}_X"] = np.concatenate(xs)
        out[f"{split}_y_locomotion"] = np.concatenate([r[1] for r in rs])
        out[f"{split}_y_gestures"] = np.concatenate([r[2] for r in rs])
        out[f"{split}_lengths"] = np.array([len(x) for x in xs])  # windows never cross recordings
    out["norm_min"], out["norm_max"] = lo, hi

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"Saved {args.out}")
    for s in SPLITS:
        print(f"  {s}: {out[s + '_X'].shape}")


if __name__ == "__main__":
    main()
