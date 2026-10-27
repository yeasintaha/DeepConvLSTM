import argparse
import json
import os

import numpy as np
import torch

from data import SENSOR_SETS, class_names, load_split
from models import build_model
from train import get_device, model_kwargs, predict, save_plots, write_test_results

# Defaults for options that did not exist when older runs were trained.
_DEFAULTS = {"filters": 64, "kernel": 5, "rnn": "lstm", "rnn_units": 128, "rnn_layers": 2,
             "bidirectional": False, "dropout": 0.5, "n_conv": 4}


def load_trained(run_dir, device="auto"):
    device = get_device(device)
    ckpt = torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state, cfg = ckpt["state_dict"], ckpt["config"]
    else:  # old format: plain state_dict
        state = ckpt
        with open(os.path.join(run_dir, "metrics.json")) as f:
            cfg = json.load(f)["config"]
    cfg = {**_DEFAULTS, **cfg}
    names = class_names(cfg["task"], include_null=not cfg["no_null"])
    n_ch = len(SENSOR_SETS[cfg["sensors"]])
    model = build_model(cfg["model"], n_ch, len(names), cfg["window"], **model_kwargs(cfg))
    model.load_state_dict(state)
    return model.to(device).eval(), cfg, names


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run folder containing best_model.pt")
    p.add_argument("--data", default=None, help="preprocessed .npz (default: the one used in training)")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--save-predictions", action="store_true", help="write predictions_<split>.csv")
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    model, cfg, names = load_trained(a.run, a.device)
    device = next(model.parameters()).device
    npz = np.load(a.data or cfg["data"])
    ds = load_split(npz, a.split, task=cfg["task"], sensors=cfg["sensors"], window=cfg["window"],
                    step=cfg["step"], include_null=not cfg["no_null"])
    y, pred, _ = predict(model, ds.X.to(device), ds.y.to(device))
    s, cm, report = write_test_results(a.run, y, pred, names, prefix=f"eval_{a.split}_")
    print(report)
    print(f"{a.split.upper()}  weighted F1 = {s['f1_weighted']:.4f}   macro F1 = {s['f1_macro']:.4f}   "
          f"accuracy = {s['accuracy']:.4f}")
    if a.save_predictions:
        import pandas as pd
        pd.DataFrame({"true": [names[i] for i in y], "pred": [names[i] for i in pred]}).to_csv(
            os.path.join(a.run, f"predictions_{a.split}.csv"), index_label="window")
        print(f"predictions written to {a.run}/predictions_{a.split}.csv")


if __name__ == "__main__":
    main()
