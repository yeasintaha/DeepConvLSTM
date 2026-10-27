import glob
import json
import os
import re

import pandas as pd

rows = []
for f in sorted(glob.glob("runs/*/metrics.json")):
    m = json.load(open(f))
    if "test_f1_weighted" not in m:  # unfinished or incomplete run
        continue
    c = {"n_conv": 4, **m["config"]}
    run = os.path.basename(os.path.dirname(f))
    rows.append({"run": run, "config": re.sub(r"_s\d+$", "", run), "model": c["model"],
                 "task": c["task"], "null": not c["no_null"], "sensors": c["sensors"],
                 "window": c["window"], "n_conv": c["n_conv"], "params": m["params"],
                 "best_epoch": m["best_epoch"], "train_min": round(m.get("train_minutes", float("nan")), 1),
                 "test_F1_weighted": m["test_f1_weighted"], "test_F1_macro": m["test_f1_macro"],
                 "test_acc": m["test_accuracy"]})
df = pd.DataFrame(rows)
if not len(df):
    print("no runs found")
    raise SystemExit
df.round(4).to_csv("results_summary.csv", index=False)
print(df.round(4).to_string(index=False))

g = df.groupby("config").agg(n=("run", "count"), params=("params", "first"),
                              F1w_mean=("test_F1_weighted", "mean"), F1w_std=("test_F1_weighted", "std"),
                              F1m_mean=("test_F1_macro", "mean"), F1m_std=("test_F1_macro", "std"),
                              train_min=("train_min", "mean")).round(4)
g.to_csv("results_by_config.csv")
print("\nMean over seeds:\n" + g.to_string())
