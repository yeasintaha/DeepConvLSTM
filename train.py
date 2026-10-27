import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from data import SENSOR_SETS, class_names, load_split
from models import build_model, count_params


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--data", default="data/opportunity.npz")
    g.add_argument("--task", choices=["gestures", "locomotion"], default="gestures")
    g.add_argument("--no-null", action="store_true", help="drop Null-class windows (Table 4 'No Null')")
    g.add_argument("--sensors", choices=list(SENSOR_SETS), default="all")
    g.add_argument("--window", type=int, default=24, help="window length in samples (30 Hz)")
    g.add_argument("--step", type=int, default=12, help="window step in samples")
    g = p.add_argument_group("model (defaults = paper)")
    g.add_argument("--model", choices=["deepconvlstm", "cnn"], default="deepconvlstm")
    g.add_argument("--n-conv", type=int, default=4, help="number of conv layers (0 = no conv)")
    g.add_argument("--filters", type=int, default=64)
    g.add_argument("--kernel", type=int, default=5)
    g.add_argument("--rnn", choices=["lstm", "gru"], default="lstm")
    g.add_argument("--rnn-units", type=int, default=128)
    g.add_argument("--rnn-layers", type=int, default=2)
    g.add_argument("--bidirectional", action="store_true")
    g.add_argument("--dropout", type=float, default=0.5)
    g = p.add_argument_group("training (defaults = paper)")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--patience", type=int, default=20, help="early stopping on val weighted F1 (0 = off)")
    g.add_argument("--batch-size", type=int, default=100)
    g.add_argument("--optimizer", choices=["rmsprop", "adam"], default="rmsprop")
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--rho", type=float, default=0.9, help="RMSProp decay factor")
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--cosine", action="store_true", help="cosine learning-rate schedule")
    g.add_argument("--class-weights", choices=["none", "inverse", "sqrt_inverse"], default="none")
    g.add_argument("--label-smoothing", type=float, default=0.0)
    g.add_argument("--augment", action="store_true", help="random jitter + per-channel scaling")
    g.add_argument("--warmup-epochs", type=int, default=0,
                   help="number of initial epochs for linear learning-rate warm-up")
    g.add_argument("--loss", choices=["ce", "focal"], default="ce",
                   help="loss function: ce (cross-entropy) or focal (focal loss)")
    g.add_argument("--focal-gamma", type=float, default=2.0,
                   help="focusing parameter gamma for focal loss")
    g.add_argument("--select-on", choices=["f1_weighted", "f1_macro"], default="f1_weighted",
                   help="validation metric used to pick the best epoch")
    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="auto")
    g.add_argument("--name", default=None, help="run name (folder under runs/)")
    g.add_argument("--out", default=None, help="full output folder (overrides --name)")
    return p.parse_args(argv)


MODEL_KEYS = ["n_conv", "filters", "kernel", "rnn", "rnn_units", "rnn_layers", "bidirectional", "dropout"]


def model_kwargs(cfg):
    return dict(n_conv=cfg["n_conv"], n_filters=cfg["filters"], kernel=cfg["kernel"], rnn=cfg["rnn"],
                rnn_units=cfg["rnn_units"], rnn_layers=cfg["rnn_layers"],
                bidirectional=cfg["bidirectional"], dropout=cfg["dropout"])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, X, y, batch_size=1000):
    """X, y: tensors already on the model's device. Returns numpy labels, predictions, mean loss."""
    model.eval()
    preds, loss_sum = [], 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for i in range(0, len(y), batch_size):
        logits = model(X[i:i + batch_size])
        loss_sum += crit(logits, y[i:i + batch_size]).item()
        preds.append(logits.argmax(1))
    return y.cpu().numpy(), torch.cat(preds).cpu().numpy(), loss_sum / len(y)


def scores(y, p):
    # Weighted F1 = paper Eq. (11): per-class F1 weighted by class proportion.
    return {"f1_weighted": f1_score(y, p, average="weighted", zero_division=0),
            "f1_macro": f1_score(y, p, average="macro", zero_division=0),
            "accuracy": accuracy_score(y, p)}


def augment(x):
    """Simple sensor augmentation on [0,1]-normalized windows (B, T, D)."""
    scale = 1.0 + 0.1 * torch.randn(x.shape[0], 1, x.shape[2], device=x.device)
    return x * scale + 0.01 * torch.randn_like(x)


def class_weight_tensor(y, n_classes, mode, device):
    if mode == "none":
        return None
    counts = torch.bincount(y.cpu(), minlength=n_classes).float().clamp(min=1)
    w = counts.sum() / (n_classes * counts)
    if mode == "sqrt_inverse":
        w = w.sqrt()
    return (w / w.mean()).to(device)


### Focal Loss has been added
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


def save_plots(out, hist, cm, names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if hist is not None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(hist["epoch"], hist["train_loss"], label="train")
        ax[0].plot(hist["epoch"], hist["val_loss"], label="val")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy"); ax[0].legend()
        ax[1].plot(hist["epoch"], hist["val_f1_weighted"], label="val weighted F1")
        ax[1].plot(hist["epoch"], hist["val_f1_macro"], label="val macro F1")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("F1"); ax[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(out, "learning_curves.png"), dpi=120); plt.close(fig)

    n = len(names)
    fig, ax = plt.subplots(figsize=(0.55 * n + 3, 0.55 * n + 2))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(out, "confusion_matrix.png"), dpi=120); plt.close(fig)


def write_test_results(out, y, p, names, prefix=""):
    s = scores(y, p)
    labels = list(range(len(names)))
    cm = confusion_matrix(y, p, labels=labels)
    report = classification_report(y, p, labels=labels, target_names=names, zero_division=0, digits=3)
    pd.DataFrame(cm, index=names, columns=names).to_csv(os.path.join(out, f"{prefix}confusion_matrix.csv"))
    with open(os.path.join(out, f"{prefix}classification_report.txt"), "w") as f:
        f.write(report)
    return s, cm, report


def main(argv=None):
    a = parse_args(argv)
    cfg = vars(a)
    set_seed(a.seed)
    device = get_device(a.device)
    tag = a.name or (f"{a.model}_{a.task}{'_nonull' if a.no_null else ''}_{a.sensors}"
                     f"_w{a.window}_c{a.n_conv}_s{a.seed}")
    out = a.out or os.path.join("runs", tag)
    os.makedirs(out, exist_ok=True)

    npz = np.load(a.data)
    kw = dict(task=a.task, sensors=a.sensors, window=a.window, step=a.step, include_null=not a.no_null)
    train_ds, val_ds, test_ds = (load_split(npz, s, **kw) for s in ("train", "val", "test"))
    names = class_names(a.task, include_null=not a.no_null)
    n_ch = train_ds.X.shape[2]
    print(f"[{tag}] device={device}  windows train/val/test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
          f"  channels={n_ch}  classes={len(names)}")

    # The whole dataset is small (~600 MB), so keep it on the GPU and slice batches directly.
    # This is much faster than a DataLoader that builds every batch window by window.
    Xtr, ytr = train_ds.X.to(device), train_ds.y.to(device)
    Xva, yva = val_ds.X.to(device), val_ds.y.to(device)
    Xte, yte = test_ds.X.to(device), test_ds.y.to(device)

    model = build_model(a.model, n_ch, len(names), a.window, **model_kwargs(cfg)).to(device)
    print(f"parameters: {count_params(model):,}")
    if a.optimizer == "rmsprop":
        opt = torch.optim.RMSprop(model.parameters(), lr=a.lr, alpha=a.rho, weight_decay=a.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    
    if a.warmup_epochs > 0 and a.cosine:
        w_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, end_factor=1.0, total_iters=a.warmup_epochs
        )
        c_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs - a.warmup_epochs, 1))
        sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[w_sched, c_sched], milestones=[a.warmup_epochs])
    elif a.warmup_epochs > 0:
        sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, end_factor=1.0, total_iters=a.warmup_epochs
        )
    elif a.cosine:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    else:
        sched = None

    weights = class_weight_tensor(ytr, len(names), a.class_weights, device)
    if a.loss == "focal":
        crit = FocalLoss(gamma=a.focal_gamma, weight=weights)
    else:
        crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=a.label_smoothing)

    ckpt_path = os.path.join(out, "best_model.pt")
    best, best_epoch, hist = -1.0, 0, []
    best_state_dict = None
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        perm = torch.randperm(len(ytr), device=device)
        for i in range(0, len(perm), a.batch_size):
            idx = perm[i:i + a.batch_size]
            x, y = Xtr[idx], ytr[idx]
            if a.augment:
                x = augment(x)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        if sched:
            sched.step()
        yv, pv, vloss = predict(model, Xva, yva)
        s = scores(yv, pv)
        hist.append({"epoch": epoch, "train_loss": tot / len(ytr), "val_loss": vloss,
                     **{f"val_{k}": v for k, v in s.items()}})
        flag = ""
        if s[a.select_on] > best:
            best, best_epoch, flag = s[a.select_on], epoch, " *"
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            os.makedirs(out, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "config": cfg, "class_names": names,
                        "n_channels": n_ch, "epoch": epoch, "val_scores": s}, ckpt_path)
        print(f"epoch {epoch:3d}  train_loss {tot / len(ytr):.4f}  val_loss {vloss:.4f}  "
              f"val_F1w {s['f1_weighted']:.4f}  val_F1m {s['f1_macro']:.4f}  "
              f"({time.time() - t0:.0f}s){flag}", flush=True)
        if a.patience and epoch - best_epoch >= a.patience:
            print(f"early stopping (no val improvement for {a.patience} epochs)", flush=True)
            break
    train_min = (time.time() - t0) / 60

    os.makedirs(out, exist_ok=True)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    elif best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        torch.save({"state_dict": best_state_dict, "config": cfg, "class_names": names,
                    "n_channels": n_ch, "epoch": best_epoch, "val_scores": s}, ckpt_path)
    t1 = time.time()
    yt, pt, _ = predict(model, Xte, yte)
    test_s, cm, report = write_test_results(out, yt, pt, names)
    print(report)
    print(f"TEST  weighted F1 = {test_s['f1_weighted']:.4f}   macro F1 = {test_s['f1_macro']:.4f}   "
          f"accuracy = {test_s['accuracy']:.4f}   (best epoch {best_epoch}, {train_min:.1f} min training)")

    hist = pd.DataFrame(hist)
    hist.to_csv(os.path.join(out, "history.csv"), index=False)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"config": cfg, "params": count_params(model), "best_epoch": best_epoch,
                   "best_val_" + a.select_on: best, "train_minutes": train_min,
                   "test_seconds": time.time() - t1,
                   **{f"test_{k}": v for k, v in test_s.items()}}, f, indent=2)
    save_plots(out, hist, cm, names)
    print(f"results saved to {out}/")


if __name__ == "__main__":
    main()
