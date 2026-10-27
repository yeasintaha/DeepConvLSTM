import numpy as np
import torch
from torch.utils.data import Dataset

LOCOMOTION_CLASSES = ["Null", "Stand", "Walk", "Sit", "Lie"]
GESTURE_CLASSES = ["Null", "Open Door 1", "Open Door 2", "Close Door 1", "Close Door 2",
                   "Open Fridge", "Close Fridge", "Open Dishwasher", "Close Dishwasher",
                   "Open Drawer 1", "Close Drawer 1", "Open Drawer 2", "Close Drawer 2",
                   "Open Drawer 3", "Close Drawer 3", "Clean Table", "Drink from Cup",
                   "Toggle Switch"]


def _imu(kind):
    """Channel indices of the 5 jacket IMUs (BACK, RUA, RLA, LUA, LLA).
    Among the 113 channels, IMU k starts at 36 + 9k with acc(3), gyro(3), mag(3)."""
    off = {"acc": 0, "gyro": 3, "mag": 6}[kind]
    return [36 + 9 * k + off + i for k in range(5) for i in range(3)]


# Sensor subsets of paper Table 10 (all use the 5 jacket IMUs) plus the full set.
SENSOR_SETS = {
    "all": list(range(113)),                                   # 113 channels
    "acc": _imu("acc"),                                         # 15
    "gyro": _imu("gyro"),                                       # 15
    "acc_gyro": sorted(_imu("acc") + _imu("gyro")),             # 30
    "acc_gyro_mag": sorted(_imu("acc") + _imu("gyro") + _imu("mag")),  # 45
}


def class_names(task, include_null=True):
    names = LOCOMOTION_CLASSES if task == "locomotion" else GESTURE_CLASSES
    return names if include_null else names[1:]


def make_windows(X, y, lengths, window, step):
    """Cut each recording into windows of `window` samples every `step` samples.
    The label of a window is the label of its LAST sample (paper Fig. 5)."""
    xs, ys = [], []
    start = 0
    for n in lengths:
        xr, yr = X[start:start + n], y[start:start + n]
        start += n
        if n < window:
            continue
        idx = np.arange(0, n - window + 1, step)
        xs.append(np.stack([xr[i:i + window] for i in idx]))
        ys.append(yr[idx + window - 1])
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64)


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # (N, T, D)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_split(npz, split, task, sensors="all", window=24, step=12, include_null=True):
    X = npz[f"{split}_X"][:, SENSOR_SETS[sensors]]
    y = npz[f"{split}_y_{task}"]
    Xw, yw = make_windows(X, y, npz[f"{split}_lengths"], window, step)
    if not include_null:  # "No Null class" setting of Tables 4, 7, 8
        keep = yw != 0
        Xw, yw = Xw[keep], yw[keep] - 1
    return WindowDataset(Xw, yw)
