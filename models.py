import torch
import torch.nn as nn


def conv_out_len(window, n_conv=4, kernel=5):
    return window - n_conv * (kernel - 1)


class ConvFeatures(nn.Module):
    def __init__(self, n_filters=64, kernel=5, n_conv=4):
        super().__init__()
        layers, in_ch = [], 1
        for _ in range(n_conv):
            layers += [nn.Conv2d(in_ch, n_filters, kernel_size=(kernel, 1)), nn.ReLU()]
            in_ch = n_filters
        self.net = nn.Sequential(*layers)
        self.out_channels = n_filters if n_conv > 0 else 1

    def forward(self, x):                 # x: (B, T, D)
        return self.net(x.unsqueeze(1))   # (B, F, T', D)


class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels, n_classes, window=24, n_filters=64, kernel=5, n_conv=4,
                 rnn="lstm", rnn_units=128, rnn_layers=2, bidirectional=False, dropout=0.5):
        super().__init__()
        if conv_out_len(window, n_conv, kernel) < 1:
            raise ValueError(f"window={window} too short for {n_conv} conv layers of kernel {kernel}")
        self.conv = ConvFeatures(n_filters, kernel, n_conv)
        self.drop = nn.Dropout(dropout)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn]
        self.rnn = rnn_cls(self.conv.out_channels * n_channels, rnn_units, num_layers=rnn_layers,
                           batch_first=True, bidirectional=bidirectional,
                           dropout=dropout if rnn_layers > 1 else 0.0)
        self.bidirectional, self.units = bidirectional, rnn_units
        self.fc = nn.Linear(rnn_units * (2 if bidirectional else 1), n_classes)
        init_orthogonal(self)

    def forward(self, x):
        f = self.conv(x)                           # (B, F, T', D)
        f = f.permute(0, 2, 1, 3).flatten(2)        # (B, T', F*D): all feature maps at time t
        out, _ = self.rnn(self.drop(f))             # (B, T', H or 2H)
        if self.bidirectional:                      # forward dir. ends at T, backward dir. ends at 0
            h = torch.cat([out[:, -1, :self.units], out[:, 0, self.units:]], dim=1)
        else:
            h = out[:, -1]                          # prediction at the last time step (Sec. 4.2)
        return self.fc(self.drop(h))


class BaselineCNN(nn.Module):
    def __init__(self, n_channels, n_classes, window=24, n_filters=64, kernel=5, n_conv=4,
                 dense_units=128, dropout=0.5, **_):
        super().__init__()
        t_out = conv_out_len(window, n_conv, kernel)
        if t_out < 1:
            raise ValueError(f"window={window} too short for {n_conv} conv layers of kernel {kernel}")
        self.conv = ConvFeatures(n_filters, kernel, n_conv)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(self.conv.out_channels * t_out * n_channels, dense_units), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dense_units, dense_units), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dense_units, n_classes),
        )
        init_orthogonal(self)

    def forward(self, x):
        return self.head(self.conv(x))


def init_orthogonal(model):
    for name, p in model.named_parameters():
        if "bias" in name:
            nn.init.zeros_(p)
        elif p.dim() >= 2:
            nn.init.orthogonal_(p)


def build_model(name, n_channels, n_classes, window, **kw):
    """kw: n_conv, n_filters, kernel, dropout, rnn, rnn_units, rnn_layers, bidirectional"""
    if name == "deepconvlstm":
        return DeepConvLSTM(n_channels, n_classes, window=window, **kw)
    if name == "cnn":
        return BaselineCNN(n_channels, n_classes, window=window, **kw)
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
