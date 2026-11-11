import torch
import torch.nn as nn

def make_path(cin, cout, kernel, separable):
    if separable:
        return nn.Sequential(nn.Conv1d(cin, cin, kernel, padding=kernel // 2, groups=cin, bias=False), nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout), nn.ReLU())
    return nn.Sequential(nn.Conv1d(cin, cout, kernel, padding=kernel // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU())

class SharedBlock(nn.Module):

    def __init__(self, cin, cout, kernel, stride, separable, multiscale):
        super().__init__()
        self.down = nn.AvgPool1d(stride, stride=stride, ceil_mode=True)
        self.first = make_path(cin, cout, kernel, separable)
        self.second = make_path(cin, cout, kernel + 4, separable) if multiscale else nn.Identity()
        self.multiscale = multiscale

    def forward(self, x):
        downsampled = self.down(x)
        result = self.first(downsampled)
        if self.multiscale:
            result = (result + self.second(downsampled)) * 0.5
        return result

class Flicker_Net_Opt(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(SharedBlock(1, 32, 15, 2, True, True), SharedBlock(32, 32, 15, 2, True, True), SharedBlock(32, 32, 15, 2, True, True))
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.maximum = nn.AdaptiveMaxPool1d(1)
        self.hidden = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        main = x
        encoded = self.encoder(main)
        features = self.avg(encoded).flatten(1)
        features = torch.cat([features, self.maximum(encoded).flatten(1)], dim=1)
        features = self.hidden(features)
        return self.classifier(self.dropout(features))
