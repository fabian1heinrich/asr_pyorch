import torch.nn as nn


class PermuteLayer(nn.Module):
    def __init__(self, permutation):
        super(PermuteLayer, self).__init__()

        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)
