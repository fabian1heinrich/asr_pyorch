import torch.nn as nn

from model.residual_conv2d import ResConv2d
from model.permute_layer import PermuteLayer
from model.bidirectional_gru_block import BiDirectionalGRUBlock


class ASRModel2(nn.Module):
    def __init__(self):
        super(ASRModel2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ), ).to(0)

        self.res_conv = nn.Sequential(*[
            ResConv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
            ) for _ in range(3)
        ]).to(0)

        self.dense = nn.Sequential(
            PermuteLayer((0, 2, 1, 3)),
            nn.Flatten(start_dim=2),
            nn.Linear(2048, 512),
        ).to(0)

        self.gru = nn.Sequential(*[
            BiDirectionalGRUBlock(
                input_size=512 if i == 0 else 1024,
                hidden_size=512,
            ) for i in range(3)
        ]).to(1)

        self.classification = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 29),
            nn.LogSoftmax(dim=2),
        ).to(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.res_conv(x)
        x = self.dense(x)

        x = x.to(1)
        x = self.gru(x)
        x = self.classification(x)
        return x