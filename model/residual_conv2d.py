import torch.nn as nn


class ResConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResConv2d, self).__init__()

        self.res_conv2d = nn.Sequential(
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
        )

    def forward(self, x):
        res = x
        x = self.res_conv2d(x)
        x += res
        return x
