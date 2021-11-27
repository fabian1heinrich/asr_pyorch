import torch.nn as nn


class BiDirectionalGRUBlock(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        batch_first=True,
    ):
        super(BiDirectionalGRUBlock, self).__init__()

        self.layer_norm = nn.LayerNorm(input_size)
        self.bidirectional_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.layer_norm(x)
        x, _ = self.bidirectional_gru(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
