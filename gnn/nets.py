"""Network architectures are collected here"""

import torch
import torch.nn.functional as func
from torch_geometric.nn import NNConv

class NNConvNet(torch.nn.Module):
    """Our application of NNConv"""

    tunable_hyperparameters = (
        'channel_width',
        'encoder_depth',
        'gnn_depth',
        'decoder_depth',
        'out_features',
        'dropout_rate'
    )

    def __init__(
        self,
        in_features,
        edge_features,
        channel_width,
        encoder_depth=0,
        gnn_depth=1,
        decoder_depth=0,
        out_features=1,
        dropout_rate=0.5
    ):
        super().__init__()

        # Record network depth
        self.dropout_rate = dropout_rate

        # First layer: input x to channel width
        self.lin_in = torch.nn.Linear(in_features, channel_width, bias=True)

        # Encoder layers
        self.encoder = torch.nn.Sequential(*(
            torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=dropout_rate),
                torch.nn.Linear(channel_width, channel_width, bias=True)
            )
            for _ in range(encoder_depth)
        ))

        # Middle layers: NNConv layers
        # NN for NNConv is shared across layers
        self.edge_nn = torch.nn.Sequential(
            torch.nn.Linear(edge_features, channel_width**2, bias=True),
            torch.nn.LeakyReLU()
        )

        # The NNConv layers
        self.nnconv = torch.nn.ModuleList(
            NNConv(
                in_channels=channel_width,
                out_channels=channel_width,
                nn=self.edge_nn,
                aggr='add',
                root_weight=True,
                bias=True
            )
            for _ in range(gnn_depth)
        )

        # Decoder layers
        self.decoder = torch.nn.Sequential(*(
            torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p=dropout_rate),
                torch.nn.Linear(channel_width, channel_width, bias=True)
            )
            for _ in range(decoder_depth)
        ))

        # Last layer: transform to result dimension.
        self.lin_out = torch.nn.Linear(channel_width, out_features, bias=True)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.lin_in(x)
        h = self.encoder(h)

        for nnc_layer in self.nnconv:
            h = func.leaky_relu(h)
            h = func.dropout(h, p=self.dropout_rate)
            h = nnc_layer(h, edge_index, edge_attr)

        h = self.decoder(h)

        out = torch.squeeze(self.lin_out(h), -1)

        return func.log_softmax(
            torch.stack([torch.zeros_like(out), out], dim=1),
            dim=1)
