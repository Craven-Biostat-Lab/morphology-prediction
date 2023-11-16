"""Train an NNConv net given morphology data."""

from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as func

from torch_geometric.nn import NNConv

# May use ignite in the future
#from ignite.engine import Engine, Events
#from ignite.metrics import MeanSquaredError
#from ignite.contrib.metrics.regression import R2Score

def create_parser():
    """Command line argument parser."""
    
    from argparse import ArgumentParser
    parser = ArgumentParser('NNConv net training')
    parser.add_argument(
        '--data',
        type=Path
    )
    parser.add_argument(
        '--model-path',
        type=Path
    )
    parser.add_argument(
        '--loss-curve-csv',
        type=Path
    )

    return parser

class NNConvNet(torch.nn.Module):
    """Our application of NNConv"""

    def __init__(
        self,
        in_features,
        edge_features,
        channel_width,
        #encoder_depth=1,
        gnn_depth=1,
        #decoder_depth=1,
        out_features=1,
        dropout_rate=0.5
    ):
        super().__init__()

        # Record network depth
        self.dropout_rate = dropout_rate

        # First layer: input x to channel width
        self.lin_in = torch.nn.Linear(in_features, channel_width, bias=True)

        # Middle layers: NNConv layers
        # NN for NNConv is shared across layers
        self.edge_nn = torch.nn.Sequential(
            torch.nn.Linear(edge_features, channel_width**2, bias=True),
            torch.nn.LeakyReLU()
        )

        # The NNConv layers
        self.nnconv = [
            NNConv(
                in_channels=channel_width,
                out_channels=channel_width,
                nn=self.edge_nn,
                aggr='add',
                root_weight=True,
                bias=True
            )
            for _ in range(gnn_depth)
        ]

        # Last layer: transform to result dimension.
        self.lin_out = torch.nn.Linear(channel_width, out_features, bias=True)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = func.leaky_relu(self.lin_in(x))
        h = func.dropout(h, p=self.dropout_rate)

        for nnc_layer in self.nnconv:
            h = nnc_layer(h, edge_index, edge_attr)
            h = func.leaky_relu(h)
            h = func.dropout(h, p=self.dropout_rate)
        
        return self.lin_out(h)

def training_loop(data, model, epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.to(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)

    # Convert our column vector of 1 and -1 to 1, 0
    y = (data.y + 1) * 0.5

    model.train()
    loss_curves = np.zeros((epochs, 2), dtype=float)
    for epoch in range(epochs):
        # Reset gradient accumulator
        optimizer.zero_grad()
        # Forward pass
        out = model(data)
        training_loss = func.cross_entropy(out[data.train_mask, 0], y[data.train_mask, 0])
        test_loss = func.cross_entropy(out[data.test_mask, 0], y[data.test_mask, 0])
        # Record loss curves
        loss_curves[epoch, 0] = training_loss.numpy(force=True)
        loss_curves[epoch, 1] = test_loss.numpy(force=True)
        if epoch % 10 == 9:
            print(f"epoch {epoch}, train, test losses {loss_curves[epoch]}")
        # Backprop
        training_loss.backward()
        # Step
        optimizer.step()

    return model, loss_curves

def main(args):

    # Load data
    data = torch.load(args.data)

    # Initialize model
    model = NNConvNet(
        in_features=data.x.shape[1],
        edge_features=data.edge_attr.shape[1],
        channel_width=10
    )

    # Run training loop
    model, loss_curves = training_loop(data, model)

    # Save loss curves
    args.loss_curve_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(loss_curves, columns=['training', 'testing']).to_csv(args.loss_curve_csv)

    # Save model
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.model_path)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)