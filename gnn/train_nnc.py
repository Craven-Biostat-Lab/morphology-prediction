"""Train an NNConv net given morphology data."""

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as func

from torch_geometric.nn import NNConv

from torcheval.metrics.functional import binary_auroc, binary_normalized_entropy

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

    parser.add_argument(
        '--epochs',
        type=int,
        default=1000
    )

    return parser

class NNConvNet(torch.nn.Module):
    """Our application of NNConv"""

    def __init__(
        self,
        in_features,
        edge_features,
        channel_width,
        encoder_depth=0,
        gnn_depth=1,
        decoder_depth=0,
        out_features=2,
        dropout_rate=0.5
    ):
        super().__init__()

        # Record network depth
        self.dropout_rate = dropout_rate

        # First layer: input x to channel width
        self.lin_in = torch.nn.Linear(in_features, channel_width, bias=True)

        # Encoder layers
        self.encoder = torch.nn.ModuleList(
            torch.nn.Linear(channel_width, channel_width, bias=True)
            for _ in range(encoder_depth)
        )

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
        self.decoder = torch.nn.ModuleList(
            torch.nn.Linear(channel_width, channel_width, bias=True)
            for _ in range(decoder_depth)
        )

        # Last layer: transform to result dimension.
        self.lin_out = torch.nn.Linear(channel_width, out_features, bias=True)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.lin_in(x)

        for encoder_layer in self.encoder:
            h = func.leaky_relu(h)
            h = func.dropout(h, p=self.dropout_rate)
            h = encoder_layer(h)

        for nnc_layer in self.nnconv:
            h = func.leaky_relu(h)
            h = func.dropout(h, p=self.dropout_rate)
            h = nnc_layer(h, edge_index, edge_attr)

        for decoder_layer in self.decoder:
            h = func.leaky_relu(h)
            h = func.dropout(h, p=self.dropout_rate)
            h = decoder_layer(h)
        
        return self.lin_out(h)

def training_loop(data, model, epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = model.to(device)

    # For organizing output
    subset_masks = [
        ('training', data.train_mask),
        ('testing', data.test_mask),
        ('validation', data.val_mask)
    ]

    metrics = [
        ('loss', binary_normalized_entropy),
        ('auroc', binary_auroc)
    ]

    best_auroc = 0
    best_parameters = {}
    best_epoch = -1

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)

    # Convert our column vector of 1 and -1 to 1, 0
    y_float = (data.y + 1) * 0.5
    # Get integer y for crose entropy function
    y = torch.tensor(y_float, dtype=torch.int64) 

    loss_curves = defaultdict(lambda: np.zeros(epochs, dtype=float))
    for epoch in range(epochs):
        # Training step
        model.train()
        # Reset gradient accumulator
        optimizer.zero_grad()
        # Forward pass
        out = model(data)
        training_loss = func.cross_entropy(out[data.train_mask], y[data.train_mask, 0])
        # Backprop
        training_loss.backward()
        # Step
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            y_hat = func.softmax(model(data), dim=1)
            # Metrics of interest
            for subset, mask in subset_masks:
                for metric, compute in metrics:
                    value = compute(y_hat[mask, 1], y_float[mask, 0]).numpy(force=True)
                    loss_curves[(subset, metric)][epoch] = value
                    if epoch % 10 == 9:
                        print(f"epoch {epoch}: {subset} {metric} = {value}")

        # Save best model
        auroc = loss_curves[('validation', 'auroc')][epoch]
        if auroc > best_auroc:
            best_auroc = auroc
            best_parameters = model.state_dict()
            best_epoch = epoch


    return model, loss_curves, best_parameters, best_epoch

def main(args):

    # Load data
    data = torch.load(args.data)

    hyperparameters = {
        'in_features': data.x.shape[1],
        'edge_features': data.edge_attr.shape[1],
        'channel_width': 10,
        'encoder_depth': 2,
        'gnn_depth': 2,
        'decoder_depth': 2
    }

    # Initialize model
    model = NNConvNet(**hyperparameters)

    # Run training loop
    model, loss_curves, best_parameters, best_epoch = training_loop(data, model, epochs=args.epochs)

    # Save loss curves
    args.loss_curve_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(loss_curves).to_csv(args.loss_curve_csv)

    # Save model
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'hyperparameters': hyperparameters,
            'parameters': best_parameters,
            'best_epoch': best_epoch
        },
        args.model_path
    )

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)