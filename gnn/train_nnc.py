"""Train an NNConv net given morphology data."""

from pathlib import Path
from collections import defaultdict

import json

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as func

from torcheval.metrics.functional import binary_auroc, binary_normalized_entropy

from nets import NNConvNet

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
        '--best-metrics',
        type=Path
    )
    parser.add_argument(
        '--hyperparameters',
        type=Path
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=1000
    )

    return parser


def training_loop(data, model, epochs=200, lr=0.1, weight_decay=5e-4):
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Convert our column vector of 1 and -1 to 1, 0
    y_float = (data.y + 1) * 0.5
    # Get integer y for crose entropy function
    y = torch.tensor(y_float, dtype=torch.int64)[:,0]

    # Count labels for balanced batching
    with torch.no_grad():
        values = torch.bincount(y)
        n_classes = values.shape[0]
        minority_size = values.min().item()
        class_inds = [
            (data.train_mask & (y == c)).nonzero()[:,0]
            for c in range(n_classes)
        ]
        # Trick to get cieling of division with integer divide
        n_batches = -(-data.train_mask.sum().item() // minority_size)

    loss_curves = defaultdict(lambda: np.zeros(epochs, dtype=float))
    for epoch in range(epochs):
        # Training step
        model.train()

        # Permute class-specific node indeces
        permuted_class_inds = [
            inds[torch.randperm(inds.shape[0])]
            for inds in class_inds
        ]

        for batch in range(n_batches):
            # Build batch
            i_start = batch * minority_size
            i_end = i_start + minority_size
            batch_inds = torch.cat([
                ind[torch.range(i_start, i_end, dtype=torch.int64) % ind.shape[0]]
                for ind in permuted_class_inds
            ])            
            # Reset gradient accumulator
            optimizer.zero_grad()
            # Forward pass (loop)
            out = model(data)
            training_loss = func.nll_loss(out[batch_inds], y[batch_inds])
            # Backprop
            training_loss.backward()
            # Step
            optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            y_hat = torch.exp(model(data))
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


    return model, loss_curves, best_parameters, best_epoch, best_auroc

def main(args):

    # Load data
    data = torch.load(args.data)

    # Read hyperparameters
    with args.hyperparameters as in_handle:
        input_hyperparameters = json.load(in_handle)

    hyperparameters = {
        'in_features': data.x.shape[1],
        'edge_features': data.edge_attr.shape[1],
    }
    hyperparameters.update({
        k: v
        for k, v in input_hyperparameters
        if k in NNConvNet.tunable_hyperparameters
    })

    learning_rate = input_hyperparameters.get('learning_rate', 0.1)
    weight_decay = input_hyperparameters.get('weight_decay', 5e-4)

    # Initialize model
    model = NNConvNet(**hyperparameters)

    # Run training loop
    model, loss_curves, best_parameters, best_epoch, best_auroc = training_loop(
        data,
        model,
        epochs=args.epochs,
        lr=learning_rate,
        weight_decay=weight_decay
    )

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

    with args.metric_path.open('w') as out_handle:
        json.dump({'best_auroc': best_auroc}, out_handle)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)