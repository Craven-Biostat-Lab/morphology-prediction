"""Download CPG datasets"""

from argparse import ArgumentParser
import logging

import numpy as np
import pandas as pd
import pycytominer

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'log level': logging.DEBUG,
    'dataset': 'cpg0003-A549',
    'cpg0003 uri template': (
        "s3://cellpainting-gallery/cpg0003-rosetta/"
        "broad/workspace/preprocessed_data/"
        "{dataset}/CellPainting/"
        "replicate_level_cp_{processing}.csv.gz"
    ),
    'datasets': ["LUAD-BBBC041-Caicedo", "TA-ORF-BBBC037-Rohban"]
}

KNOWN_DATASETS = {
    'cpg0003-A549': {
        'dataset uri': (
            "s3://cellpainting-gallery/cpg0003-rosetta/"
            "broad/workspace/preprocessed_data/"
            "LUAD-BBBC041-Caicedo/CellPainting/"
            "replicate_level_cp_augmented.csv.gz"
        ),
        'add to metadata cols': ['Symbol']
    },
    'cpg0003-U2OS': {
        'dataset uri': (
            "s3://cellpainting-gallery/cpg0003-rosetta/"
            "broad/workspace/preprocessed_data/"
            "TA-ORF-BBBC037-Rohban/CellPainting/"
            "replicate_level_cp_augmented.csv.gz"
        )
    },
    #'cpg0016': None
}


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()
    return parser


def get_config(args):
    config = {} | DEFAULT_CONFIG
    if not config.get('dataset uri'):
        config['dataset uri'].update(KNOWN_DATASETS[config['dataset']])
    return config


def main(config):
    
    # Load profiles
    profiles = pd.read_csv(
        config['dataset uri'],
        storage_options={'anon': True},
        dtype=config['explicit dtypes']
    )

    columns = profiles.columns

    # Identify profile columns
    profile_cols = columns[
        columns.str.startswith('Nuclei_') |
        columns.str.startswith('Cells_') |
        columns.str.startswith('Cytoplasm_')
    ]

    # Rename columns if needed
    columns_to_add = config.get('add to metadata cols')
    if columns_to_add:
        profiles.rename({c: f'Metadata_{c}' for c in columns_to_add}, inplace=True)
        columns = profiles.columns

    # Identify metadata columns
    meta_cols = columns[columns.str.startswith('Metadata_')]

    # Identify controls

    # Normalize
    normalized_profiles = profiles.set_index(meta_cols).groupby('Metadata_Plate', group_keys=False).apply(
        pycytominer.normalize,
        # profiles is first positional argument
        # features=list(feature_cols),
        # image_features=True, # Include Image_### features
        method="standardize",
        samples="Metadata_pert_type == 'control'"
    )

    # Write normalized profiles to file
    normalized_profiles.to_parquet(config['out_path'])


if __name__ == '__main__':
    parser = create_parser()
    config = get_config(parser.parse_args())
    logging.basicConfig(level = config['log level'])
    main(config)