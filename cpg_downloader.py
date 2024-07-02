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
        'explicit dtypes': {'Metadata_NCBIGeneID': 'str'}
    },
    'cpg0003-U2OS': (
        "s3://cellpainting-gallery/cpg0003-rosetta/"
        "broad/workspace/preprocessed_data/"
        "TA-ORF-BBBC037-Rohban/CellPainting/"
        "replicate_level_cp_augmented.csv.gz"
    ),
    'cpg0016': None
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

    # Identify metadata columns
    meta_cols = ['Metadata_Plate', 'Metadata_Well', 'Symbol', 'NCBIGeneID']

    # Enforce consistent metadata naming

    # Identify controls

    # Normalize


if __name__ == '__main__':
    parser = create_parser()
    config = get_config(parser.parse_args())
    logging.basicConfig(level = config['log level'])
    main(config)