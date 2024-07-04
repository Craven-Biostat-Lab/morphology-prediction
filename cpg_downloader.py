"""Download CPG datasets"""

from argparse import ArgumentParser
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pycytominer

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'log level': logging.INFO,
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
        'rename cols': {'Symbol': 'Metadata_Symbol'}
    },
    'cpg0003-U2OS': {
        'dataset uri': (
            "s3://cellpainting-gallery/cpg0003-rosetta/"
            "broad/workspace/preprocessed_data/"
            "TA-ORF-BBBC037-Rohban/CellPainting/"
            "replicate_level_cp_augmented.csv.gz"
        ),
        'rename cols': {
            'Metadata_GeneID': 'Metadata_NCBIGeneID',
            'Metadata_gene_name': 'Metadata_Symbol',
            'pert_type': 'Metadata_pert_type'
        }
    }
    #'cpg0016': None
}


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Download control-normalized CPG data.')
    parser.add_argument('-d', '--dataset', choices=KNOWN_DATASETS.keys(), required=True)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('-v', '--verbose', action = 'store_true')
    return parser


def get_config(args):
    config = {} | DEFAULT_CONFIG
    config['log level'] = logging.DEBUG if args.verbose else logging.INFO
    if args.dataset:
        config['dataset'] = args.dataset
    if args.output:
        config['out_path'] = args.output
    if not config.get('dataset uri'):
        config.update(KNOWN_DATASETS[config['dataset']])
    return config


def main(config):
    
    # Load profiles
    profiles = pd.read_csv(
        config['dataset uri'],
        storage_options={'anon': True}
    )

    columns = profiles.columns

    # Identify profile columns
    profile_cols = columns[
        columns.str.startswith('Nuclei_') |
        columns.str.startswith('Cells_') |
        columns.str.startswith('Cytoplasm_')
    ]

    # Rename columns if needed
    columns_to_rename = config.get('rename cols')
    if columns_to_rename:
        logger.info(f'Renaming columns according to {columns_to_rename}.')
        # Drop renaming targets if they are present (to avoid duplicate column names downstream)
        profiles.drop(columns=columns_to_rename.values(), inplace=True, errors='ignore')
        # Rename columns
        profiles.rename(columns=columns_to_rename, inplace=True)
        columns = profiles.columns
        logger.debug(f'Renaming result: {columns}')
    
    # Clean up Metadata_NCBIGeneID
    profiles['Metadata_NCBIGeneID'] = profiles['Metadata_NCBIGeneID'].astype('Int64').astype('str')

    # Identify indexing columns
    meta_cols = ['Metadata_Plate', 'Metadata_Well', 'Metadata_NCBIGeneID', 'Metadata_Symbol', 'Metadata_pert_type']

    logger.debug(profiles[meta_cols])

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