"""Run profile prediction using XGB"""

from argparse import ArgumentParser
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_predict
import xgboost as xgb


logger = logging.getLogger(__name__)


DEFAULT_FUNCTIONAL_DATA_PATH = Path('/ua/ml-group/igvf/data')

DEFAULT_CONFIG = {
    'profiles_path': Path('../data/cpg0016/version_2024-04-15/normalized_profiles'),
    'abundances_path': DEFAULT_FUNCTIONAL_DATA_PATH / 'cellular-localization' / 'gene_abundances_U2OS.tsv',
    'undetected_path': DEFAULT_FUNCTIONAL_DATA_PATH / 'cellular-localization' / 'undetected_genes_U2OS.tsv',
    'subcellular_path': DEFAULT_FUNCTIONAL_DATA_PATH / 'subcellular-localization' / 'uniprot_reactome_hpa_merged.tsv',
    'go_path': DEFAULT_FUNCTIONAL_DATA_PATH / 'GO_Embeddings' / 'go_embedding_64.csv',
    'predictions_dir': Path('results'),
    'predictions_version': '2024-06-11',
    'kfold_seed': 20240611,
    'kfold_splits': 10
}


def create_parser() -> ArgumentParser:

    parser = ArgumentParser('Profile prediction using XGB')

    parser.add_argument('--config', type=Path, required=False)

    return parser

def main(config):

    # Load profiles
    normalized_profiles = pd.read_parquet(config['profiles_path']).set_index(['Metadata_Plate', 'Metadata_Well'])

    # Load feature sets
    ## Gene abundances
    cell_line_df = pd.read_table(config['abundances_path'])
    ## Undetected genes
    undetected_df = pd.read_table(config['undetected_path'])
    ## Subcellular Localization
    sc_df = (
        pd
        .read_table(
            config['subcellular_path'],
            usecols=['gene_id', 'hpa_location']
        )
        .drop_duplicates()
        .pivot_table(index='gene_id', columns='hpa_location', aggfunc=lambda _: 1, fill_value=0)
    )
    ## GO embeddings
    go_embedding = pd.read_csv(config['go_path'], index_col=0)

    # Assemble tables

    ## Get the list of knockouts
    knockouts = normalized_profiles[normalized_profiles['Metadata_Perturbation'] == 'CRISPR-trt']['Metadata_Symbol'].drop_duplicates()

    ## Identify profile columns
    profile_columns = normalized_profiles.columns[~normalized_profiles.columns.str.startswith('Metadata')]

    ## Assemble abundances table
    abundance_df = cell_line_df[['Gene', 'RNA line ab']]
    abundance_df.loc[:, 'Protein present'] = 1
    abundance_df.loc[abundance_df['Gene'].isin(undetected_df['Gene']), 'Protein present'] = 0
    abundance_df = abundance_df.set_index('Gene')

    ## Putting feature vectors together
    gene_features = (
        go_embedding
        .merge(
            abundance_df,
            how='inner',
            left_index=True,
            right_index=True            
        )
        .merge(
            sc_df,
            how='inner',
            left_index=True,
            right_index=True
        )
        .filter(knockouts.values, axis='index')
    )

    ## Merging with profile vectors
    well_features_and_profiles = gene_features.merge(
        normalized_profiles,
        left_index=True,
        right_on='Metadata_Symbol'
    )

    # Define feature groups

    feature_column_groups = {
        'GO': go_embedding.columns,
        'Abundance': abundance_df.columns,
        'SC': sc_df.columns,
        'GO+Abundance': list(go_embedding.columns)+list(abundance_df.columns),
        'GO+SC': list(go_embedding.columns)+list(sc_df.columns),
        'Abundance+SC': list(abundance_df.columns)+list(sc_df.columns),
        'All': gene_features.columns
    }

    # Set up output targets

    predictions_paths = {
        group: config['predictions_dir'] / f'XGB Predictions {group} {config["predictions_version"]}.parquet'
        for group in feature_column_groups.keys()
    }

    # Prepare cross-validation
    n_splits = config['kfold_splits']
    logger.info(f'Using {n_splits}-fold CV')

    splits = KFold(n_splits=10, shuffle=True, random_state=config['kfold_seed']).split(gene_features)
    gene_list = gene_features.index

    well_splits = [
        (
            np.flatnonzero(well_features_and_profiles['Metadata_Symbol'].isin(gene_list[train])),
            np.flatnonzero(well_features_and_profiles['Metadata_Symbol'].isin(gene_list[test]))
        )
        for train, test in splits
    ]

    model = xgb.XGBRegressor(tree_method='hist')

    # Run predictions
    for group, path in predictions_paths.items():
        logger.info(f'XGN predicting {group}')
        df = well_features_and_profiles[profile_columns].apply(
            lambda profile_component:
                cross_val_predict(
                    model,
                    well_features_and_profiles[feature_column_groups[group]],
                    profile_component,
                    cv=well_splits
                )
        )
        logger.info(f'Saving to {path}')
        df.to_parquet(path)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    # TODO: extract config from args
    logging.basicConfig(level=logging.INFO)
    main(DEFAULT_CONFIG)