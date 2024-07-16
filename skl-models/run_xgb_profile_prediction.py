"""Run profile prediction using XGB"""

from argparse import ArgumentParser
import logging
from pathlib import Path
import datetime

import yaml
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
    'predictions_version': '2024-06-18',
    'task': 'fitting', # fitting or prediction
    'kfold_seed': 20240611,
    'kfold_splits': 10
}


def create_parser() -> ArgumentParser:

    parser = ArgumentParser(description='Profile prediction using XGB')

    parser.add_argument(
        '-p', '--profiles', type=Path, required=False,
        help='Path to normalized CellProfiler profiles.'
    )
    parser.add_argument('-o', '--output', type=Path, required=False, help='Output folder path.')
    parser.add_argument(
        '-t', '--tag', type=str, default=datetime.date.today().isoformat(),
        help='Version tag to use for output files.'
    )
    parser.add_argument(
        '-c', '--config', type=Path, required=False,
        help='Configuration (YAML) file. Values in the file will override settings set by command line arguments.'
    )
    parser.add_argument('-v', '--verbose', action='store_true')

    return parser


def load_config(args):
    config = {} | DEFAULT_CONFIG
    config['log level'] = logging.DEBUG if args.verbose else logging.INFO
    if args.profiles:
        config['profiles_path'] = args.profiles
    if args.output:
        config['predictions_dir'] = args.output
    config['predictions_version'] = args.tag

    if args.config:
        with args.config.open('rt') as instream:
            config |= yaml.safe_load(instream)
    
    config['predictions_dir'] = Path(config['predictions_dir'])
    
    return config


class MorphologyData():

    def __init__(
        self,
        profiles_path,
        abundances_path,
        undetected_path,
        subcellular_path,
        go_path,
        metadata_perturbation_column='Metadata_Perturbation',
        metadata_perturbation_trt_val='CRISPR-trt',
        **_
    ):

        # Load profiles
        normalized_profiles = (
            pd.read_parquet(profiles_path)
            .reset_index()
            .set_index(['Metadata_Plate', 'Metadata_Well'])
        )

        # Load feature sets
        ## Gene abundances
        cell_line_df = pd.read_table(abundances_path)
        ## Undetected genes
        undetected_df = pd.read_table(undetected_path)
        ## Subcellular Localization
        sc_df = (
            pd
            .read_table(
                subcellular_path,
                usecols=['gene_id', 'hpa_location']
            )
            .drop_duplicates()
            .pivot_table(index='gene_id', columns='hpa_location', aggfunc=lambda _: 1, fill_value=0)
        )
        ## GO embeddings
        go_embedding = pd.read_csv(go_path, index_col=0)

        # Assemble tables
        ## Identify profile columns
        self.profile_columns = normalized_profiles.columns[~normalized_profiles.columns.str.startswith('Metadata')]

        ## Assemble abundances table
        abundance_df = cell_line_df[['Gene', 'RNA line ab']]
        abundance_df.loc[:, 'Protein present'] = 1
        abundance_df.loc[abundance_df['Gene'].isin(undetected_df['Gene']), 'Protein present'] = 0
        abundance_df = abundance_df.set_index('Gene')

        # Get the list of knockouts
        knockouts = normalized_profiles[
            normalized_profiles[metadata_perturbation_column] == metadata_perturbation_trt_val
        ]['Metadata_Symbol'].drop_duplicates()

        ## Putting feature vectors together
        self.gene_features = (
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
        self.well_features_and_profiles = self.gene_features.merge(
            normalized_profiles,
            left_index=True,
            right_on='Metadata_Symbol'
        )

        # Define feature groups

        self.feature_column_groups = {
            'GO': go_embedding.columns,
            'Abundance': abundance_df.columns,
            'SC': sc_df.columns,
            'GO+Abundance': list(go_embedding.columns)+list(abundance_df.columns),
            'GO+SC': list(go_embedding.columns)+list(sc_df.columns),
            'Abundance+SC': list(abundance_df.columns)+list(sc_df.columns),
            'All': self.gene_features.columns
        }

def main(config):

    if config['task'] == 'prediction':
        run_prediction(config)
    elif config['task'] == 'fitting':
        run_fitting(config)
    else:
        logger.error(f'Unrecognized task "{config["task"]}"!')

def run_prediction(config):

    morphology_data = MorphologyData(**config)

    # Set up output targets

    predictions_paths = {
        group: config['predictions_dir'] / f'XGB Predictions {group} {config["predictions_version"]}.parquet'
        for group in morphology_data.feature_column_groups.keys()
    }

    # Prepare cross-validation
    n_splits = config['kfold_splits']
    logger.info(f'Using {n_splits}-fold CV')

    splits = KFold(n_splits=10, shuffle=True, random_state=config['kfold_seed']).split(morphology_data.gene_features)
    gene_list = morphology_data.gene_features.index

    well_splits = [
        (
            np.flatnonzero(morphology_data.well_features_and_profiles['Metadata_Symbol'].isin(gene_list[train])),
            np.flatnonzero(morphology_data.well_features_and_profiles['Metadata_Symbol'].isin(gene_list[test]))
        )
        for train, test in splits
    ]

    model = xgb.XGBRegressor(tree_method='hist')

    # Run predictions
    for group, path in predictions_paths.items():
        logger.info(f'XGN predicting {group}')
        df = morphology_data.well_features_and_profiles[morphology_data.profile_columns].apply(
            lambda profile_component:
                cross_val_predict(
                    model,
                    morphology_data.well_features_and_profiles[morphology_data.feature_column_groups[group]],
                    profile_component,
                    cv=well_splits
                )
        )
        logger.info(f'Saving to {path}')
        df.to_parquet(path)

def run_fitting(config):

    morphology_data = MorphologyData(**config)

    # Set up output targets

    predictions_paths = {
        group: config['predictions_dir'] / f'Fit XGB {group} {config["predictions_version"]}.parquet'
        for group in morphology_data.feature_column_groups.keys()
    }

    model = xgb.XGBRegressor(tree_method='hist')

    # Run fitting
    for group, path in predictions_paths.items():
        logger.info(f'XGN fitting {group}')
        df = morphology_data.well_features_and_profiles[morphology_data.profile_columns].apply(
            lambda profile_component: (
                model
                .fit(
                    morphology_data.well_features_and_profiles[morphology_data.feature_column_groups[group]],
                    profile_component
                )
                .predict(morphology_data.well_features_and_profiles[morphology_data.feature_column_groups[group]])
            )
        )
        logger.info(f'Saving to {path}')
        df.to_parquet(path)


if __name__ == '__main__':
    parser = create_parser()
    config = load_config(parser.parse_args())
    logging.basicConfig(level=config['log level'])
    main(config)