import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

class FunctionalData():

    def __init__(
        self,
        abundances_path,
        undetected_path,
        subcellular_path,
        go_path,
        **_
    ):
        
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

        ## Assemble abundances table
        abundance_df = cell_line_df[['Gene', 'RNA line ab']]
        abundance_df.loc[:, 'Protein present'] = 1
        abundance_df.loc[abundance_df['Gene'].isin(undetected_df['Gene']), 'Protein present'] = 0
        abundance_df = abundance_df.set_index('Gene')

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


class MorphologyData:

    def __init__(
        self,
        data_path,
        index_cols=['Metadata_Plate', 'Metadata_Well'],
        metadata_prefix='Metadata_',
        gene_col='Metadata_Symbol',
        file_format=None # Inferred from filename
    ) -> None:
        
        if file_format is None:
            file_format = Path(data_path).suffix.lstrip('.')
            logger.info("Inferred file format '{file_format}' from file extension.")
        
        if file_format == 'csv':
            self.normalized_profiles = pd.read_csv(data_path, index_col=index_cols)
        elif file_format == 'parquet':
            self.normalized_profiles = pd.read_parquet(data_path).set_index(index_cols)
        else:
            msg = f"I don't know how to read a '{file_format}' file"
            logger.error(msg)
            raise ValueError(msg)
        
        all_columns = self.normalized_profiles.columns
        self.profile_columns = all_columns[~all_columns.str.startswith(metadata_prefix)]

        self.gene_profiles = self.normalized_profiles.groupby(gene_col)[self.profile_columns].mean()
