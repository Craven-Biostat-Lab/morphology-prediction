""" Prepare CPG0016 data for GNN training/testing
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric import Data
from stringdb_alias import HGNCMapper

def prep_edges(stringdb_txt):

    # Load the STRING graph:
    string_df = pd.read_table(stringdb_txt, sep=' ')

    # Filter edges here if desired. Right now we don't.

    # Get series of unique proteins (this also gives us enumeration for free)
    all_proteins = string_df.protein1.drop_duplicates().sort_values(ignore_index=True)

    protein_lookup = pd.Series(all_proteins.index, index=all_proteins)

    # Get edge list and edge features
    edge_index = torch.tensor(
        [
            protein_lookup[string_df.protein1].to_numpy(),
            protein_lookup[string_df.protein2].to_numpy()
        ],
        dtype=torch.long
    )

    edge_attr = torch.tensor(
        string_df[[
            'neighborhood',
            'fusion',
            'cooccurence',
            'coexpression',
            'experimental',
            'database',
            'textmining',
            'combined_score'
        ]].to_numpy() / 1000.0,
        dtype=torch.float
    )

    return protein_lookup, edge_index, edge_attr

def get_protein_go_map(mapfile_xlsx, protein_col='Unnamed: 20', go_col='Unnamed: 6'):

    raw_df = pd.read_excel(mapfile_xlsx)

    protein_lists = raw_df[protein_col].str.split('; ')
    go_lists = raw_df[go_col].str.split('; ')

    go_mat = (
        pd.DataFrame({'Protein': protein_lists, 'GO': go_lists})
        .explode('Protein')
        .explode('GO')
        .pivot_table(index='Protein', columns='GO', aggfunc=lambda _: 1, fill_value=0)
    )

    return go_mat


def broadcast_feature_matrix(protein_lookup, feature_df):
    """Broadcast feature_df to a numpy matrix with rows matching protein_lookup values"""

    n_proteins = len(protein_lookup)
    n_features = feature_df.shape[1]

    result = np.zeros([n_proteins, n_features])
    result[protein_lookup[feature_df.index], :] = feature_df


def get_protein_labels(gene_labels_csv, gene_col, label_col, mapper):

    labels_df = pd.read_csv(gene_labels_csv)

    # Use our ID converter
    return pd.DataFrame({'label': labels_df[label_col]}, index=mapper.get_string_ids(labels_df[gene_col]))


def build_node_classification_set(protein_lookup, edge_index, edge_attr, labels, *feature_dfs):
    
    n_proteins = len(protein_lookup)

    y = np.zeros([n_proteins, 1], dtype=int)

    x = np.concatenate(
        [
            broadcast_feature_matrix(protein_lookup, feature_df)
            for feature_df in feature_dfs
        ],
        axis=1
    )

    # This object won't have the train/validation/test masks. Those can be attached later.

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )