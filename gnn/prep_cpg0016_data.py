""" Prepare CPG0016 data for GNN training/testing
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from stringdb_alias import HGNCMapper

def prep_edges(stringdb_txt):

    # Load the STRING graph:
    string_df = pd.read_table(stringdb_txt, sep=' ')

    # Filter edges here if desired. Right now we don't.

    # Get series of unique proteins (this also gives us enumeration for free)
    all_proteins = (
        string_df.protein1
        .drop_duplicates()
        .sort_values(ignore_index=True)
        .reset_index(drop=True)
    )

    protein_lookup = pd.Series(all_proteins.index, index=all_proteins)

    # Get edge list and edge features
    edge_index = torch.tensor(
        np.concatenate(
            (
                protein_lookup[string_df.protein1].to_numpy(),
                protein_lookup[string_df.protein2].to_numpy()
            ),
            axis=0
        ),
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
        .assign(Protein=lambda df: '9606.' + df.Protein.str.partition('.')[0])
        .pivot_table(index='Protein', columns='GO', aggfunc=lambda _: 1, fill_value=0)
    )

    return go_mat


def broadcast_feature_matrix(protein_lookup, feature_df):
    """Broadcast feature_df to a numpy matrix with rows matching protein_lookup values"""

    n_proteins = len(protein_lookup)
    n_features = feature_df.shape[1]


    feature_df = feature_df.filter(items=protein_lookup.index, axis='index')

    result = np.zeros([n_proteins, n_features])
    result[protein_lookup[feature_df.index], :] = feature_df

    return result


def get_protein_labels(gene_labels_csv, gene_col, label_col, id_mapper, label_map):

    labels_df = pd.read_csv(gene_labels_csv)

    # Use our ID converter
    string_ids = id_mapper.get_string_ids(labels_df[gene_col])

    labels_df = labels_df[~string_ids.isna()]
    string_ids = string_ids[~string_ids.isna()]

    return pd.DataFrame(
        {'label': labels_df[label_col].map(label_map), 'protein' :string_ids}
    ).set_index('protein')


def build_node_classification_set(protein_lookup, edge_index, edge_attr, labels, *feature_dfs):
    
    n_proteins = len(protein_lookup)

    y = np.zeros([n_proteins, 1], dtype=int)
    y[protein_lookup[labels.index], 0] = labels.label.to_numpy()

    x = np.concatenate(
        [
            broadcast_feature_matrix(protein_lookup, feature_df)
            for feature_df in feature_dfs
        ],
        axis=1
    )

    # This object won't have the train/validation/test masks. Those can be attached later.

    return Data(
        x=torch.tensor(x),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(y)
    )

def prep_data(stringdb_txt, mapfile_xlsx, gene_labels_csv, string_info_file, string_alias_file, generator):
    """Main function. Prepares our data for the GNN. Returns a Data object."""

    # Get Network
    protein_lookup, edge_index, edge_attr = prep_edges(stringdb_txt)

    # Get GO features
    go_feature_matrix = get_protein_go_map(mapfile_xlsx)

    # ID mapper
    mapper = HGNCMapper(string_info_file, string_alias_file)

    # Get Labels
    labels = get_protein_labels(
        gene_labels_csv=gene_labels_csv,
        gene_col='Metadata_Symbol',
        label_col='gene_label',
        id_mapper=mapper,
        label_map={
            'Positive': 1,
            'Indeterminate': 0,
            'Negative': -1
        }
    )

    # Populate data object
    data = build_node_classification_set(protein_lookup, edge_index, edge_attr, labels, go_feature_matrix)

    # Set up train/test/validation split
    split_lengths = [0.3, 0.3, 0.4]
    pos_idx = np.nonzero(data.y == 1)
    neg_idx = np.nonzero(data.y == 0)
    pos_test, pos_val, pos_train = torch.utils.data.random_split(pos_idx, split_lengths, generator)
    neg_test, neg_val, neg_train = torch.utils.data.random_split(neg_idx, split_lengths, generator)

    n=len(protein_lookup)
    train_mask = np.zeros(n, dtype=int)
    train_mask[pos_train] = 1
    train_mask[neg_train] = 1
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    test_mask = np.zeros(n, dtype=int)
    test_mask[pos_test] = 1
    test_mask[neg_test] = 1
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
    val_mask = np.zeros(n, dtype=int)
    val_mask[pos_val] = 1
    val_mask[neg_val] = 1
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)

    return data

def test_prep():
    return prep_data(
        stringdb_txt='../../../data/STRING/9606.protein.links.detailed.v11.5.txt.gz',
        mapfile_xlsx='../data/GO/HUMAN_9606_idmapping.xlsx',
        gene_labels_csv='../data/cpg0016/version_2023-06-14/gene_labels.csv',
        string_info_file='../../../data/STRING/9606.protein.info.v11.5.txt.gz',
        string_alias_file='../../../data/STRING/9606.protein.aliases.v11.5.txt.gz',
        generator=torch.Generator()
    )

if __name__ == "__main__":
    # TODO: Make these arguments

    data = test_prep()