import pandas as pd
import torch
import numpy as np
import config as cfg
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

# === Main Function ===
def get_train_val_test_data():
    
    # === Read Data from File ===
    df_mouse = pd.read_csv(cfg.PATH_MOUSE_FEAT)
    df_virus = pd.read_csv(cfg.PATH_VIRUS_FEAT)
    df_interactions = pd.read_csv(cfg.PATH_INTERACTIONS)


    # Setup data tensor
    df_mouse = df_mouse.sort_values(cfg.COLNAME_MID)
    mouse_feat = torch.from_numpy(df_mouse.drop(columns=[cfg.COLNAME_MID, cfg.COLNAME_MUNIPROTID]).values)

    df_virus = df_virus.sort_values(cfg.COLNAME_VID)
    virus_feat = torch.from_numpy(df_virus.drop(columns=[cfg.COLNAME_VID, cfg.COLNAME_VUNIPROTID]).values)

    edge_index_mouse_to_virus = torch.t(torch.from_numpy(df_interactions[[cfg.COLNAME_MID, cfg.COLNAME_VID]].values))


    # === Setup HetreoData ===
    data = HeteroData()

    # Save node indices:
    data[cfg.NODE_MOUSE].node_id = torch.arange(len(df_mouse))
    data[cfg.NODE_VIRUS].node_id = torch.arange(len(df_virus))

    # Add the node features and edge indices:
    data[cfg.NODE_MOUSE].x = mouse_feat
    data[cfg.NODE_VIRUS].x = virus_feat
    data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_index = edge_index_mouse_to_virus
    
    # Add edge attributes
    data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_attrs = torch.from_numpy(df_interactions['rating'].values)

    # We also need to make sure to add the reverse edges from virus to mouse
    # in order to let a GNN be able to pass messages in both directions.
    # We can leverage the `T.ToUndirected()` transform for this from PyG:
    data = T.ToUndirected()(data)

    # Print to Console for Checking HeteroData Setup
    # print("Full data:")
    # print("==============")
    # print(data)


    # === Train/Val/Test Split ===
    # Split the set of edges into training, validation, and testing edges.
    # Across the training edges, we use 70% of edges for message passing,
    # and 30% of edges for supervision.
    # Negative sampling removed
    transform = T.RandomLinkSplit(
        num_val=cfg.DATASPLIT_RATIO_VAL,
        num_test=cfg.DATASPLIT_RATIO_TEST,
        disjoint_train_ratio=cfg.DATASPLIT_DISJOINT_TRAIN_RATIO,
        neg_sampling_ratio=0,
        add_negative_train_samples=False,
        edge_types=(cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS),
        rev_edge_types=(cfg.NODE_VIRUS, cfg.REV_EDGE_INTERACT, cfg.NODE_MOUSE),
    )

    train_data, val_data, test_data = transform(data)

    # Ensure no negative edges added:
    assert train_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label.min() == 1
    assert train_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label.max() == 1
    assert val_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label.min() == 1
    assert val_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label.max() == 1
    assert test_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label.min() == 1
    assert test_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label.max() == 1

    # Set Edge Labels
    train_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label = get_edge_label(df_interactions, train_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label_index)
    val_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label = get_edge_label(df_interactions, val_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label_index)
    test_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label = get_edge_label(df_interactions, test_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label_index)

    # Print to Console for Checking Train/Val/Test Split
    # Size of Train edge_label = total number of edges * 0.7 (% of edges in training set) * 0.3 (% of training edges used for supervision)
    # Size of Val edge_label = total number of edges * 0.2 (% of edges in validation set)
    # Size of Test edge_label = total number of edges * 0.1 (% of edges in test set)
    # Total number of edges = Size of Train edge_label + Size(dim=1) of Train edge_index + Size of Val edge_label + Size of Test edge_label 
    # print("Training data:")
    # print("==============")
    # print(train_data)
    # print()
    # print("Validation data:")
    # print("================")
    # print(val_data)
    # print("Test data:")
    # print("================")
    # print(test_data)
    
    return train_data, val_data, test_data

    
# === get_edge_label Function ===
# Function to get edge labels from df_interactions dataframe
# Expected shape of edge_label_index tensor: (2, Number of edges), 
#   where edge_label_index[cfg.INDEX_EDGE_LABEL_MID,:] are mIDs and edge_label_index[cfg.INDEX_EDGE_LABEL_VID,:]
# Assumes all mID-vID pairs given in edge_label_index are in dataframe, and each mID-vID pair has a unique label
def get_edge_label(df_interactions, edge_label_index):

    arr_edge_label_index = edge_label_index.T.numpy()
    df_edge_label_index = pd.DataFrame({cfg.COLNAME_MID: arr_edge_label_index[:, cfg.INDEX_EDGE_LABEL_MID], cfg.COLNAME_VID: arr_edge_label_index[:, cfg.INDEX_EDGE_LABEL_VID]})
    df_merged = pd.merge(df_edge_label_index, df_interactions, how='left', left_on=[cfg.COLNAME_MID,cfg.COLNAME_VID], right_on = [cfg.COLNAME_MID,cfg.COLNAME_VID])
   
    assert df_merged.isnull().values.any() == False
    edge_labels = torch.t(torch.from_numpy(df_merged[cfg.COLNAME_LABEL].values)) 
    
    assert edge_labels.dim() == 1 
    assert edge_labels.size(dim=0) == edge_label_index.size(dim=1)
    
    return edge_labels


