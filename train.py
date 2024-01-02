from torch_geometric.loader import LinkNeighborLoader
import scripts.data_setup as data_setup
import config as cfg
from torch_geometric import seed_everything


def main():
    # Set Random Seed
    seed_everything(cfg.RANDOM_SEED)
    
    # Retrieve Data from files
    train_data, val_data, test_data = data_setup.get_train_val_test_data()
    
    # Define seed edges:
    edge_label_index = train_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label_index
    edge_label = train_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label


    # Mini batch loader to sample subgraphs
    # No negative edge sampling
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=cfg.TRAIN_SUBGRAPH_NUM_NEIGHBOURS,
        neg_sampling_ratio=0,
        edge_label_index=((cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS), edge_label_index),
        edge_label=edge_label,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=True,
    )
    
    # Inspect a sample:
    # sampled_data = next(iter(train_loader))

    # print("Sampled mini-batch:")
    # print("===================")
    # print(sampled_data)
    
    # print()

    
    


if __name__ == "__main__":
    main()