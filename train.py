from torch_geometric.loader import LinkNeighborLoader
from torch_geometric import seed_everything
import torch
import tqdm
import torch.nn.functional as F
import data_setup
import config as cfg
import model as m


def main():
    # Set Random Seed
    seed_everything(cfg.RANDOM_SEED)
    
    # Retrieve Data from files
    train_data, val_data, test_data, data_metadata = data_setup.get_train_val_test_data()
    
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
    
    model = m.HP_PPI_Model(num_hidden_chnls=64, data_metadata=data_metadata)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.ADAMW_LR)
    start_epoch = 1
    
    

    for epoch in range(start_epoch, (cfg.NUM_EPOCHS + 1)):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()

            sampled_data.to(device)
            pred = model(sampled_data)

            ground_truth = sampled_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label
            loss = F.cross_entropy(pred, ground_truth)
            
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
    
    


if __name__ == "__main__":
    main()