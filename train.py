from torch_geometric.loader import LinkNeighborLoader
from torch_geometric import seed_everything
import torch
import tqdm
import torch.nn.functional as F
import data_setup
import config as cfg
import model as m
from sklearn.metrics import roc_auc_score


def main():
    # Set Random Seed
    seed_everything(cfg.RANDOM_SEED)
    
    # Retrieve Data from files
    train_data, val_data, test_data, data_metadata = data_setup.get_train_val_test_data()
    
    # Define seed edges:
    train_edge_label_index = train_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label_index
    train_edge_label = train_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label


    # Mini batch loader to sample subgraphs
    # No negative edge sampling
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=cfg.TRAIN_SUBGRAPH_NUM_NEIGHBOURS,
        neg_sampling_ratio=0,
        edge_label_index=((cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS), train_edge_label_index),
        edge_label=train_edge_label,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=True,
    )
    
    # Define the validation seed edges:
    val_edge_label_index = val_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label_index
    val_edge_label = val_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[20, 10],
        edge_label_index=((cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS), val_edge_label_index),
        edge_label=val_edge_label,
        batch_size=3 * 128,
        shuffle=False,
    )
    
    # Model Setup
    model = m.HP_PPI_Model(num_hidden_chnls=64, data_metadata=data_metadata)    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.ADAMW_LR, weight_decay=cfg.ADAMW_WEIGHT_DECAY)
    start_epoch = 1
    
    # Setup Softmax for Evaluation Metrics
    softmax = torch.nn.Softmax(dim=1)

    # Loop through epochs
    for epoch in range(start_epoch, (cfg.NUM_EPOCHS + 1)):
        
        # Training Loop
        total_train_loss = total_train_data_instances = total_train_corrects = 0
        model.train()
        for sampled_data in tqdm.tqdm(train_loader):
            
            # Reset Gradiant
            optimizer.zero_grad()

            # Input batch into model
            sampled_data.to(device)
            pred = model(sampled_data)

            # Calculate Cross Entropy between Labels and Prediction
            ground_truth = sampled_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label
            loss = F.cross_entropy(pred, ground_truth)

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Record Loss and Data Instance Count
            num_data_instances = len(ground_truth)
            total_train_loss += float(loss) * num_data_instances
            total_train_data_instances += num_data_instances
            
            # Calculate accuracy
            pred_indices = softmax(pred).argmax(dim=1)
            total_train_corrects += (pred_indices == ground_truth).sum().item()
            
            
        # Compute average training loss per supervision edge
        train_loss = total_train_loss / total_train_data_instances
        
        # Compute accuracy
        train_acc = total_train_corrects / total_train_data_instances
                
        # Validation Loop
        arr_val_softmax_preds = []
        arr_val_ground_truths = []
        total_val_loss = total_val_corrects = 0
        
        model.eval()
        for sampled_data in tqdm.tqdm(val_loader):
            with torch.no_grad():                
                # Input Batch into model
                sampled_data.to(device)
                pred = model(sampled_data)
                sm_pred = softmax(pred)
                
                # Calculate Cross Entropy between Labels and Prediction
                ground_truth = sampled_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label
                loss = F.cross_entropy(pred, ground_truth)
                
                # Record Loss
                total_val_loss += float(loss) * len(ground_truth)
                
                # Calculate accuracy
                pred_indices = sm_pred.argmax(dim=1)
                total_val_corrects += (pred_indices == ground_truth).sum().item()
                
                # Add to arrays for ROC AUC Calculations
                arr_val_softmax_preds.append(sm_pred)
                arr_val_ground_truths.append(ground_truth)
                        
        arr_val_softmax_preds = torch.cat(arr_val_softmax_preds, dim=0).cpu().numpy()
        arr_val_ground_truths = torch.cat(arr_val_ground_truths, dim=0).cpu().numpy()
        
        # Computes the average AUC of all possible pairwise combinations of classes
        auc = roc_auc_score(arr_val_ground_truths, arr_val_softmax_preds, multi_class='ovo')
        
        # Compute average validation loss per validation edge
        num_val_ground_truths = len(arr_val_ground_truths)
        val_loss = total_val_loss / num_val_ground_truths
        
        # Compute accuracy
        val_acc = total_val_corrects / num_val_ground_truths
        
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, ROC-AUC: {auc:.4f}")
        
            
    


if __name__ == "__main__":
    main()