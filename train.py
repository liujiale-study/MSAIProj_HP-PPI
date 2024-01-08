import argparse
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric import seed_everything
import torch
import tqdm
import torch.nn.functional as F
import config as cfg
import model as m
import data_setup
import file_io
from sklearn.metrics import roc_auc_score


def main(args):
    # Set Random Seed
    seed_everything(cfg.RANDOM_SEED)
    
    # Retrieve Data from files
    print("Retrieving Data...")
    train_data, val_data, test_data, data_metadata = data_setup.get_train_val_test_data()
    
    # Define seed edges:
    train_edge_label_index = train_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label_index
    train_edge_label = train_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label


    # Mini batch loader to sample subgraphs
    # No negative edge sampling
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=cfg.SUBGRAPH_NUM_NEIGHBOURS,
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
        num_neighbors=cfg.SUBGRAPH_NUM_NEIGHBOURS,
        edge_label_index=((cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS), val_edge_label_index),
        edge_label=val_edge_label,
        batch_size=cfg.VAL_BATCH_SIZE,
        shuffle=False,
    )
    
    # Model Setup
    model = m.HP_PPI_Prediction_Model(num_hidden_chnls=cfg.MODEL_HIDDEN_NUM_CHNLS, data_metadata=data_metadata)    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.ADAMW_LR, weight_decay=cfg.ADAMW_WEIGHT_DECAY)
    start_epoch = 1
    
    # List for recording metric scores
    list_rec = []
    
    # Setup Softmax for Evaluation Metrics
    softmax = torch.nn.Softmax(dim=1)
    
    # Load from checkpoint if any specified
    if args.cpfolder != None:
        fpath_chkpoint_folder = args.cpfolder
        print("Loading from Checkpoint Folder: " + fpath_chkpoint_folder)
        
        chkpt_epoch, model_state_dict, optim_state_dict, chkpt_list_rec = file_io.load_checkpoint(fpath_chkpoint_folder)

        start_epoch = chkpt_epoch + 1
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optim_state_dict)
        list_rec = chkpt_list_rec
    
    

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
        list_val_softmax_preds = []
        list_val_ground_truths = []
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
                list_val_softmax_preds.append(sm_pred)
                list_val_ground_truths.append(ground_truth)
                        
        list_val_softmax_preds = torch.cat(list_val_softmax_preds, dim=0).cpu().numpy()
        list_val_ground_truths = torch.cat(list_val_ground_truths, dim=0).cpu().numpy()
        
        # Computes the average AUC of all possible pairwise combinations of classes
        val_roc_auc = roc_auc_score(list_val_ground_truths, list_val_softmax_preds, multi_class='ovo')
        
        # Compute average validation loss per validation edge
        num_val_ground_truths = len(list_val_ground_truths)
        val_loss = total_val_loss / num_val_ground_truths
        
        # Compute accuracy
        val_acc = total_val_corrects / num_val_ground_truths
        
        # Print Results to Console
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, ROC-AUC: {val_roc_auc:.4f}")
        
        # Update metric records
        list_rec.append([epoch, train_loss, train_acc, val_loss, val_acc, val_roc_auc])
        
        # Checkpoint every x epoch
        if epoch % cfg.CHKPOINT_EVERY_NUM_EPOCH == 0:
            file_io.save_checkpoint(epoch, model, optimizer, list_rec)
        
    # All Epochs Finished
    # Save to checkpoint
    print("Training Finished")
    file_io.save_checkpoint(cfg.NUM_EPOCHS, model, optimizer, list_rec, True)
        



if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='HP PPI Model Training Script.')
    parser.add_argument('-cpf', '--cpfolder', help='Path to folder containing checkpoint to load', default=None)
    args = parser.parse_args()
    main(args)