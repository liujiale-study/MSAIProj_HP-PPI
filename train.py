import argparse
from torch_geometric.loader import LinkNeighborLoader, ImbalancedSampler
from torch_geometric import seed_everything
import torch
import tqdm
import torch.nn.functional as F
import config as cfg
import model as m
import data_setup
import util
from sklearn.metrics import confusion_matrix, classification_report


def main(args):
    # Set Random Seed
    seed_everything(cfg.RANDOM_SEED)
    
    # Retrieve Data from files
    print("Retrieving Data...")
    train_data, val_data, _, data_metadata = data_setup.get_train_val_test_data()
    
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
    model = m.HP_PPI_Prediction_Model(data_metadata=data_metadata)    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.ADAMW_LR, weight_decay=cfg.ADAMW_WEIGHT_DECAY)
    start_epoch = 1
    
    # List for recording metric scores
    list_rec = []
    
    # Setup CrossEntropy/Softmax
    softmax = torch.nn.Softmax(dim=1)
    cross_entropy = torch.nn.CrossEntropyLoss(weight=torch.tensor(cfg.CROSS_ENTROPY_WEIGHTS).to(device=device))
    
    # Load from checkpoint if any specified
    if args.cpfolder != None:
        fpath_chkpoint_folder = args.cpfolder
        print("Loading from Checkpoint Folder: " + fpath_chkpoint_folder)
        
        chkpt_epoch, model_state_dict, optim_state_dict, chkpt_list_rec = util.load_checkpoint(fpath_chkpoint_folder)

        start_epoch = chkpt_epoch + 1
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optim_state_dict)
        list_rec = chkpt_list_rec
    
    

    # Loop through epochs
    for epoch in range(start_epoch, (cfg.NUM_EPOCHS + 1)):
        
        # Training Loop
        
        # Vars for Recording Results
        train_list_preds = []
        train_list_ground_truths = []
        total_train_loss = 0
        
        model.train()
        for sampled_data in tqdm.tqdm(train_loader):
            
            # Reset Gradiant
            optimizer.zero_grad()

            # Input batch into model
            sampled_data.to(device)
            pred = model(sampled_data)
            pred_classes = softmax(pred).argmax(dim=1)

            # Calculate Cross Entropy between Labels and Prediction
            ground_truth = sampled_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label
            loss = cross_entropy(pred, ground_truth)

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Record Loss and Data Instance Count
            num_data_instances = len(ground_truth)
            total_train_loss += float(loss) * num_data_instances
            
            # Add prediction and ground truths to list
            train_list_preds.append(pred_classes)
            train_list_ground_truths.append(ground_truth)
        
        # Get 1D Arrays of Predictions and Ground Truth
        arr_train_preds = torch.cat(train_list_preds, dim=0).cpu().numpy()
        arr_train_ground_truths = torch.cat(train_list_ground_truths, dim=0).cpu().numpy()
    
        # Number of evaluated data instances
        train_num_total_data_instances =  len(arr_train_ground_truths)
    
        # Get Confusion Matrix
        arr_train_cmatrix = confusion_matrix(arr_train_ground_truths, arr_train_preds)
    
        # Compute average loss per test instance
        train_loss = total_train_loss / train_num_total_data_instances
    
        # Compute Accuracy Scores
        list_train_acc = util.get_list_acc(arr_train_cmatrix)
                
        # Validation Loop
        
        # Vars for Recording Results
        val_list_preds = []
        val_list_ground_truths = []
        total_val_loss = 0
        
        model.eval()
        for sampled_data in tqdm.tqdm(val_loader):
            with torch.no_grad():                
                # Input Batch into model
                sampled_data.to(device)
                pred = model(sampled_data)
                pred_classes = softmax(pred).argmax(dim=1)
                
                # Calculate Cross Entropy between Labels and Prediction
                ground_truth = sampled_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label
                loss = F.cross_entropy(pred, ground_truth)
                
                # Record Loss
                total_val_loss += float(loss) * len(ground_truth)
                
                # Add prediction and ground truths to list
                val_list_preds.append(pred_classes)
                val_list_ground_truths.append(ground_truth)
              
        # Get 1D Arrays of Predictions and Ground Truth                
        arr_val_preds = torch.cat(val_list_preds, dim=0).cpu().numpy()
        arr_val_ground_truth = torch.cat(val_list_ground_truths, dim=0).cpu().numpy()
        
        # Number of evaluated data instances
        val_num_total_data_instances =  len(arr_val_ground_truth)
    
        # Get Confusion Matrix
        arr_val_cmatrix = confusion_matrix(arr_val_ground_truth, arr_val_preds)
    
        # Compute average loss per evaluated data instance
        val_loss = total_val_loss / val_num_total_data_instances
    
        # Compute Accuracy Scores
        list_val_acc = util.get_list_acc(arr_val_cmatrix)
        
        # Get Classification Report
        dict_val_classification_report = classification_report(arr_val_ground_truth, arr_val_preds, 
                                                               target_names=cfg.CLASSIFICATION_REPORT_CLASS_LABELS, output_dict=True, zero_division=0)
        list_f1 = []
        for label in cfg.CLASSIFICATION_REPORT_CLASS_LABELS:
            list_f1.append(dict_val_classification_report[label]["f1-score"])
        
        
        # Print Results to Console
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Overall Train Acc: {list_train_acc[-1]:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Overall Acc: {list_val_acc[-1]:.2f}%")
        
        # Update metric records
        list_rec.append([epoch, train_loss] + list_train_acc + [val_loss] + list_val_acc + list_f1)
        
        # Checkpoint every x epoch
        if epoch % cfg.CHKPOINT_EVERY_NUM_EPOCH == 0:
            util.save_checkpoint(epoch, model, optimizer, list_rec)
        
    # All Epochs Finished
    # Save to checkpoint
    print("Training Finished")
    util.save_checkpoint(cfg.NUM_EPOCHS, model, optimizer, list_rec, True)
        



if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='HP PPI Model Training Script.')
    parser.add_argument('-cpf', '--cpfolder', help='Path to folder containing checkpoint to load', default=None)
    args = parser.parse_args()
    main(args)