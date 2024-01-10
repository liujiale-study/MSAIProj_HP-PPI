import argparse
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric import seed_everything
import torch
import tqdm
import torch.nn.functional as F
import config as cfg
import model as m
import data_setup
import util
from sklearn.metrics import classification_report, confusion_matrix


def main(args):
    # Set Random Seed
    seed_everything(cfg.RANDOM_SEED)
    
    # Retrieve Data from files
    print("Retrieving Data...")
    _, _, test_data, data_metadata = data_setup.get_train_val_test_data()
    
    # Define seed edges:
    test_edge_label_index = test_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label_index
    test_edge_label = test_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=cfg.SUBGRAPH_NUM_NEIGHBOURS,
        edge_label_index=((cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS), test_edge_label_index),
        edge_label=test_edge_label,
        batch_size=cfg.TEST_BATCH_SIZE,
        shuffle=False,
    )
    
    # Model Setup
    model = m.HP_PPI_Prediction_Model(data_metadata=data_metadata)    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.ADAMW_LR, weight_decay=cfg.ADAMW_WEIGHT_DECAY)
    
    # Setup Softmax for Evaluation Metrics
    softmax = torch.nn.Softmax(dim=1)
    
    # Check for checkpoint folder to load from
    if args.cpfolder == None:
        print("Error: No checkpoint folder specified")
        return
    else:
        # Load from checkpoint if any specified
        fpath_chkpoint_folder = args.cpfolder
        print("Loading from Checkpoint Folder: " + fpath_chkpoint_folder)
        
        _, model_state_dict, optim_state_dict, _ = util.load_checkpoint(fpath_chkpoint_folder)

        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optim_state_dict)    
    


    # Vars for Recording Results
    list_preds = []
    list_ground_truths = []
    total_loss  = 0
    
    # Test Loop
    model.eval()
    for sampled_data in tqdm.tqdm(test_loader):
        with torch.no_grad():                
            # Input Batch into model
            sampled_data.to(device)
            pred = model(sampled_data)
            pred_classes = softmax(pred).argmax(dim=1)
            
            # Calculate Cross Entropy between Labels and Prediction
            ground_truth = sampled_data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label
            loss = F.cross_entropy(pred, ground_truth)
            
            # Record Loss
            total_loss += float(loss) * len(ground_truth)
            
            # Add prediction and ground truths to list
            list_preds.append(pred_classes)
            list_ground_truths.append(ground_truth)
                    
    # Get 1D Arrys of Predictions and Ground Truth
    arr_preds = torch.cat(list_preds, dim=0).cpu().numpy()
    arr_ground_truths = torch.cat(list_ground_truths, dim=0).cpu().numpy()
    
    # Number of data instances
    num_total_data_instances =  len(arr_ground_truths)
    
    # Get Confusion Matrix
    arr_confusion_matrix = confusion_matrix(arr_ground_truths, arr_preds)
    
    # Compute average loss per test instance
    loss = total_loss / num_total_data_instances
    
    # Compute Accuracy Scores
    list_acc = util.get_list_acc(arr_confusion_matrix)
    
    # Computes the average AUC of all possible pairwise combinations of classes
    classification_report = classification_report(arr_ground_truths, arr_preds, target_names=cfg.CLASSIFICATION_REPORT_CLASS_LABELS)
    
    util.write_test_results(fpath_chkpoint_folder, loss, list_acc, classification_report)
        
        
        


        



if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='HP PPI Model Training Script.')
    parser.add_argument('-cpf', '--cpfolder', help='Path to folder containing checkpoint to load', default=None)
    args = parser.parse_args()
    main(args)