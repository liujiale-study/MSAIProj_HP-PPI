import argparse
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric import seed_everything
import torch
import tqdm
import torch.nn.functional as F
import os
import config as cfg
import model as m
import data_setup
import util
from sklearn.metrics import classification_report, matthews_corrcoef


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
    
    # Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")
    
    
    # Check for checkpoint folder to load from
    if args.cpfolder == None:
        print("Error: No checkpoint folder specified")
        return
    else:
        # Load from checkpoint if any specified
        fpath_chkpoint_folder = args.cpfolder
        print("Loading from Checkpoint Folder: " + fpath_chkpoint_folder)
        
        currmodel_epoch, currmodel_state_dict, model_gnn_op_type, _, _, bestmodel_state_dict, bestmodel_epoch = util.load_checkpoint(fpath_chkpoint_folder)
        
        print("Loaded model type: " + cfg.GNN_OP_TYPE_NAMES[model_gnn_op_type])

        # Base Model Setup
        model = m.PPIVirulencePredictionModel(data_metadata=data_metadata, gnn_op_type=model_gnn_op_type)
        model.to(device)
        
        # Load current model state dict and evaluate
        model.load_state_dict(currmodel_state_dict)
        currmodel_loss, currmodel_classifi_report, currmodel_mcc = test_model(test_loader, model, device)
        fpath_currmodel_test_result = os.path.join(fpath_chkpoint_folder, cfg.FNAME_TEST_RESULT_CURRMODEL_TXT)
        str_currmodel_headerline =  "==== Current Model Results ===="
        util.write_test_results(fpath_currmodel_test_result, str_currmodel_headerline, currmodel_epoch, currmodel_loss, currmodel_classifi_report, currmodel_mcc)
        print()
        
        # Load best model state dict and evaluate
        model.load_state_dict(bestmodel_state_dict)
        bestmodel_loss, bestmodel_classifi_report, bestmodel_mcc = test_model(test_loader, model, device)
        fpath_currmodel_test_result = os.path.join(fpath_chkpoint_folder, cfg.FNAME_TEST_RESULT_BEST_MODEL_TXT)
        str_bestmodel_headerline = "==== Best Model Results ===="
        util.write_test_results(fpath_currmodel_test_result, str_bestmodel_headerline, bestmodel_epoch, bestmodel_loss, bestmodel_classifi_report, bestmodel_mcc)
        
        
# Function to test a given model on training set
def test_model(test_loader, model, device):
    # Setup Softmax for Evaluation Metrics
    softmax = torch.nn.Softmax(dim=1)
    
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
    
    # Compute average loss per test instance
    loss = total_loss / num_total_data_instances
    
    # Get classification report
    class_report = classification_report(arr_ground_truths, arr_preds, target_names=cfg.CLASSIFICATION_REPORT_CLASS_LABELS, zero_division=0, digits=6)
    
    # Get Matthews Correlation
    mcc = matthews_corrcoef(arr_ground_truths, arr_preds)
    
    return loss, class_report, mcc
    
        
        
        


        



if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='HP PPI Model Training Script.')
    parser.add_argument('-cpf', '--cpfolder', help='Path to folder containing checkpoint to load', default=None)
    args = parser.parse_args()
    main(args)