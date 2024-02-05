import argparse
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric import seed_everything
import torch
import torch.nn.functional as F
import pandas as pd
import os
import config as cfg
import model as m
import data_setup
import util



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
        model = m.PPIVirulencePredictionModel(data_metadata=data_metadata, gnn_op_type=model_gnn_op_type, is_output_intermediate_rep=True)
        model.to(device)
        
        # Load best model state dict
        model.load_state_dict(bestmodel_state_dict)
        
        # Grab first batch from test loader
        sampled_data = next(iter(test_loader))
        
        # Input Batch into model
        model.eval()
        with torch.no_grad():                
            sampled_data.to(device)
            pred, intermediate_rep = model(sampled_data)
            
            # Output to file in checkpoint folder
            arr_intermediate_rep = intermediate_rep.cpu().numpy() # convert to Numpy array
            df = pd.DataFrame(arr_intermediate_rep) # convert to a dataframe
            fpath_bestmodel_intermediate_rep = os.path.join(fpath_chkpoint_folder, cfg.FNAME_INTERMEDIATE_REP_BEST_MODEL)
            df.to_csv(fpath_bestmodel_intermediate_rep,index=False) #save to file
            
            print("Intermediate Representations outputted to: " + fpath_bestmodel_intermediate_rep)
        
        
        


        



if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='HP PPI Model Training Script.')
    parser.add_argument('-cpf', '--cpfolder', help='Path to folder containing checkpoint to load', default=None)
    args = parser.parse_args()
    main(args)