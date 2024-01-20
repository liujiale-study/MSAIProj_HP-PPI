import os
import pandas as pd
import torch
import config as cfg
from datetime import datetime

# Function for Saving Checkpoint
# Arguments
#   epoch: Current epoch number
#   model: Model to save
#   optimzer: Optimizer to save
#   list_rec: Training/Validation results record, 1 entry per epoch (see dataframe setup below for format)
#   last_val_classifi_report: Last classification report on validation set
#   is_train_finished: True/False to indicate training finished. If true, checkpoint foldername will have a special suffix
#   bestmodel_state: Model state dictionary of best fit model
#   bestmodel_epoch: Number of training epochs elasped by best fit model
def save_checkpoint(epoch, model, optimizer, list_rec, last_val_classifi_report, bestmodel_state_dict, bestmodel_epoch, is_train_finished=False):
    # Ensure parent checkpoint folder exist
    if not os.path.isdir(cfg.PATH_CHECKPOINTS):
        os.mkdir(cfg.PATH_CHECKPOINTS)
    
    # current timestamp
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    str_gnn_op_type_name = cfg.GNN_OP_TYPE_NAMES[model.gnn_op_type]
    fname_chkptfolder = cfg.CHKPT_FOLDER_NAME_FORMAT.format(timestamp=date_time_str, num_epoch=epoch, gnn_op_type=str_gnn_op_type_name)
    
    if is_train_finished:
        fname_chkptfolder += cfg.CHKPT_FOLDER_SUFFIX_FINISHED

    # Get checkpoint folder path
    fpath_chkpoint_folder = os.path.join(cfg.PATH_CHECKPOINTS, fname_chkptfolder)

    # Create current checkpoint's specific subfolder
    if not os.path.isdir(fpath_chkpoint_folder):
        os.mkdir(fpath_chkpoint_folder)
        
    # Convert metric records to dataframe and output to CSV
    df_rec = pd.DataFrame(list_rec, columns=cfg.REC_COLUMNS)
    fpath_metric_results = os.path.join(fpath_chkpoint_folder, cfg.FNAME_METRIC_RESULT_CSV)
    df_rec.to_csv(fpath_metric_results, index=False)
    
    # Save last classification report
    fpath_classifi_report = os.path.join(fpath_chkpoint_folder, cfg.FNAME_VALIDATION_LAST_REPORT)
    lines=[]
    lines.append("==== Classification Report ====\n")
    lines.append(str(last_val_classifi_report))
    with open(fpath_classifi_report, 'w') as f:
        f.writelines(lines)
    
    # Store relevant checkpoint information to dictionary and output to PTH
    dict_checkpoint = {
        cfg.CHKPT_DICTKEY_EPOCH: epoch,
        cfg.CHKPT_DICTKEY_MODEL_STATE: model.state_dict(),
        cfg.CHKPT_DICTKEY_MODEL_GNN_OP_TYPE: model.gnn_op_type,
        cfg.CHKPT_DICTKEY_OPTIM_STATE: optimizer.state_dict(),
        cfg.CHKPT_DICTKEY_BESTMODEL_STATE_DICT: bestmodel_state_dict,
        cfg.CHKPT_DICTKEY_BESTMODEL_EPOCH: bestmodel_epoch,
    }
    fpath_checkpoint = os.path.join(fpath_chkpoint_folder, cfg.FNAME_CHKPT_PTH)
    torch.save(dict_checkpoint, fpath_checkpoint)


# Function for Loading Checkpoint
# Arguments
#   fpath_chkpoint_folder: Path to folder containing checkpoint
def load_checkpoint(fpath_chkpoint_folder):
    # Load PTH file
    fpath_checkpoint = os.path.join(fpath_chkpoint_folder, cfg.FNAME_CHKPT_PTH)
    dict_checkpoint = torch.load(fpath_checkpoint)
    
    # Load CSV File
    fpath_metric_results = os.path.join(fpath_chkpoint_folder, cfg.FNAME_METRIC_RESULT_CSV)
    df_rec = pd.read_csv(fpath_metric_results)
    
    # Fill variables
    epoch = dict_checkpoint[cfg.CHKPT_DICTKEY_EPOCH]
    model_state_dict =  dict_checkpoint[cfg.CHKPT_DICTKEY_MODEL_STATE]
    model_gnn_op_type = dict_checkpoint[cfg.CHKPT_DICTKEY_MODEL_GNN_OP_TYPE]
    optim_state_dict = dict_checkpoint[cfg.CHKPT_DICTKEY_OPTIM_STATE]
    bestmodel_state_dict = dict_checkpoint[cfg.CHKPT_DICTKEY_BESTMODEL_STATE_DICT]
    bestmodel_epoch = dict_checkpoint[cfg.CHKPT_DICTKEY_BESTMODEL_EPOCH]
    list_rec = df_rec.values.tolist()
    
    return epoch, model_state_dict, model_gnn_op_type, optim_state_dict, list_rec, bestmodel_state_dict, bestmodel_epoch

# Function for writing results on test set to file
# Arguments
#   fpath_test_result: File to write test results to
#   str_headerline: First line to be printed in file
#   loss: Average loss per evaluated data instance
#   classification_report: Classification Report from sklearn.metrics.classification_report
#   is_print_to_console: Print file contents to console too
def write_test_results(fpath_test_result, str_headerline, epoch, loss, classifi_report, is_print_to_console=True):
    
    # Prepare lines to write
    lines = []
    lines.append(str_headerline+"\n")
    lines.append("No. of Epochs Trained: " + str(epoch) + '\n')
    lines.append("\n")
    lines.append("==== Loss ====\n")
    lines.append("Test Loss:" + str(loss) + '\n')
    lines.append("\n")
    
    lines.append("==== Classification Report ====\n")
    lines.append(str(classifi_report))
    
    # Write to file
    with open(fpath_test_result, 'w') as f:
        f.writelines(lines)
        
    # Print to console
    if is_print_to_console:
        for line in lines:
            print(line.rstrip())
