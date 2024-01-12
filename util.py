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
#   train_loader: Training DataLoader to save
#   list_rec: Training/Validation results record, 1 entry per epoch (see dataframe setup below for format)
#   last_val_classifi_report: Last classification report on validation set
#   is_train_finished: True/False to indicate training finished. If true, checkpoint foldername will have a special suffix
def save_checkpoint(epoch, model, optimizer, train_loader, list_rec, last_val_classifi_report, is_train_finished=False):
    # Ensure parent checkpoint folder exist
    if not os.path.isdir(cfg.PATH_CHECKPOINTS):
        os.mkdir(cfg.PATH_CHECKPOINTS)
    
    # current timestamp
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    fname_chkptfolder = cfg.CHKPT_FOLDER_NAME_FORMAT.format(timestamp=date_time_str, num_epoch=epoch)
    
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
        cfg.CHKPT_DICTKEY_OPTIM_STATE: optimizer.state_dict(),
    }
    fpath_checkpoint = os.path.join(fpath_chkpoint_folder, cfg.FNAME_CHKPT_PTH)
    torch.save(dict_checkpoint, fpath_checkpoint)
    
    # Save train_loader as a seperate file
    fpath_train_loader = os.path.join(fpath_chkpoint_folder, cfg.FNAME_TRAIN_LOADER_PT)
    torch.save(train_loader, fpath_train_loader)


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
    
    # Path of Training DataLoader file
    fpath_train_loader = os.path.join(fpath_chkpoint_folder, cfg.FNAME_TRAIN_LOADER_PT)
    
    # Fill variables
    epoch = dict_checkpoint[cfg.CHKPT_DICTKEY_EPOCH]
    model_state_dict =  dict_checkpoint[cfg.CHKPT_DICTKEY_MODEL_STATE]
    optim_state_dict = dict_checkpoint[cfg.CHKPT_DICTKEY_OPTIM_STATE]
    train_loader = torch.load(fpath_train_loader)
    list_rec = df_rec.values.tolist()
    
    return epoch, model_state_dict, optim_state_dict, train_loader, list_rec

# Function for writing test result to file
# Arguments
#   loss: Average loss per evaluated data instance
#   classification_report: Classification Report from sklearn.metrics.classification_report
#   is_print_to_console: Print file contents to console too
def write_test_results(fpath_chkpoint_folder, loss, classifi_report, is_print_to_console=True):
    fpath_test_result = os.path.join(fpath_chkpoint_folder, cfg.FNAME_TEST_RESULT_TXT)
    
    # Prepare lines to write
    lines = []
    lines.append("==== Losss ====\n")
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
