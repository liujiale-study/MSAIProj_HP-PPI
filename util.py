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
def save_checkpoint(epoch, model, optimizer, list_rec, last_val_classifi_report, is_train_finished=False):
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
    lines.append("==== Classification Report ====")
    lines.append(str(last_val_classifi_report))
    with open(fpath_classifi_report, 'w') as f:
        f.writelines(lines)
    
    # Store relevant checkpoint information to dictionary and output to PTH
    dict_checkpoint = {
        cfg.CHKPT_DICTKEY_EPOCH: epoch,
        cfg.CHKPT_DICTKEY_MODEL_STATE: model.state_dict(),
        cfg.CHKPT_DICTKEY_OPTIM_STATE: optimizer.state_dict()
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
    optim_state_dict = dict_checkpoint[cfg.CHKPT_DICTKEY_OPTIM_STATE]
    list_rec = df_rec.values.tolist()
    
    return epoch, model_state_dict, optim_state_dict, list_rec

# Function for writing test result to file
# Arguments
#   loss: Average loss per evaluated data instance
#   list_acc: List of accuracy scores, arrange according to the same sequence as RESULT_LINES_BY_CLASS
#   classification_report: Classification Report from sklearn.metrics.classification_report
#   is_print_to_console: Print file contents to console too
def write_test_results(fpath_chkpoint_folder, loss, list_acc, classifi_report, is_print_to_console=True):
    fpath_test_result = os.path.join(fpath_chkpoint_folder, cfg.FNAME_TEST_RESULT_TXT)
    
    # Prepare lines to write
    lines = []
    lines.append("==== Losss ====")
    lines.append("Test Loss:" + str(loss))
    
    lines.append("==== Accuracy ====")
    for i in range(0, len(cfg.RESULT_ACC_LINES_BY_CLASS)):
        lines.append(cfg.RESULT_ACC_LINES_BY_CLASS[i] + str(list_acc[i]) + "%")
    lines.append("")    
    
    lines.append("==== Classification Report ====")
    lines.append(str(classifi_report))
    
    # Write to file
    with open(fpath_test_result, 'w') as f:
        f.writelines(lines)
        
    # Print to console
    if is_print_to_console:
        for line in lines:
            print(line)

# Function for getting list of accuracies
# Arguments
#   cm: Confusion Matrix
# Return
#   list_acc: List of accuracy per class in order of [class 0 acc., class 1 acc., class 2 acc., ..., overall acc], numbers are in percentages
def get_list_acc(cm):
    num_class, _ = cm.shape
    list_acc = [] # accuracy per class, followed by overall accuracy
    for curr_class in range(0, num_class):
        num_tp = cm[curr_class][curr_class] # True Positive Count
        num_tn = 0
        for tmp_gt in range(0, num_class):
            for tmp_pred in range(0, num_class):
                if tmp_gt != curr_class and tmp_pred != curr_class:
                    num_tn += cm[tmp_gt][tmp_pred]

        tmp_acc = (num_tn + num_tp)*100/cm.sum()
        list_acc.append(tmp_acc)
    
    list_acc.append(cm.diagonal().sum()*100/cm.sum())
    
    return list_acc