import os
import pandas as pd
import torch
import config as cfg
from datetime import datetime

def save_checkpoint(epoch, model, optimizer, list_rec, is_train_finished=False):
    # Ensure parent checkpoint folder exist
    if not os.path.isdir(cfg.PATH_CHECKPOINTS):
        os.mkdir(cfg.PATH_CHECKPOINTS)
    
    # current timestamp
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    fname_chkptfolder = date_time_str + "_epoch" + str(epoch)
    
    if is_train_finished:
        fname_chkptfolder += "_fin"

    # Get checkpoint folder path
    fpath_chkpoint_folder = os.path.join(cfg.PATH_CHECKPOINTS, fname_chkptfolder)

    # Create current checkpoint's specific subfolder
    if not os.path.isdir(fpath_chkpoint_folder):
        os.mkdir(fpath_chkpoint_folder)
        
    # Convert metric records to dataframe and output to CSV
    df_rec = pd.DataFrame(list_rec, columns=[cfg.REC_COLNAME_EPOCH, cfg.REC_COLNAME_TRAIN_LOSS, cfg.REC_COLNAME_TRAIN_ACC, 
                               cfg.REC_COLNAME_VAL_LOSS, cfg.REC_COLNAME_VAL_ACC, cfg.REC_COLNAME_VAL_ROCAUC])
    
    fpath_metric_results = os.path.join(fpath_chkpoint_folder, cfg.FNAME_METRIC_RESULT_CSV)
    df_rec.to_csv(fpath_metric_results, index=False)
    
    # Store relevant checkpoint information to dictionary and output to PTH
    dict_checkpoint = {
        cfg.CHKPT_DICTKEY_EPOCH: epoch,
        cfg.CHKPT_DICTKEY_MODEL_STATE: model.state_dict(),
        cfg.CHKPT_DICTKEY_OPTIM_STATE: optimizer.state_dict()
    }
    fpath_checkpoint = os.path.join(fpath_chkpoint_folder, cfg.FNAME_CHKPT_PTH)
    torch.save(dict_checkpoint, fpath_checkpoint)



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
    