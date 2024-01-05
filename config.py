# Data Paths
PATH_VIRUS_FEAT = "./data/virus_nodes.csv"
PATH_MOUSE_FEAT = "./data/mouse_nodes.csv"
PATH_INTERACTIONS = "./data/interaction_edges.csv"
PATH_CHECKPOINTS = "./checkpoints"

# Global Random Seed
RANDOM_SEED = 0

# === Data Files' Properties and HeteroData Setup ===

# Data Column Names
# Note: Some temp data columns are hardcoded in code
# Note on Data: mID, vID and label values must be integers that start from 0 and increment by 1 in sequence
COLNAME_MID = "mID"
COLNAME_MUNIPROTID = "mUniProtID"
COLNAME_VID = "vID"
COLNAME_VUNIPROTID  = "vUniProtID"
COLNAME_LABEL = "virulence_class"
COLNAME_INTERACT_PROB = "interact_prob"

# Number of Data Features
NUM_FEAT_MOUSE = 2049       # Size of feature-vector for mouse
NUM_FEAT_VIRUS = 2048       # Size of feature-vector for virus
NUM_FEAT_INTERACTION = 1    # Size of feature-vector for interactions

# Number of Classes in Model Prediction
NUM_PREDICT_CLASSES = 3 

# Graph Node/Edge Names
NODE_MOUSE = "mouse"
NODE_VIRUS = "virus"
EDGE_INTERACT = "interact"
REV_EDGE_INTERACT = "rev_interact"

# Train / Val / Test Split
# Train 70%, Validation 20%, Test 10%
DATASPLIT_RATIO_VAL = 0.2
DATASPLIT_RATIO_TEST = 0.1

# Training Edges Split
# 70% of edges for message passing and 30% of edges for supervision
DATASPLIT_DISJOINT_TRAIN_RATIO = 0.3

# Indices for Edge Label Tensor
INDEX_EDGE_LABEL_MID = 0
INDEX_EDGE_LABEL_VID = 1

# =============================================


# === Model Setup and Train/Val/Test Loop ===

# Subgraph Sampling
# Index 0: Number of 1-hop neighbours to sample
# Index 1: Number of 2-hop neighbours to sample
# ... etc.
# Applies to each node per iteration
SUBGRAPH_NUM_NEIGHBOURS = [20,10] 

# Batch Sizes
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 128

# (Training) Optimizer Params
ADAMW_LR = 0.001
ADAMW_WEIGHT_DECAY = 0.01

# Number of Training Epochs
NUM_EPOCHS = 6

# =============================================



# === Checkpointing and Train/Val Output ===

# Checkpoint at Every X Epoch
# Note: Program will save after final epoch
CHKPOINT_EVERY_NUM_EPOCH = 3

# Training Record Datafram Column Names
REC_COLNAME_EPOCH = "epoch"
REC_COLNAME_TRAIN_LOSS = "train_loss"
REC_COLNAME_TRAIN_ACC = "train_acc"
REC_COLNAME_VAL_LOSS = "val_loss"
REC_COLNAME_VAL_ACC = "val_acc"
REC_COLNAME_VAL_ROCAUC = "val_roc_auc"

# Checkpoint Filenames
FNAME_CHKPT_PTH = "checkpoint.pth"
FNAME_METRIC_RESULT_CSV = "metric_results.csv"

# Checkpoint Dictionary Properties
CHKPT_DICTKEY_EPOCH = "epoch"
CHKPT_DICTKEY_MODEL_STATE = "model_state"
CHKPT_DICTKEY_OPTIM_STATE = "optim_state"