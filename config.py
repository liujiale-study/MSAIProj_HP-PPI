# Data Paths
PATH_VIRUS_FEAT = "./data/virus_proteins.csv"
PATH_MOUSE_FEAT = "./data/mouse_proteins.csv"
PATH_INTERACTIONS = "./data/pp_interactions.csv"
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

# Number of input channels for model's hidden layers
MODEL_HIDDEN_NUM_CHNLS=512
# Subgraph Sampling
# Index 0: Number of 1-hop neighbours to sample
# Index 1: Number of 2-hop neighbours to sample
# ... etc.
# Applies to each node per iteration
SUBGRAPH_NUM_NEIGHBOURS = [50,50] 

# Batch Sizes
TRAIN_BATCH_SIZE = 1024
VAL_BATCH_SIZE = 2048
TEST_BATCH_SIZE = 2048

# (Training) Optimizer Params
ADAMW_LR = 0.001
ADAMW_WEIGHT_DECAY = 0.01

# Number of Training Epochs
NUM_EPOCHS = 200

# Graph Neural Network Operator Types 
# IDs must be unique among all operator types
GNN_OP_ID_RESGATEDGRAPHCONV = 0 # Type ID for ResGatedGraphConv
GNN_OP_ID_GAT = 1

# The type that will be used by Default
GNN_OP_DEFAULT = GNN_OP_ID_RESGATEDGRAPHCONV

# String names of GNN Operator
# Use for printing to console or labeling checkpoint folders
# Name of operator with ID 0 = this array's index 0, name of operator with ID 1 = this array's index 1, etc.
GNN_OP_TYPE_NAMES = ["ResGatedGraphConv", "GAT"]


# =============================================



# === Checkpointing and Train/Val/Test Output ===

# Checkpoint at Every X Epoch
# Note: Program will save after final epoch
CHKPOINT_EVERY_NUM_EPOCH = 10

# Training Record Datafram Column Names
REC_COLNAME_EPOCH = "Epoch"
REC_COLNAME_TRAIN_LOSS = "Train Loss"
REC_COLNAME_TRAIN_F1_CLASS_NON_INTERACTING = "Train F1 Non-Interacting"
REC_COLNAME_TRAIN_F1_CLASS_LOW = "Train F1 Low"
REC_COLNAME_TRAIN_F1_CLASS_INTERMEDIATE = "Train F1 Intermediate"
REC_COLNAME_TRAIN_F1_CLASS_HIGH = "Train F1. High"
REC_COLNAME_TRAIN_ACC_OVERALL = "Train Acc. Overall"
REC_COLNAME_VAL_LOSS = "Val. Loss"
REC_COLNAME_VAL_F1_CLASS_NON_INTERACTING = "Val F1 Non-Interacting"
REC_COLNAME_VAL_F1_CLASS_LOW = "Val. F1 Low"
REC_COLNAME_VAL_F1_CLASS_INTERMEDIATE = "Val. F1 Intermediate"
REC_COLNAME_VAL_F1_CLASS_HIGH = "Val. F1 High"
REC_COLNAME_VAL_ACC_OVERALL = "Val. Acc. Overall"

REC_COLUMNS = [REC_COLNAME_EPOCH, REC_COLNAME_TRAIN_LOSS, 
                REC_COLNAME_TRAIN_F1_CLASS_LOW,
                REC_COLNAME_TRAIN_F1_CLASS_INTERMEDIATE,
                REC_COLNAME_TRAIN_F1_CLASS_HIGH,        
                REC_COLNAME_TRAIN_ACC_OVERALL,                                  
                REC_COLNAME_VAL_LOSS, 
                REC_COLNAME_VAL_F1_CLASS_LOW, 
                REC_COLNAME_VAL_F1_CLASS_INTERMEDIATE,
                REC_COLNAME_VAL_F1_CLASS_HIGH,
                REC_COLNAME_VAL_ACC_OVERALL]
assert len(REC_COLUMNS) == 5 + NUM_PREDICT_CLASSES * 2

# Class Labels for Classification Report
CLASSIFICATION_REPORT_CLASS_LABELS = ["Low", "Intermediate", "High"]
assert len(CLASSIFICATION_REPORT_CLASS_LABELS) == NUM_PREDICT_CLASSES

# Checkpoint Filenames
FNAME_CHKPT_PTH = "checkpoint.pth"
FNAME_METRIC_RESULT_CSV = "metric_results.csv"
FNAME_VALIDATION_LAST_REPORT = "last_validationset_classification_report.txt"
FNAME_TEST_RESULT_TXT = "test_results.txt"


# Checkpoint Dictionary Properties
CHKPT_DICTKEY_EPOCH = "epoch"
CHKPT_DICTKEY_MODEL_STATE = "model_state"
CHKPT_DICTKEY_OPTIM_STATE = "optim_state"
CHKPT_DICTKEY_MODEL_GNN_OP_TYPE = "model_gnn_op_type"

# Checkpoint Folder Name Format
CHKPT_FOLDER_NAME_FORMAT = "{timestamp}_epoch{num_epoch}_{gnn_op_type}"
CHKPT_FOLDER_SUFFIX_FINISHED = "_fin"