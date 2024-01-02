# Data Paths
PATH_VIRUS_FEAT = "./data/movie_feat.csv"
PATH_MOUSE_FEAT = "./data/user_feat.csv"
PATH_INTERACTIONS = "./data/ratings.csv"

# Global Random Seed
RANDOM_SEED = 0

# Data Column Names
# Note: Some temp data columns are hardcoded in code
# Note on Data: mID, vID and label values must be integers that start from 0 and increment by 1 in sequence
COLNAME_MID = "mID"
COLNAME_MUNIPROTID = "mUniProtID"
COLNAME_VID = "vID"
COLNAME_VUNIPROTID  = "vUniProtID"
COLNAME_LABEL = "label"

# Graph Node/Edge Names
NODE_MOUSE = "mouse"
NODE_VIRUS = "virus"
EDGE_INTERACT = "interact"
REV_EDGE_INTERACT = "rev_interact"

# (Data Setup) Train / Val / Test Split
# Train 70%, Validation 20%, Test 10%
DATASPLIT_RATIO_VAL = 0.2
DATASPLIT_RATIO_TEST = 0.1

# (Data Setup) Training Edges Split
# 70% of edges for message passing and 30% of edges for supervision
DATASPLIT_DISJOINT_TRAIN_RATIO = 0.3

# (Data Setup) Indices for Edge Label Tensor
INDEX_EDGE_LABEL_MID = 0
INDEX_EDGE_LABEL_VID = 1


# (Training) Subgraph Sampling
# Index 0: Number of 1-hop neighbours to sample
# Index 1: Number of 2-hop neighbours to sample
# ... etc.
# Applies to each node per iteration
TRAIN_SUBGRAPH_NUM_NEIGHBOURS = [20,10] 

# Batch Sizes
TRAIN_BATCH_SIZE = 128