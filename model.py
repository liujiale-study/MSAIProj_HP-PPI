import torch
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
import torch.nn.functional as F

import config as cfg

class GNN(torch.nn.Module):
    def __init__(self, num_hidden_chnls):
        super().__init__()

        self.conv1 = SAGEConv(num_hidden_chnls, num_hidden_chnls)
        self.conv2 = SAGEConv(num_hidden_chnls, num_hidden_chnls)

    def forward(self, x, edge_index) -> torch.Tensor:        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x


class HP_PPI_Model(torch.nn.Module):
    def __init__(self, num_hidden_chnls, data_metadata):
        super().__init__()
        
        self.mouse_lin = torch.nn.Linear(cfg.NUM_FEAT_MOUSE, num_hidden_chnls)
        self.virus_lin = torch.nn.Linear(cfg.NUM_FEAT_VIRUS, num_hidden_chnls)
        
        # Instantiate homogeneous GNN:
        self.gnn = GNN(num_hidden_chnls)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data_metadata)

        # Classification head
        self.classifer = torch.nn.Linear(num_hidden_chnls*2, cfg.NUM_PREDICT_CLASSES)

    def forward(self, data: HeteroData) -> torch.Tensor:
        
        x_dict = {
          cfg.NODE_MOUSE: self.mouse_lin(data[cfg.NODE_MOUSE].x),
          cfg.NODE_VIRUS: self.virus_lin(data[cfg.NODE_VIRUS].x)
        }
        
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        # `x_edge` contains edge features
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        
        x_mouse = x_dict[cfg.NODE_MOUSE]
        x_virus = x_dict[cfg.NODE_VIRUS]
        edge_label_index = data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label_index
        
        # Convert node embeddings to edge-level representations:
        edge_embed_mouse = x_mouse[edge_label_index[0]]
        edge_embed_virus = x_virus[edge_label_index[1]]
        edge_embed = torch.cat((edge_embed_mouse, edge_embed_virus), 1)
        
        
        out = self.classifer(edge_embed)

        
        return out

