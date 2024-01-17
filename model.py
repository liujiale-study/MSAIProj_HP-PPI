import torch
from torch_geometric.nn import ResGatedGraphConv, GATConv, TransformerConv, GINEConv, to_hetero
from torch_geometric.data import HeteroData
import torch.nn.functional as F

import config as cfg

class GNN(torch.nn.Module):
    def __init__(self, num_hidden_chnls, gnn_op_type):
        super().__init__()
        
        # Graph Neural Network Operator Layers Definition
        self.gnn_op_type = gnn_op_type
        if gnn_op_type == cfg.GNN_OP_ID_RESGATEDGRAPHCONV:
            self.graphOperator1 = ResGatedGraphConv(num_hidden_chnls, num_hidden_chnls, edge_dim=cfg.NUM_FEAT_INTERACTION)
            self.graphOperator2 = ResGatedGraphConv(num_hidden_chnls, num_hidden_chnls, edge_dim=cfg.NUM_FEAT_INTERACTION)
        elif gnn_op_type == cfg.GNN_OP_ID_GAT:
            self.graphOperator1 = GATConv(num_hidden_chnls, num_hidden_chnls, add_self_loops=False, edge_dim=cfg.NUM_FEAT_INTERACTION)
            self.graphOperator2 = GATConv(num_hidden_chnls, num_hidden_chnls, add_self_loops=False, edge_dim=cfg.NUM_FEAT_INTERACTION)
        elif gnn_op_type == cfg.GNN_OP_ID_TRANSFORMERCONV:
            self.graphOperator1 = TransformerConv(num_hidden_chnls, num_hidden_chnls, edge_dim=cfg.NUM_FEAT_INTERACTION)
            self.graphOperator2 = TransformerConv(num_hidden_chnls, num_hidden_chnls, edge_dim=cfg.NUM_FEAT_INTERACTION)
        elif gnn_op_type == cfg.GNN_OP_ID_GINE:
            self.graphOperator1 = GINEConv(torch.nn.Sequential(torch.nn.Linear(num_hidden_chnls, num_hidden_chnls)), edge_dim=cfg.NUM_FEAT_INTERACTION)
            self.graphOperator2 = GINEConv(torch.nn.Sequential(torch.nn.Linear(num_hidden_chnls, num_hidden_chnls)), edge_dim=cfg.NUM_FEAT_INTERACTION)
        else:
            print("Error! Invalid GNN operator type ID specified: "+ str(gnn_op_type) )
            assert False
        
        
        # Batch Norm Layers Definition
        self.batchNorm1 = torch.nn.BatchNorm1d(num_hidden_chnls)
        self.batchNorm2 = torch.nn.BatchNorm1d(num_hidden_chnls)

    def forward(self, x, edge_index, x_edge) -> torch.Tensor:                
        x = self.graphOperator1(x, edge_index, x_edge)
        x = self.batchNorm1(x)
        x = F.relu(x)
        x = F.dropout(x,training=self.training)
        
        x = self.graphOperator2(x, edge_index, x_edge)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = F.dropout(x,training=self.training)

        return x


class PPIVirulencePredictionModel(torch.nn.Module):
    def __init__(self, data_metadata, gnn_op_type=cfg.GNN_OP_DEFAULT):
        super().__init__()
        # Set number of hidden channels
        num_hidden_chnls = cfg.MODEL_HIDDEN_NUM_CHNLS
        
        # Linear Layers for Downsampling
        self.mouse_lin = torch.nn.Linear(cfg.NUM_FEAT_MOUSE, num_hidden_chnls)
        self.virus_lin = torch.nn.Linear(cfg.NUM_FEAT_VIRUS, num_hidden_chnls)
        
        # Instantiate homogeneous GNN:
        self.gnn_op_type = gnn_op_type
        self.gnn = GNN(num_hidden_chnls, gnn_op_type)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data_metadata)

        # Classification head
        self.classifer = torch.nn.Linear(num_hidden_chnls*2, cfg.NUM_PREDICT_CLASSES)

    def forward(self, data: HeteroData) -> torch.Tensor:
        
        # Reduce number of mouse node features
        x_mouse = self.mouse_lin(data[cfg.NODE_MOUSE].x)
        
        # Reduce number of virus node features
        x_virus = self.virus_lin(data[cfg.NODE_VIRUS].x)

        # Dictionary for Node features
        x_dict = {
          cfg.NODE_MOUSE: x_mouse,
          cfg.NODE_VIRUS: x_virus
        }
        
        # Dictionary for Edge Features
        x_edge = {
            (cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS): data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].x,
            (cfg.NODE_VIRUS, cfg.REV_EDGE_INTERACT, cfg.NODE_MOUSE): data[cfg.NODE_VIRUS, cfg.REV_EDGE_INTERACT, cfg.NODE_MOUSE].x,
        }
        
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        # `x_edge` contains feature matrices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict, x_edge)
        
        x_mouse = x_dict[cfg.NODE_MOUSE]
        x_virus = x_dict[cfg.NODE_VIRUS]
        edge_label_index = data[cfg.NODE_MOUSE, cfg.EDGE_INTERACT, cfg.NODE_VIRUS].edge_label_index
        
        # Convert node features to edge features:
        # An edge's features is derived by concatnating the features of the mouse and virus nodes that it connects
        edge_embed_mouse = x_mouse[edge_label_index[0]]
        edge_embed_virus = x_virus[edge_label_index[1]]
        edge_embed = torch.cat((edge_embed_mouse, edge_embed_virus), 1)
        
        out = self.classifer(edge_embed)

        return out

