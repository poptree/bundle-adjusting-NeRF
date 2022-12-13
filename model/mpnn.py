import torch
import torch.functional as F
import numpy as np
import torch.nn as nn
from torch_scatter import scatter 
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d as BN, Dropout

class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim:int, activation: str="relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
            

class base_MLP(nn.Module):
    def __init__(self, layers=8, skips=[4], in_channels=32, inner_channels=32, out_channels=32, activate: str="relu"):
        super().__init__()
        
        self.layers = layers
        self.skips = skips
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.out_channels = out_channels
        self.mlps = nn.ModuleList()

        self.mlps.append(DenseLayer(in_channels, inner_channels))
        in_ch = inner_channels
        out_ch = inner_channels
        for i in range(1, layers-1):
            if i in skips:
                in_ch += in_channels
            self.mlps.append(DenseLayer(in_ch, out_ch))
            in_ch = out_ch
        self.mlps.append(DenseLayer(in_ch, out_channels))

        if activate == "relu":
            self.activation = torch.nn.ReLU()
        elif activate == "leakyrelu":
            self.activation = torch.nn.LeakyReLU()
        else:
            self.activation = lambda x: x

    def forward(self, input):
        x = input
        for i, l in enumerate(self.mlps):
            if i in self.skips:
                x = torch.cat([x, input], dim=-1)
            x = l(x)
            if i != self.layers - 1:
                x = self.activation(x)
        return x
    

class BaseEdgeConv(nn.Module):
    def __init__(self, node_channels, edge_channels, out_channels):
        super(BaseEdgeConv, self).__init__()
        self.mlp = Seq(Linear(2*node_channels+edge_channels, out_channels),
                       torch.nn.ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        W = torch.cat([torch.cat([x_i, x_j], dim=1), edge_attr],
                      dim=1)  # tmp has shape [E, 2 * in_channels]
        edge_out = self.mlp(W)
        node_out = scatter(
            edge_out, edge_index[0], dim=0, dim_size=x.shape[0], reduce="mean")
        return node_out, edge_out



class MPNN_3Conv(torch.nn.Module):
    def __init__(self, node_channels=32, edge_channels=32, graph_channels=32, out_channels=32, activate="leakyrelu") -> None:
        super().__init__()

        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.graph_channels = graph_channels
        self.out_channels = out_channels

        
        self.conv1 = BaseEdgeConv(self.node_channels, self.edge_channels, self.graph_channels)
        self.conv2 = BaseEdgeConv(self.graph_channels, self.graph_channels + self.edge_channels, self.graph_channels)
        self.conv3 = BaseEdgeConv(self.graph_channels, self.graph_channels * 2 + self.edge_channels, self.graph_channels)

        # self.ac = torch.nn.LeakyReLU(negative_slope=0.2)
        self.ac = lambda x: x
        if activate == "relu":
            self.ac = nn.ReLU()
        elif activate == "leakyrelu":
            self.ac = nn.LeakyReLU()


        self.edge_out_mlp = Seq(Linear(self.graph_channels, graph_channels),
                       torch.nn.LeakyReLU(negative_slope=0.2),
                       Linear(graph_channels, graph_channels),
                       torch.nn.LeakyReLU(negative_slope=0.2),
                       Linear(graph_channels, self.out_channels))
        self.node_out_mlp = Seq(Linear(self.graph_channels, graph_channels),
                       torch.nn.LeakyReLU(negative_slope=0.2),
                       Linear(graph_channels, graph_channels),
                       torch.nn.LeakyReLU(negative_slope=0.2),
                       Linear(graph_channels, self.out_channels))

    def forward(self, node_attr, edge_attr, edge_index, node_shape=None):
        # node_attr = torch.zeros_like(node_attr)
        node_1, edge_1 = self.conv1(node_attr, edge_index, edge_attr)
        node_1 = self.ac(node_1)
        edge_1 = self.ac(edge_1)

        node_2, edge_2 = self.conv2(node_1, edge_index, torch.cat([edge_1, edge_attr], dim=-1))
        node_2 = self.ac(node_2)
        edge_2 = self.ac(edge_2)

        node_3, edge_3 = self.conv3(node_2, edge_index, torch.cat([edge_2, edge_1, edge_attr], dim=-1))
        node_3 = self.ac(node_3)
        edge_3 = self.ac(edge_3)

        return self.node_out_mlp(node_3), self.edge_out_mlp(edge_3)
    
# class MPNN_3Conv_node(torch.nn.Module):
#     def __init__(self, node_channels=32, edge_channels=32, graph_channels=32, out_channels=32, activate="leakyrelu") -> None:
#         super().__init__()

#         self.node_channels = node_channels
#         self.edge_channels = edge_channels
#         self.graph_channels = graph_channels
#         self.out_channels = out_channels

        
#         self.conv1 = BaseEdgeConv(self.node_channels, self.edge_channels, self.graph_channels)
#         self.conv2 = BaseEdgeConv(self.graph_channels, self.graph_channels + self.edge_channels, self.graph_channels)
#         self.conv3 = BaseEdgeConv(self.graph_channels, self.graph_channels * 2 + self.edge_channels, self.graph_channels)

#         # self.ac = torch.nn.LeakyReLU(negative_slope=0.2)
#         self.ac = lambda x: x
#         if activate == "relu":
#             self.ac = nn.ReLU()
#         elif activate == "leakyrelu":
#             self.ac = nn.LeakyReLU()


#         self.edge_out_mlp = Seq(Linear(self.graph_channels, graph_channels),
#                        torch.nn.LeakyReLU(negative_slope=0.2),
#                        Linear(graph_channels, graph_channels),
#                        torch.nn.LeakyReLU(negative_slope=0.2),
#                        Linear(graph_channels, self.out_channels))
#         self.node_out_mlp = Seq(Linear(self.graph_channels, graph_channels),
#                        torch.nn.LeakyReLU(negative_slope=0.2),
#                        Linear(graph_channels, graph_channels),
#                        torch.nn.LeakyReLU(negative_slope=0.2),
#                        Linear(graph_channels, self.out_channels))

#     def forward(self, node_attr, edge_attr, edge_index, node_shape=None):
#         # node_attr = torch.zeros_like(node_attr)
#         node_1, edge_1 = self.conv1(node_attr, edge_index, edge_attr)
#         node_1 = self.ac(node_1)
#         edge_1 = self.ac(edge_1)

#         node_2, edge_2 = self.conv2(node_1, edge_index, torch.cat([edge_1, edge_attr], dim=-1))
#         node_2 = self.ac(node_2)
#         edge_2 = self.ac(edge_2)

#         node_3, edge_3 = self.conv3(node_2, edge_index, torch.cat([edge_2, edge_1, edge_attr], dim=-1))
#         node_3 = self.ac(node_3)
#         edge_3 = self.ac(edge_3)

#         return self.node_out_mlp(node_3), self.edge_out_mlp(edge_3)
    