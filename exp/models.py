# -*- coding: utf-8 -*-
"""GNN baseline 模型（与 chain_fraud/models.py 同构，独立副本）
GCN / GraphSAGE / GAT / GIN / GCNII，基于 PyTorch Geometric。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCN2Conv, GCNConv, GINConv, SAGEConv


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
        self.convs.append(SAGEConv(hidden, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GAT(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden * heads, out_dim, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GIN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            inp = in_dim if i == 0 else hidden
            mlp = nn.Sequential(
                nn.Linear(inp, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINConv(mlp))
        self.classifier = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class GCNII(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, alpha=0.1, theta=0.5, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCN2Conv(hidden, alpha=alpha, theta=theta, layer=i + 1))
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x0 = x
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, x0, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


def build_model(name, in_dim, hidden, out_dim, num_layers=2, dropout=0.5):
    name = name.lower()
    if name == "gcn":
        return GCN(in_dim, hidden, out_dim, num_layers, dropout)
    if name == "sage":
        return GraphSAGE(in_dim, hidden, out_dim, num_layers, dropout)
    if name == "gat":
        return GAT(in_dim, hidden, out_dim, num_layers, dropout=dropout)
    if name == "gin":
        return GIN(in_dim, hidden, out_dim, num_layers, dropout)
    if name == "gcnii":
        return GCNII(in_dim, hidden, out_dim, num_layers, dropout=dropout)
    raise ValueError(f"unknown model: {name}")
