import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_adj(adj):
    last_dim  = adj.size(-1)
    rowsum = adj.sum(2,keepdim=True).repeat(1,1,last_dim)

def graph_pooling(inputs,num_vertices):
    out = inputs.sum(1)
    return torch.div(out,num_vertices.unsqueeze(-1).expand_as(out))

class DirectedGraphConvolution(nn.Module):
    def __init__(self,in_features,out_features) -> None:
        super(DirectedGraphConvolution,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.zeros((in_features,out_features)))
        self.weight2 = nn.Parameter(torch.zeros((in_features,out_features)))
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight1.data)
        nn.init.xavier_normal_(self.weight2.data)
    
    def forward(self,inputs,adj):
        norm_adj = normalize_adj(adj)
        output1  = F.relu(torch.matmul(normalize_adj,torch.matmul(inputs,self.weight1)))
        inv_norm_adj = normalize_adj(adj.transpose(1,2))
        output2  =  F.relu(torch.matmul(inv_norm_adj,torch.matmul(inputs,self.weight2)))
        out = (output1 + output2) / 2
        out = self.dropout(out)
        return out

class Encoder(nn.Module):
    def __init__(self,inital_hidden=5,gcn_hidden=144,gcn_layers=4,linear_hidden=128):
        super(Encoder,self).__init__()
        self.gcn = [DirectedGraphConvolution(inital_hidden if i==0 else gcn_hidden,gcn_hidden)
                   for i in range(gcn_layers)]
        self.gcn = nn.ModuleList(self.gcn)
    
    def forward(self,inputs):
        adj,out = inputs['adjacency'],inputs['operations']
        gs = adj.size(1)

        adj_with_diag =  normalize_adj(adj+torch.eye(gs,device=adj.device))
        for layer in self.gcn:
            out  = layer(out,adj_with_diag)
