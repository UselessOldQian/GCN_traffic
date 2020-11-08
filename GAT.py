# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903 
    Graph Attention Layer
    """
    def __init__(self, in_c, out_c, alpha=0.01):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c   # The node represents the number of input features of a vector
        self.out_c = out_c   # The node represents the number of output features of a vector
        self.alpha = alpha     # leakyrelu parameter
        
        # trainable parameter, W and a in the thesis
        self.W = nn.Parameter(torch.zeros(size=(in_c, out_c)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # initialization
        self.a = nn.Parameter(torch.zeros(size=(2*out_c, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   # initialization
        
        # use leakyrelu as activation function
        self.leakyrelu = nn.LeakyReLU(self.alpha) #while x<0, alpha*x
    
    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features represents the number of input eigenvector elements of a node the input data, [B, N, C]
        adj: adjacency matrix  [N, N] 0 or 1
        """
        B, N = inp.size(0), inp.size(1)
        adj = adj + torch.eye(N, dtype=adj.dtype) #A+I,
        h = torch.matmul(inp, self.W)   # [B,N,out_features] ,use matmul to ensure that dimensions do not collapse

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_c)
        # [B, N, N, 2 * out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        # [B,N, N, 1] => [B, N, N] The correlation coefficient of graph attention (not normalized)
        
        zero_vec = -1e12 * torch.ones_like(e)    # The unconnected edge is set to be a 1-matrix whose shape is consistent with E

        attention = torch.where(adj > 0, e, zero_vec)   # [B,N, N]
        #
        # If the adjacency matrix element is greater than 0,
        # the two nodes are connected, and the attention coefficient of this position is retained.
        # Otherwise, mask concatenation is required to be very small,
        # because this minimum value will not be considered in softmax.
        attention = F.softmax(attention, dim=2)    # softmax keep the shape unchanged [N, N]，get normalized attention weight
        #attention = F.dropout(attention, self.dropout, training=self.training)   # dropout，prevent overfitting
        h_prime = torch.matmul(attention, h)  # [B,N, N].[N, out_features] => [B,N, out_features]
        # Get a representation updated by the attention weights of surrounding nodes
        return h_prime  
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT_model(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param out_c: int, number of output channels.
        :param K:
        """
        super(GAT_model, self).__init__()
        self.conv1 = GraphAttentionLayer(in_c, hid_c)
        self.conv2 = GraphAttentionLayer(hid_c, out_c)
        self.act = nn.ReLU()

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  # [N, N]
        flow_x = data["flow_x"].to(device)  # [B, N, H, D]
        B, N = flow_x.size(0), flow_x.size(1)
        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]
        output_1 = self.act(self.conv1(flow_x, graph_data))
        output_2 = self.act(self.conv2(output_1, graph_data))

        return output_2.unsqueeze(2)#[B,1,N,1]
