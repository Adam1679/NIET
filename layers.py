import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable

CUDA = torch.cuda.is_available()

class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):

        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output

#ctx is a context object that can be used to stash information for backward computation
class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(#创建一个稀疏tensor，格式为COO类型(给出非零元素的坐标形式)
            edge, edge_w, torch.Size([N, N, out_features]))#40943*40943*1
        b = torch.sparse.sum(a, dim=1)#40943*1a每行求和
        ctx.N = b.shape[0]#40943
        ctx.outfeat = b.shape[1]#1
        ctx.E = E#86835
        ctx.indices = a._indices()[0, :]#e1 id

        return b.to_dense()#??? 40943*1

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices#e1 id

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None

class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)

class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True):
#        super(SpGraphAttentionLayer, self).__init__()
        super().__init__()
        self.in_features = in_features #=50 model_gat.sparse_gat_1.attentions[0].in_features
        self.out_features = out_features #=100
        self.num_nodes = num_nodes #=40943
        self.alpha = alpha #=0.2
        self.concat = concat #=true
        self.nrela_dim = nrela_dim #=50

        self.a = nn.Parameter(torch.zeros(
            size=(out_features, 2 * in_features + nrela_dim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414) #torch.Size([100, 150]) |||out_att:200*600
        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414) #torch.Size([1, 100])|||out_att:1*200

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge, edge_embed):
        N = input.size()[0] #attention:40943 |||out_att:40943

        # Self-attention on the nodes - Shared attention mechanism
        #edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1)#横着拼
        #edge_embed = torch.cat(
        #    (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)#竖着拼

        edge_h = torch.cat(
            (input[edge[0, :], :], input[edge[1, :], :], edge_embed[:, :]), dim=1).t()#(e1 id1*86835对应的entity_embedding40943*50=86835*50,e2同理,embeding)
        # edge_h: (2*in_dim + nrela_dim) x E=150*86835  |||out_att:600*86835

        edge_m = self.a.mm(edge_h)
        # edge_m: D * E

        # to be checked later
        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())#取负
        edge_e = torch.exp(powers).unsqueeze(1)#86835*1
        assert not torch.isnan(edge_e).any() #断言，如果edge_e有空就崩溃
        # edge_e: E

        e_rowsum = self.special_spmm_final(
            edge, edge_e, N, edge_e.shape[0], 1)#40943*1
        e_rowsum[e_rowsum == 0.0] = 1e-12#将所有0换为1e-12,疑问：这里是否应该用输入参数

#        e_rowsum = e_rowsum #？？？没看懂这步在干嘛
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1) #size从86835*1变为86835

        edge_e = self.dropout(edge_e)#dropout层
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.out_features)
#        print(h_prime)

        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out

        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
