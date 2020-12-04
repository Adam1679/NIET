#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 23:54:38 2020

@author: zhouhan
"""

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
class SpecialSpmmFunctionFinal_e2t(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N1, N2, E, out_features, sum_type):
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(#创建一个稀疏tensor，格式为COO类型(给出非零元素的坐标形式)
            edge, edge_w, torch.Size([N1, N2, out_features]))#40943*40943*1
        if sum_type == 'out':
            b = torch.sparse.sum(a, dim=1)
            ctx.N = b.shape[0]
            ctx.outfeat = b.shape[1]
            ctx.E = E
            ctx.indices = a._indices()[0, :]
        elif sum_type == 'in':
            b = torch.sparse.sum(a, dim=0)
            ctx.N = b.shape[1]
            ctx.outfeat = b.shape[0]
            ctx.E = E
            ctx.indices = a._indices()[1, :]

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
        return None, grad_values, None, None, None, None, None


class SpecialSpmmFinal_e2t(nn.Module):
    def forward(self, edge, edge_w, N1, N2, E, out_features, sum_type):
        return SpecialSpmmFunctionFinal_e2t.apply(edge, edge_w, N1, N2, E, out_features, sum_type)


class SpGraphAttentionLayer_e2t(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_entities, num_types, entity_in_features, type_in_features, entity_out_features, type_out_features, nrela_dim, dropout, alpha, concat=True):
#        super(SpGraphAttentionLayer, self).__init__()
        super().__init__()
        self.entity_in_features = entity_in_features #=200
        self.type_in_features = type_in_features
        self.entity_out_features = entity_out_features #=100
        self.type_out_features = type_out_features
        self.num_entities = num_entities #=14951
        self.num_types = num_types#3851
        self.alpha = alpha #=0.2
        self.concat = concat #=true
        self.nrela_dim = nrela_dim #=200

        self.a = nn.Parameter(torch.zeros(
            size=(entity_out_features, entity_in_features + type_in_features + nrela_dim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414) #torch.Size([100, 600]) |||out_att:200*600
        self.a_2 = nn.Parameter(torch.zeros(size=(1, entity_out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414) #torch.Size([1, 100])|||out_att:1*200

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final_e2t = SpecialSpmmFinal_e2t()

    def forward(self, x1, x2, edge, edge_embed):
        entity_N = x1.size()[0] #attention:14951 |||out_att:40943
        type_N = x2.size()[0]

        # Self-attention on the nodes - Shared attention mechanism
        #2hop删除
#        edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1)#横着拼
#        edge_embed = torch.cat(
#            (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)#竖着拼

        edge_h = torch.cat(
            (x1[edge[0, :], :], x2[edge[1, :], :], edge_embed[:, :]), dim=1).t()#(e1 id1*86835对应的entity_embedding40943*50=86835*50,e2同理,embeding)
        # edge_h: (2*in_dim + nrela_dim) x E=150*86835  |||out_att:600*86835

        edge_m = self.a.mm(edge_h)
        # edge_m: D * E

        # to be checked later
        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())#取负
        edge_e = torch.exp(powers).unsqueeze(1)#136618*1
        assert not torch.isnan(edge_e).any() #断言，如果edge_e有空就崩溃
        # edge_e: E

#######更新e
        entity_e_rowsum = self.special_spmm_final_e2t(
            edge, edge_e, entity_N, type_N, edge_e.shape[0], 1, 'out')#14951*1
        entity_e_rowsum[entity_e_rowsum == 0.0] = 1e-12#将所有0换为1e-12,疑问：这里是否应该用输入参数
#######更新type
        type_e_rowsum = self.special_spmm_final_e2t(
            edge, edge_e, entity_N, type_N, edge_e.shape[0], 1, 'in')#14951*1
        type_e_rowsum[type_e_rowsum == 0.0] = 1e-12#将所有0换为1e-12,疑问：这里是否应该用输入参数
        
#        e_rowsum = e_rowsum #？？？没看懂这步在干嘛
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)#size从86835*1变为86835

        edge_e = self.dropout(edge_e)#dropout层
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D
#######更新entity        
        entity_h_prime = self.special_spmm_final_e2t(
            edge, edge_w, entity_N, type_N, edge_w.shape[0], self.entity_out_features, 'out')
#        print(entity_h_prime)
        assert not torch.isnan(entity_h_prime).any()
        # h_prime: N x out
        entity_h_prime = entity_h_prime.div(entity_e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(entity_h_prime).any()
#######更新type
        type_h_prime = self.special_spmm_final_e2t(
            edge, edge_w, entity_N, type_N, edge_w.shape[0], self.entity_out_features, 'in')
#        print(type_h_prime)
        assert not torch.isnan(type_h_prime).any()
        # h_prime: N x out
        type_h_prime = type_h_prime.div(type_e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(type_h_prime).any()        
        
        if self.concat:
            # if this layer is not last layer,
            return F.elu(entity_h_prime), F.elu(type_h_prime)
        else:
            # if this layer is last layer,
            return entity_h_prime, type_h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.entity_in_features) + ' -> ' + str(self.entity_out_features) + ')'
