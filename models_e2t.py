#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 01:22:14 2020

@author: zhouhan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers_e2t import SpGraphAttentionLayer_e2t, ConvKB

CUDA = torch.cuda.is_available()  # checking cuda availability

class SpGAT_e2t(nn.Module):
    def __init__(self, num_entities, num_types, nfeat1, nfeat2, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat1 -> Entity Input Embedding dimensions
            nfeat2 -> Type Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT_e2t, self).__init__()
        self.dropout = dropout #=0.3
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer_e2t(num_entities, num_types, nfeat1, nfeat2,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) #W_entities200*200正态初始化权重矩阵

        self.out_att = SpGraphAttentionLayer_e2t(num_entities, num_types, nhid * nheads, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed, type_embed,
                edge_list, edge_type, edge_embed, edge_list_nhop):
        x1 = entity_embeddings
        x2 = type_embed


        edge_embed_nhop = torch.LongTensor([])
        
#attention forward
#        x1, x2 = torch.cat([att(x1, x2, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)
#                       for att in self.attentions], dim=1)
        result_1, result_2 = [att(x1, x2, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)
                       for att in self.attentions]
        x1 = torch.cat([result_1[0], result_2[0]], dim=1)
        x2 = torch.cat([result_1[1], result_2[1]], dim=1)                                   

        x1 = self.dropout_layer(x1)
        x2 = self.dropout_layer(x2)

        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]


        x1, x2 = self.out_att(x1, x2, edge_list, edge_embed,
                               edge_list_nhop, edge_embed_nhop)
        x1 = F.elu(x1)
        x2 = F.elu(x2)
        return x1, x2, out_relation_1


class SpKBGATModified_e2t(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, initial_type_emb, entity_out_dim, relation_out_dim, type_out_dim,
                 drop_GAT, alpha, nheads_GAT):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_entities -> number of entities in the Graph
        num_types -> number of types in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''
        super().__init__()

        self.num_entities = initial_entity_emb.shape[0] #=14951
        self.entity_in_dim = initial_entity_emb.shape[1]#=200
        self.entity_out_dim_1 = entity_out_dim[0]#=100
        
        self.nheads_GAT_1 = nheads_GAT[0]#=2
        self.entity_out_dim_2 = entity_out_dim[1]#=200
        self.nheads_GAT_2 = nheads_GAT[1]#=2

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]#=1
        self.relation_dim = initial_relation_emb.shape[1]#=200
        self.relation_out_dim_1 = relation_out_dim[0]#=100
        
        # Properties of types
        self.num_types = initial_type_emb.shape[0]#=3851
        self.type_dim = initial_type_emb.shape[1]#=100
        self.type_in_dim = initial_type_emb.shape[1]#100
        self.type_out_dim_1 = entity_out_dim[0]#=100        

        self.drop_GAT = drop_GAT#=0.3
        self.alpha = alpha# For leaky relu =0.2

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_entities, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))
        
        self.final_type_embeddings = nn.Parameter(
            torch.randn(self.num_types, self.entity_out_dim_1 * self.nheads_GAT_1))        

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)#类型转换为nn.Parameter，使其可学习
        self.type_embeddings = nn.Parameter(initial_type_emb)

        self.sparse_gat_1 = SpGAT_e2t(self.num_entities, self.num_types, self.entity_in_dim, self.type_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)#W_entities200*200正态初始化权重矩阵

        self.W_types = nn.Parameter(torch.zeros(
            size=(self.type_in_dim, self.type_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_types.data, gain=1.414)#W_entities200*200正态初始化权重矩阵


    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop):
        # getting edge list
        edge_list = adj[0] #2*86835 e1+r id e1+e2
        edge_type = adj[1] #86835   e2   id r
        #如果不用2hop,train_indices_nhop应为一个空tensor
        if len(train_indices_nhop) != 0:
            edge_list_nhop = torch.cat(
                (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
            edge_type_nhop = torch.cat(
                [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)
        else:
            edge_list_nhop = Variable(torch.LongTensor([]))
            edge_type_nhop = Variable(torch.LongTensor([]))
            
        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()

        edge_embed = self.relation_embeddings[edge_type] #将edge type里的id对应到embedding去 torch.Size([86835, 50]) e2 embedding

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()#.detach()将 Variable 从创建它的 graph 中分离，把它作为叶子节点

        self.type_embeddings.data = F.normalize(
            self.type_embeddings.data, p=2, dim=1).detach()#.detach()将 Variable 从创建它的 graph 中分离，把它作为叶子节点


        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)
#####第一层layer1_1
        out_entity_1, out_type_1, out_relation_1 = self.sparse_gat_1(
            Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings, self.type_embeddings,
            edge_list, edge_type, edge_embed, edge_list_nhop)
        
        if(CUDA):
            mask_indices = torch.unique(batch_inputs[:, 0]).cuda() #mask_indices为求不重复e2,size为1*39870,batch_inputs为:train_indice
            mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()#mask为1*40943（entity embedding数）的零矩阵
            
            mask_indices_type = torch.unique(batch_inputs[:, 2]).cuda()
            mask_type = torch.zeros(self.type_embeddings.shape[0]).cuda()
        else:
            mask_indices = torch.unique(batch_inputs[:, 0]) #mask_indices为求不重复e2,size为1*39870,batch_inputs为:train_indice
            mask = torch.zeros(self.entity_embeddings.shape[0])#mask为1*40943（entity embedding数）的零矩阵

            mask_indices_type = torch.unique(batch_inputs[:, 2])
            mask_type = torch.zeros(self.type_embeddings.shape[0])
                        
        mask[mask_indices] = 1.0 #size 40943
        mask_type[mask_indices_type] = 1.0
#####第一层layer1_2
        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        types_upgraded = self.type_embeddings.mm(self.W_types)
#第二层layer2        
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)
        
        out_type_1 = types_upgraded + \
            mask_type.unsqueeze(-1).expand_as(out_type_1) * out_type_1
            
        out_type_1 = F.normalize(out_type_1, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity_1.data
        self.final_relation_embeddings.data = out_relation_1.data
        self.final_type_embeddings.data = out_type_1.data

        return out_entity_1, out_type_1, out_relation_1
    
