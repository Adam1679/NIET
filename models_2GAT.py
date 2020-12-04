#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 11:34:25 2020

@author: zhouhan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, ConvKB
from layers_e2t import SpGraphAttentionLayer_e2t

CUDA = torch.cuda.is_available()  # checking cuda availability



class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout #=0.3
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
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
        nn.init.xavier_uniform_(self.W.data, gain=1.414) #W_entities50*200正态初始化权重矩阵

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed):
        x = entity_embeddings

#attention forward
        x = torch.cat([att(x, edge_list, edge_embed)
                       for att in self.attentions], dim=1)
        x = self.dropout_layer(x)

        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]

        
        x = F.elu(self.out_att(x, edge_list, edge_embed))
        return x, out_relation_1


class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0] #=40943
        self.entity_in_dim = initial_entity_emb.shape[1]#=50
        self.entity_out_dim_1 = entity_out_dim#=100
        self.nheads_GAT_1 = nheads_GAT[0]#=2
#        self.entity_out_dim_2 = entity_out_dim#=200
#        self.nheads_GAT_2 = nheads_GAT[1]#=2

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]#=11
        self.relation_dim = initial_relation_emb.shape[1]#=50
        self.relation_out_dim_1 = relation_out_dim#=100

        self.drop_GAT = drop_GAT#=0.3
        self.alpha = alpha      # For leaky relu =0.2

#        self.final_entity_embeddings = nn.Parameter(
#            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

#        self.final_relation_embeddings = nn.Parameter(
#            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        #has been converted to nn.parameter in RSGAT model        
#        self.entity_embeddings = nn.Parameter(initial_entity_emb)
#        self.relation_embeddings = nn.Parameter(initial_relation_emb)#类型转换为nn.Parameter，使其可学习

        self.entity_embeddings = initial_entity_emb
        self.relation_embeddings = initial_relation_emb
        
        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)#W_entities50*200正态初始化权重矩阵

    def forward(self, Corpus_, adj, batch_inputs):
        # getting edge list
        edge_list = adj[0] #2*86835 e1+r id
        edge_type = adj[1] #86835   e2   id

        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()


        edge_embed = self.relation_embeddings[edge_type] #将edge type里的id对应到embedding去 torch.Size([86835, 50]) e2 embedding

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()#.detach()将 Variable 从创建它的 graph 中分离，把它作为叶子节点

        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)
#####第一层layer1_1
        out_entity_1, out_relation_1 = self.sparse_gat_1(
            Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings,
            edge_list, edge_type, edge_embed)
        
        if(CUDA):
            mask_indices = torch.unique(batch_inputs[:, 2]).cuda() #mask_indices为求不重复e2,size为1*39870,batch_inputs为:train_indice
            mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()#mask为1*40943（entity embedding数）的零矩阵
        else:
            mask_indices = torch.unique(batch_inputs[:, 2]) #mask_indices为求不重复e2,size为1*39870,batch_inputs为:train_indice
            mask = torch.zeros(self.entity_embeddings.shape[0])#mask为1*40943（entity embedding数）的零矩阵            
        mask[mask_indices] = 1.0 #size 40943
#####第一层layer1_2
        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
#第二层layer2        
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

#        self.final_entity_embeddings.data = out_entity_1.data
#        self.final_relation_embeddings.data = out_relation_1.data

        return out_entity_1, out_relation_1



class SpGAT_e2t(nn.Module):
    def __init__(self, num_entities, num_types, nfeat1, nfeat2, nhid1, nhid2, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat1 -> Entity Input Embedding dimensions
            nfeat2 -> Type Input Embedding dimensions
            nhid1  -> Entity Output Embedding dimensions
            nhid2  -> Type Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT_e2t, self).__init__()
        self.dropout = dropout #=0.3
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer_e2t(num_entities, num_types, nfeat1, nfeat2,
                                                 nhid1, nhid2,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        #self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid1)))
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) #W_entities200*200正态初始化权重矩阵

        self.out_att = SpGraphAttentionLayer_e2t(num_entities, num_types, nhid1 * nheads, nhid2 * nheads,
                                             nheads * nhid1, nheads * nhid2,
                                             nheads * nhid1,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed, type_embed,
                edge_list, edge_type, edge_embed):
        x1 = entity_embeddings
        x2 = type_embed
        
#attention forward
#        x1, x2 = torch.cat([att(x1, x2, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)
#                       for att in self.attentions], dim=1)
        result_1, result_2 = [att(x1, x2, edge_list, edge_embed)
                       for att in self.attentions]
        x1 = torch.cat([result_1[0], result_2[0]], dim=1)
        x2 = torch.cat([result_1[1], result_2[1]], dim=1)                                   

        x1 = self.dropout_layer(x1)
        x2 = self.dropout_layer(x2)

        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]


        x1, x2 = self.out_att(x1, x2, edge_list, edge_embed)
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
        self.entity_out_dim_1 = entity_out_dim#=100
        
        
        self.nheads_GAT_1 = nheads_GAT[0]#=2
#        self.entity_out_dim_2 = entity_out_dim#=200
#        self.nheads_GAT_2 = nheads_GAT[1]#=2

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]#=1
        self.relation_dim = initial_relation_emb.shape[1]#=200
        self.relation_out_dim_1 = relation_out_dim#=100
        
        # Properties of types
        self.num_types = initial_type_emb.shape[0]#=3851
#        self.type_dim = initial_type_emb.shape[1]#=100
        self.type_in_dim = initial_type_emb.shape[1]#100
        self.type_out_dim_1 = type_out_dim#=100        

        self.drop_GAT = drop_GAT#=0.3
        self.alpha = alpha# For leaky relu =0.2

#        self.final_entity_embeddings = nn.Parameter(
#            torch.randn(self.num_entities, self.entity_out_dim_1 * self.nheads_GAT_1))

#        self.final_relation_embeddings = nn.Parameter(
#            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))
        
#        self.final_type_embeddings = nn.Parameter(
#            torch.randn(self.num_types, self.entity_out_dim_1 * self.nheads_GAT_1))        

        #has been converted to nn.parameter in RSGAT model
#        self.entity_embeddings = nn.Parameter(initial_entity_emb)
#        self.relation_embeddings = nn.Parameter(initial_relation_emb)#类型转换为nn.Parameter，使其可学习
#        self.type_embeddings = nn.Parameter(initial_type_emb)

        self.entity_embeddings = initial_entity_emb
        self.relation_embeddings = initial_relation_emb#类型转换为nn.Parameter，使其可学习
        self.type_embeddings = initial_type_emb
        
        self.sparse_gat_1 = SpGAT_e2t(self.num_entities, self.num_types, self.entity_in_dim, self.type_in_dim, self.entity_out_dim_1, self.type_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)#W_entities200*200正态初始化权重矩阵

        self.W_types = nn.Parameter(torch.zeros(
            size=(self.type_in_dim, self.type_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_types.data, gain=1.414)#W_entities200*200正态初始化权重矩阵


    def forward(self, Corpus_, adj, batch_inputs):
        # getting edge list
        edge_list = adj[0] #2*86835 e1+r id e1+e2
        edge_type = adj[1] #86835   e2   id r

            
        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()

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
            edge_list, edge_type, edge_embed)
        
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

#        self.final_entity_embeddings.data = out_entity_1.data
#        self.final_relation_embeddings.data = out_relation_1.data
#        self.final_type_embeddings.data = out_type_1.data

        return out_entity_1, out_type_1, out_relation_1


class multiRSGAT(nn.Module):
    def __init__(self, input_entity_embeddings, input_relation_embeddings, input_type_embeddings, input_rdf_relation_embeddings, args):
        '''
        model_gat_e2t out_dim: entity 100, rdf 100, type 50
        model_gat_ere out_dim: entity 100, relation 100
        model_gat_trt out_dim: type 50, relation 50
        '''
        
        super().__init__()
        
        self.num_entities = input_entity_embeddings.shape[0]
        self.num_relation = input_relation_embeddings.shape[0]
        self.num_types = input_type_embeddings.shape[0]
        self.num_rdf_relation = input_rdf_relation_embeddings.shape[0]
        
        self.out_long_dim = args.out_long_dim
        self.out_short_dim = args.out_short_dim
        self.drop_GAT = args.drop_GAT
        self.alpha = args.alpha
        self.nheads_GAT = args.nheads_GAT
        
        self.input_entity_embeddings = nn.Parameter(input_entity_embeddings)
        self.input_relation_embeddings = nn.Parameter(input_relation_embeddings)
        self.input_type_embeddings = nn.Parameter(input_type_embeddings)
#        self.input_rdf_relation_embeddings = nn.Parameter(input_rdf_relation_embeddings)        
        
        self.output_entity_embeddings_short = nn.Parameter(
            torch.randn(self.num_entities, self.nheads_GAT[0]*self.out_short_dim))
        self.output_entity_embeddings_long = nn.Parameter(
            torch.randn(self.num_entities, self.nheads_GAT[0]*self.out_long_dim))
        self.output_relation_embeddings_short = nn.Parameter(
            torch.randn(self.num_relation, self.nheads_GAT[0]*self.out_short_dim))
        self.output_relation_embeddings_long = nn.Parameter(
            torch.randn(self.num_relation, self.nheads_GAT[0]*self.out_long_dim))
        self.output_type_embedding = nn.Parameter(
            torch.randn(self.num_types, self.nheads_GAT[0]*self.out_short_dim))

        
        
        
#        self.model_gat_e2t = SpKBGATModified_e2t(self.input_entity_embeddings, self.input_rdf_relation_embeddings, self.input_type_embeddings, 
#                                    self.out_short_dim, self.out_short_dim, self.out_short_dim, self.drop_GAT, self.alpha, self.nheads_GAT)
        self.model_gat_ere = SpKBGATModified(self.input_entity_embeddings, self.input_relation_embeddings, self.out_long_dim, self.out_long_dim,
                                self.drop_GAT, self.alpha, self.nheads_GAT)
        self.model_gat_trt = SpKBGATModified(self.input_type_embeddings, self.input_relation_embeddings, self.out_short_dim, self.out_short_dim,
                                self.drop_GAT, self.alpha, self.nheads_GAT)
        
        
        self.M = nn.Parameter(torch.zeros(
            size=(self.nheads_GAT[0]*self.out_long_dim, self.nheads_GAT[0]*self.out_short_dim)))
        nn.init.xavier_uniform_(self.M.data, gain=1.414)        
        
#        self.W_e2long = nn.Parameter(torch.zeros(
#            size=(self.nheads_GAT[0]*(self.out_long_dim + self.out_short_dim), self.out_long_dim)))
#        nn.init.xavier_uniform_(self.W_e2long.data, gain=1.414)
#        self.W_e2short = nn.Parameter(torch.zeros(
#            size=(self.nheads_GAT[0]*(self.out_long_dim + self.out_short_dim), self.out_short_dim)))
#        nn.init.xavier_uniform_(self.W_e2short.data, gain=1.414)
#        self.W_r2long = nn.Parameter(torch.zeros(
#            size=(self.nheads_GAT[0]*(self.out_long_dim + self.out_short_dim), self.out_long_dim)))
#        nn.init.xavier_uniform_(self.W_r2long.data, gain=1.414)
#        self.W_r2short = nn.Parameter(torch.zeros(
#            size=(self.nheads_GAT[0]*(self.out_long_dim + self.out_short_dim), self.out_short_dim)))
#        nn.init.xavier_uniform_(self.W_r2short.data, gain=1.414)
#        self.W_t2t = nn.Parameter(torch.zeros(
#            size=(self.nheads_GAT[0]*(self.out_short_dim + self.out_short_dim), self.out_short_dim)))
#        nn.init.xavier_uniform_(self.W_r2short.data, gain=1.414)
#        self.W_rdf = nn.Parameter(torch.zeros(
#            size=(self.nheads_GAT[0]*self.out_short_dim, self.out_short_dim)))
#        nn.init.xavier_uniform_(self.W_rdf.data, gain=1.414)
        
#        self.linear_e2long = nn.Linear(in_features = self.nheads_GAT[0]*(self.out_long_dim + self.out_short_dim), out_features = self.out_long_dim)
#        self.linear_e2short = nn.Linear(in_features = self.nheads_GAT[0]*(self.out_long_dim + self.out_short_dim), out_features = self.out_short_dim)
#        self.linear_r2long = nn.Linear(in_features = self.nheads_GAT[0]*(self.out_long_dim + self.out_short_dim), out_features = self.out_long_dim)
#        self.linear_r2short = nn.Linear(in_features = self.nheads_GAT[0]*(self.out_long_dim + self.out_short_dim), out_features = self.out_short_dim)
#        self.linear_t2t = nn.Linear(in_features = self.nheads_GAT[0]*(self.out_short_dim + self.out_short_dim), out_features = self.out_short_dim)
#        self.linear_rdf = nn.Linear(in_features = self.nheads_GAT[0]*self.out_short_dim, out_features = self.out_short_dim)
        

        
    def forward(self, e2t_Corpus_, ere_Corpus_, trt_Corpus_, train_indices_e2t, train_indices_ere, train_indices_trt):
        
#        entity_embed_e2t, type_embed_e2t, rdf_relation_embed_e2t = self.model_gat_e2t(
#            e2t_Corpus_, e2t_Corpus_.train_adj_matrix, train_indices_e2t)
        entity_embed_ere, relation_embed_ere = self.model_gat_ere(
            ere_Corpus_, ere_Corpus_.train_adj_matrix, train_indices_ere)
        type_embed_trt, relation_embed_trt = self.model_gat_trt(
            trt_Corpus_, trt_Corpus_.train_adj_matrix, train_indices_trt)
        
        out_entity_embed_short = entity_embed_ere.mm(self.M)
        
#        print('here:', 
#              entity_embed_ere.size(), type_embed_trt.size(), relation_embed_ere.size(),
#              '\n', relation_embed_trt.size(), self.M.size())        

#        union_entity_embed = torch.cat(
#            [entity_embed_e2t, entity_embed_ere], dim=1)
#        union_type_embed = torch.cat(
#            [type_embed_e2t, type_embed_trt], dim=1)
#        union_relation_embed = torch.cat(
#            [relation_embed_ere, relation_embed_trt], dim=1)
            
        
#        out_entity_embed_long = F.elu(union_entity_embed.mm(self.W_e2long))
#        out_entity_embed_short = F.elu(union_entity_embed.mm(self.W_e2short))
#        out_relation_embed_long = F.elu(union_relation_embed.mm(self.W_r2long))
#        out_relation_embed_short = F.elu(union_relation_embed.mm(self.W_r2short))
#        out_rdf_embed = F.elu(rdf_relation_embed_e2t.mm(self.W_rdf))
#        out_type_embed = F.elu(union_type_embed.mm(self.W_t2t))

#        out_entity_embed_long = F.elu(self.linear_e2long(union_entity_embed))
#        out_entity_embed_short = F.elu(self.linear_e2short(union_entity_embed))
#        out_relation_embed_long = F.elu(self.linear_r2long(union_relation_embed))
#        out_relation_embed_short = F.elu(self.linear_r2short(union_relation_embed))
#        out_rdf_embed = F.elu(self.linear_rdf(rdf_relation_embed_e2t))
#        out_type_embed = F.elu(self.linear_t2t(union_type_embed))

#        self.output_entity_embeddings.data = entity_embed_ere.data
#        self.output_relation_embeddings_ere.data = relation_embed_ere.data
#        self.output_relation_embeddings_trt.data = relation_embed_trt.data
#        self.output_type_embeddings.data = type_embed_trt.data
        
#        return entity_embed_ere, relation_embed_ere, relation_embed_trt, type_embed_trt, self.M

        
        self.output_entity_embeddings_short.data = out_entity_embed_short.data
        self.output_entity_embeddings_long.data = entity_embed_ere.data
        self.output_relation_embeddings_short.data = relation_embed_trt.data
        self.output_relation_embeddings_long.data = relation_embed_ere.data
        self.output_type_embedding.data = type_embed_trt.data
        #self.output_rdf_embedding.data = out_rdf_embed.data
        
        return entity_embed_ere, out_entity_embed_short, relation_embed_ere, relation_embed_trt, type_embed_trt


    
class SpKBGATConvOnly(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
#        self.entity_out_dim_2 = entity_out_dim[1]
#        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha      # For leaky relu
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.convKB = ConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 3, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv)

    def forward(self, Corpus_, adj, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

