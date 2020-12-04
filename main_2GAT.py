import torch

from models_2GAT import SpKBGATModified, multiRSGAT, SpKBGATConvOnly
#from models_e2t import SpKBGATModified_e2t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy
from evaluator_anxiangz import Type_Evaluator, WrapperModel, Type_Evaluator_trt, WrapperModel2
from preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch import Corpus
from create_batch_e2t import Corpus_e2t
from utils import save_model

import random
import argparse
import os
import sys
import logging
import time
import pickle

# %%
# %%from torchviz import make_dot, make_dot_from_trace
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.addHandler(logging.FileHandler(filename="./rngat.l2.log", mode='w'))

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="./data/FB15k/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=8000, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-en_emb_size", "--entity_embedding_size", type=int,
                      default=100, help="Size of entity embeddings (if pretrained not used)")
    args.add_argument("-ty_emb_size", "--type_embedding_size", type=int,
                      default=50, help="Size of relation embeddings (if pretrained not used)")    
    args.add_argument("-l", "--lr", type=float, default=1e-2)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=False)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/fb/out/", help="Folder name to save the models.")

    # arguments for e2tGAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=136618, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float, 
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[50, 100], help="Entity output embedding dimensions")
    args.add_argument("-out_dim_long", "--out_long_dim", type=int,
                      default=100, help="Longer output embedding dimensions")
    args.add_argument("-out_dim_short", "--out_short_dim", type=int,
                      default=50, help="Shorter output embedding dimensions")    
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=1, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=500,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.0, help="Dropout probability for convolution layer")

    args = args.parse_args()
    return args


args = parse_args()
# %%


def load_data(args):
    train_data_e2t, validation_data_e2t, test_data_e2t, \
    train_data_ere, validation_data_ere, test_data_ere, \
    train_data_trt, validation_data_trt, test_data_trt, \
    entity2id, relation2id, type2id, headTailSelector, unique_entities_train_e2t, unique_types_train_e2t,\
    unique_entities_train_ere, unique_types_train_trt= build_data(
            args.data, is_unweigted=False, directed=True)

    if args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'),
                                                                 os.path.join(args.data, 'relation2vec.txt'))
        print("Initialised relations and entities from TransE")

    else:
        entity_embeddings = np.random.randn(
            len(entity2id), args.entity_embedding_size)
        relation_embeddings = np.random.randn(
                len(relation2id), args.entity_embedding_size)
        rdf_relation_embeddings = np.random.randn(
            1, args.type_embedding_size)
        type_embeddings = np.random.randn(
            len(type2id), args.type_embedding_size)   
#        entity2typeMat = np.random.randn(
#            args.type_embedding_size, args.entity_embedding_size)
        print("Initialised relations and entities randomly")

    corpus_e2t = Corpus_e2t(args, train_data_e2t, validation_data_e2t, test_data_e2t, entity2id, type2id,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train_e2t, unique_types_train_e2t, args.get_2hop)
    
    corpus_ere = Corpus(args, train_data_ere, validation_data_ere, test_data_ere, entity2id, relation2id, headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train_ere, args.get_2hop)    

    corpus_trt = Corpus(args, train_data_trt, validation_data_trt, test_data_trt, type2id, relation2id, headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_types_train_trt, args.get_2hop)   
    
    return corpus_e2t, corpus_ere, corpus_trt, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings), torch.FloatTensor(rdf_relation_embeddings), torch.FloatTensor(type_embeddings)


e2t_Corpus_, ere_Corpus_, trt_Corpus_, entity_embeddings, relation_embeddings, rdf_relation_embeddings, type_embeddings = load_data(args)


if(args.get_2hop):
    file = args.data + "/2hop.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(ere_Corpus_.node_neighbors_2hop, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


if(args.use_2hop):
    print("Opening node_neighbors pickle object")
    file = args.data + "/2hop.pickle"
    with open(file, 'rb') as handle:
        node_neighbors_2hop = pickle.load(handle)

#entity_embeddings_copied = deepcopy(entity_embeddings)
#rdf_relation_embeddings_copied = deepcopy(rdf_relation_embeddings)
#type_embeddings_copied = deepcopy(type_embeddings)

print("Initial entity dimensions {} , rdf relation dimensions {}, type dimensions {}".format(
    entity_embeddings.size(), rdf_relation_embeddings.size(), type_embeddings.size()))
# %%

CUDA = torch.cuda.is_available()


def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=2, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=2, dim=1)

    if(CUDA):
        y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()
    else:
        y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples)

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss

def batch_gat_loss_e2t(gat_loss_func, train_indices, entity_embed, type_embed):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    head_embeds = entity_embed[pos_triples[:, 0]]
#    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = type_embed[pos_triples[:, 2]]

    x = head_embeds - tail_embeds
    pos_norm = torch.norm(x, p=2, dim=1)

    head_embeds = entity_embed[neg_triples[:, 0]]
#    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = type_embed[neg_triples[:, 2]]

    x = head_embeds - tail_embeds
    neg_norm = torch.norm(x, p=2, dim=1)

    if(CUDA):
        y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()
    else:
        y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples)

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


def train_gat(args):

    # Creating the gat model here.
    ####################################

    print("Defining model")

    print(
        "\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))

    model_rsgat = multiRSGAT(entity_embeddings, relation_embeddings, type_embeddings, rdf_relation_embeddings, args)
    # model_rsgat.load_state_dict(torch.load("./checkpoints/fb/out/rngat.trained_7999.pth"))
    start = 0
    wrapper_model = WrapperModel(model_rsgat)
    test_e2t = e2t_Corpus_.test_triples
    train_trt = trt_Corpus_.train_triples
    all_e2t = e2t_Corpus_.test_triples + e2t_Corpus_.validation_triples + e2t_Corpus_.train_triples
    evaluator1 = Type_Evaluator(test_e2t, all_e2t, logger)
    
    if CUDA:
        model_rsgat.cuda()

    # optimizer = torch.optim.Adam(model_rsgat.parameters(), 
    #     lr=args.lr, weight_decay=args.weight_decay_gat)
    optimizer = torch.optim.SGD(model_rsgat.parameters(), momentum=0.9,
        lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2000, gamma=0.5, last_epoch=-1)

    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

    epoch_losses_e2t = []   # losses of all epochs
    epoch_losses_ere = []
    epoch_losses_trt = []
    print("Number of epochs {}".format(args.epochs_gat))

    for epoch in range(start, start+args.epochs_gat):
        print("\nepoch-> ", epoch)
        random.shuffle(e2t_Corpus_.train_triples)
        random.shuffle(ere_Corpus_.train_triples)
        random.shuffle(trt_Corpus_.train_triples)
        
        e2t_Corpus_.train_indices = np.array(
            list(e2t_Corpus_.train_triples)).astype(np.int32)
        ere_Corpus_.train_indices = np.array(
            list(ere_Corpus_.train_triples)).astype(np.int32)        
        trt_Corpus_.train_indices = np.array(
            list(trt_Corpus_.train_triples)).astype(np.int32)
        
        model_rsgat.train()  # getting in training mode启用 BatchNormalization 和 Dropout
        
        start_time = time.time()
        epoch_loss_e2t = []
        epoch_loss_ere = []
        epoch_loss_trt = []

        if len(e2t_Corpus_.train_indices) % args.batch_size_gat == 0:
            num_iters_per_epoch = len(
                e2t_Corpus_.train_indices) // args.batch_size_gat
        else:
            num_iters_per_epoch = (
                len(e2t_Corpus_.train_indices) // args.batch_size_gat) + 1
        print(num_iters_per_epoch)
        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices_e2t, train_values_e2t = e2t_Corpus_.get_iteration_batch(iters)
            train_indices_ere, train_values_ere = ere_Corpus_.get_iteration_batch(iters)
            train_indices_trt, train_values_trt = trt_Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices_e2t = Variable(
                    torch.LongTensor(train_indices_e2t)).cuda()
                train_values_e2t = Variable(torch.FloatTensor(train_values_e2t)).cuda()
                train_indices_ere = Variable(
                    torch.LongTensor(train_indices_ere)).cuda()
                train_values_ere = Variable(torch.FloatTensor(train_values_ere)).cuda()
                train_indices_trt = Variable(
                    torch.LongTensor(train_indices_trt)).cuda()
                train_values_trt = Variable(torch.FloatTensor(train_values_trt)).cuda()                

            else:
                train_indices_e2t = Variable(torch.LongTensor(train_indices_e2t))
                train_values_e2t = Variable(torch.FloatTensor(train_values_e2t))
                train_indices_ere = Variable(torch.LongTensor(train_indices_ere))
                train_values_ere = Variable(torch.FloatTensor(train_values_ere))
                train_indices_trt = Variable(torch.LongTensor(train_indices_trt))
                train_values_trt = Variable(torch.FloatTensor(train_values_trt))

            # forward pass  在这里定义的forward方法的input
            
            out_entity_embed_long, out_entity_embed_short, out_relation_embed_long, \
            out_relation_embed_short, out_type_embed = model_rsgat(
                    e2t_Corpus_, ere_Corpus_, trt_Corpus_, train_indices_e2t, train_indices_ere, train_indices_trt)
            
            optimizer.zero_grad()#梯度置零,也就是把loss关于weight的导数变成0
                        
            loss_e2t = batch_gat_loss_e2t(
                gat_loss_func, train_indices_e2t, out_entity_embed_short, out_type_embed)
            loss_ere = batch_gat_loss(
                    gat_loss_func, train_indices_ere, out_entity_embed_long, out_relation_embed_long)
            loss_trt = batch_gat_loss(
                    gat_loss_func, train_indices_trt, out_type_embed, out_relation_embed_short)

            loss_e2t.backward(retain_graph=True)
            loss_ere.backward(retain_graph=True)
            loss_trt.backward()
            
            optimizer.step()

            epoch_loss_e2t.append(loss_e2t.data.item())
            epoch_loss_ere.append(loss_ere.data.item())
            epoch_loss_trt.append(loss_trt.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss_e2t {2:.4f}, Iteration_loss_ere {3:.4f}, Iteration_loss_trt {4:.4f}".format(
                iters, end_time_iter - start_time_iter, loss_e2t.data.item(), loss_ere.data.item(), loss_trt.data.item()))


        # scheduler.step()
        print("Epoch {} , average loss e2t {} , average loss ere {} , average loss trt {} , epoch_time {}".format(
            epoch, sum(epoch_loss_e2t) / len(epoch_loss_e2t), sum(epoch_loss_ere) / len(epoch_loss_ere), 
            sum(epoch_loss_trt) / len(epoch_loss_trt), time.time() - start_time))
        epoch_losses_e2t.append(sum(epoch_loss_e2t) / len(epoch_loss_e2t))
        epoch_losses_ere.append(sum(epoch_loss_ere) / len(epoch_loss_ere))
        epoch_losses_trt.append(sum(epoch_loss_trt) / len(epoch_loss_trt))

        save_model(model_rsgat, "rngat", epoch,
                   args.output_folder, args.epochs_gat)
        if (epoch + 1) % 50 == 0:
            evaluator1(wrapper_model)
        

train_gat(args)
