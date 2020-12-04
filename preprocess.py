import torch
import os
import numpy as np


def read_entity_from_id(filename='./data/FB15k/entity2id.txt'):
    entity2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split('\t')) > 1:
                entity, entity_id = line.strip().split('\t'
                )[0].strip(), line.strip().split('\t')[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id


def read_relation_from_id(filename='./data/FB15k/relation2id.txt'):
    relation2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split('\t')) > 1:
                relation, relation_id = line.strip().split('\t'
                )[0].strip(), line.strip().split('\t')[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id

def read_type_from_id(filename='./data/FB15k/type2id.txt'):
    type2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split('\t')) > 1:
                types, type_id = line.strip().split('\t'
                )[0].strip(), line.strip().split('\t')[1].strip()
                type2id[types] = int(type_id)
    return type2id


def init_embeddings(entity_file, relation_file):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            entity_emb.append([float(val) for val in line.strip().split()])

    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val) for val in line.strip().split()])

    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)


def parse_line(line):
    line = line.strip().split('\t')
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2


def load_data(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    # this is list for relation triples
    triples_data = []

    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # columnn of sparse tensor, data contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    rows, cols, data = [], [], []
    unique_entities = set()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append(
            (entity2id[e1], relation2id[relation], entity2id[e2]))
        if not directed:
                # Connecting source and tail entity
            rows.append(entity2id[e1])
            cols.append(entity2id[e2])
            if is_unweigted:
                data.append(1)
            else:
                data.append(relation2id[relation])

        # Connecting tail and source entity
        rows.append(entity2id[e2])
        cols.append(entity2id[e1])
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation2id[relation])

    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data), list(unique_entities)

def parse_line_e2t(line):
    line = line.strip().split('\t')
    e, t = line[0].strip(), line[1].strip()
    return e, t

def load_data_e2t(filename, entity2id, type2id, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    # this is list for relation triples
    triples_data = []

    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # columnn of sparse tensor, data contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    rows, cols, data = [], [], []
    unique_entities = set()
    unique_types = set()
    for line in lines:
        e, t = parse_line_e2t(line)
        unique_entities.add(e)
        unique_types.add(t)
        triples_data.append(
            (entity2id[e], 0, type2id[t]))
        if not directed:
                # Connecting source and tail entity
            rows.append(entity2id[e])
            cols.append(type2id[t])
            data.append(0)


        # Connecting tail and source entity
        rows.append(entity2id[e])
        cols.append(type2id[t])
        data.append(0)

    print("number of unique_entities ->", len(unique_entities))
    print("number of unique_types ->", len(unique_types))
    return triples_data, (rows, cols, data), list(unique_entities), list(unique_types)

def build_data(path='./data/FB15k/', is_unweigted=False, directed=True):
    entity2id = read_entity_from_id(path + 'entity2id.txt')
    relation2id = read_relation_from_id(path + 'relation2id.txt')
    type2id = read_type_from_id(path + 'type2id.txt')

    # Adjacency matrix only required for training phase
    # Currenlty creating as unweighted, undirected
    train_triples_e2t, train_adjacency_mat_e2t, unique_entities_train_e2t, unique_types_train_e2t = load_data_e2t(os.path.join(
        path, 'e2t_train.txt'), entity2id, type2id, directed)
    validation_triples_e2t, valid_adjacency_mat_e2t, unique_entities_validation_e2t, unique_types_validation_e2t = load_data_e2t(
        os.path.join(path, 'e2t_valid.txt'), entity2id, type2id, directed)
    test_triples_e2t, test_adjacency_mat_e2t, unique_entities_test_e2t, unique_types_test_e2t = load_data_e2t(os.path.join(
        path, 'e2t_test.txt'), entity2id, type2id, directed)

    train_triples_ere, train_adjacency_mat_ere, unique_entities_train_ere = load_data(os.path.join(
            path, 'ere_train.txt'), entity2id, relation2id, is_unweigted, directed)
    validation_triples_ere, valid_adjacency_mat_ere, unique_entities_validation_ere = load_data(
        os.path.join(path, 'ere_valid.txt'), entity2id, relation2id, is_unweigted, directed)
    test_triples_ere, test_adjacency_mat_ere, unique_entities_test_ere = load_data(os.path.join(
        path, 'ere_test.txt'), entity2id, relation2id, is_unweigted, directed)
    
    train_triples_trt, train_adjacency_mat_trt, unique_types_train_trt = load_data(os.path.join(
            path, 'trt_train_disc.txt'), type2id, relation2id, is_unweigted, directed)
    validation_triples_trt, valid_adjacency_mat_trt, unique_types_validation_trt = load_data(
        os.path.join(path, 'trt_valid.txt'), type2id, relation2id, is_unweigted, directed)
    test_triples_trt, test_adjacency_mat_trt, unique_types_test_trt = load_data(os.path.join(
        path, 'trt_test.txt'), type2id, relation2id, is_unweigted, directed)  
    
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    left_entity, right_entity = {}, {}

    with open(os.path.join(path, 'ere_train.txt')) as f:
        lines = f.readlines()

    for line in lines:
        e1, relation, e2 = parse_line(line)

        # Count number of occurences for each (e1, relation)
        if relation2id[relation] not in left_entity:
            left_entity[relation2id[relation]] = {}
        if entity2id[e1] not in left_entity[relation2id[relation]]:
            left_entity[relation2id[relation]][entity2id[e1]] = 0
        left_entity[relation2id[relation]][entity2id[e1]] += 1

        # Count number of occurences for each (relation, e2)
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1

    left_entity_avg = {}
    for i in range(len(relation2id)):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_entity_avg = {}
    for i in range(len(relation2id)):
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}
    for i in range(len(relation2id)):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i])
    
    
    return (train_triples_e2t, train_adjacency_mat_e2t), (validation_triples_e2t, valid_adjacency_mat_e2t), (test_triples_e2t, test_adjacency_mat_e2t), \
        (train_triples_ere, train_adjacency_mat_ere), (validation_triples_ere, valid_adjacency_mat_ere), (test_triples_ere, test_adjacency_mat_ere), \
        (train_triples_trt, train_adjacency_mat_trt), (validation_triples_trt, valid_adjacency_mat_trt), (test_triples_trt, test_adjacency_mat_trt), \
        entity2id, relation2id, type2id, headTailSelector, unique_entities_train_e2t, unique_types_train_e2t, \
        unique_entities_train_ere, unique_types_train_trt
