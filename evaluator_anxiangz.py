import os, pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
CUDA = torch.cuda.is_available ()

class WrapperModel():
    def __init__(self, model):
        self.model = model
        self.e_short = self.model.output_entity_embeddings_short
        self.e_long = self.model.output_entity_embeddings_long
        self.r_long = self.model.output_relation_embeddings_long
        self.r_short = self.model.output_relation_embeddings_short
        self.t = self.model.output_type_embedding
        t_emb_size = self.t.size(1)
        r_emb_size = self.r_long.size(1)

    def _scores_et(self, e, et=None):
        # 给定一个三元组，求替换了label的
        T = self.t
        E = self.e_short
        if et is None :
            score = E[e] - T
            score = score ** 2

        else:
            score = E[e] - T[et]
            score = score.reshape (1, -1)
            score = score ** 2

        return torch.sum(score, dim=1)

    def _scores_trt(self, et, r, et2 = None, change_head=True):
        T = self.t
        R = self.r_short
        if et2 is None:
            if change_head :
                score = T + R[r] - T[et]
            else :
                score = T[et] + R[r] - T
            score = score ** 2
        else:
            if change_head :
                score = T[et2] + R[r] - T[et]
            else :
                score = T[et] + R[r] - T[et2]
            score = score.reshape (1, -1)
            score = score ** 2

        return torch.sum (score, dim=1)

class ConnectE2T_TRT(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, initial_type_emb, initial_short_relation_emb):
        super ().__init__ ()     
        e_size, r_size, et_size = initial_entity_emb.size(1), initial_relation_emb.size(1), initial_type_emb.size(1)
        self.final_type_embeddings = nn.Parameter (torch.randn (et_size), requires_grad=True)
        self.final_type_embeddings.data = initial_type_emb.data
        
        self.final_relation_embeddings = nn.Parameter (torch.randn (r_size), requires_grad=True)
        self.final_relation_embeddings.data = initial_relation_emb.data
        
        self.final_short_relation_embeddings = nn.Parameter (torch.randn (et_size), requires_grad=True)
        self.final_short_relation_embeddings.data = initial_short_relation_emb.data
        
        self.final_entity_embeddings = nn.Parameter (torch.randn (e_size), requires_grad=True)
        self.final_entity_embeddings.data = initial_entity_emb.data
        
        self.e2t_layer = nn.Linear(e_size, et_size)

    def forward(self):
        return self.e2t_layer(self.final_entity_embeddings), self.final_relation_embeddings, self.final_short_relation_embeddings, self.final_type_embeddings

""" wrapper of connectE2T_TRT"""
class WrapperModel2():
    name = "ConnectE"
    def __init__(self, e2t_trt):
        self.e2t_trt = e2t_trt
        self.t = self.e2t_trt.final_type_embeddings
    def get_type_size(self):
        return self.e2t_trt.final_type_embeddings.shape[0]
    
    def _scores_et(self, e, et=None) :
        # 给定一个三元组，求替换了label的
        T = self.e2t_trt.final_type_embeddings
        E = self.e2t_trt.final_entity_embeddings
        if et is None :
            score = self.e2t_trt.e2t_layer(E[e]) - T
            score = score ** 2

        else :
            score = self.e2t_trt.e2t_layer(E[e]) - T[et]
            score = score.reshape (1, -1)
            score = score ** 2

        return torch.sum (score, dim=1)
    
    def _scores_trt(self, et, r, et2 = None, change_head=True):
        T = self.e2t_trt.final_type_embeddings
        R = self.e2t_trt.final_short_relation_embeddings
        if et2 is None:
            if change_head :
                score = T + R[r] - T[et]
            else :
                score = T[et] + R[r] - T
            score = score ** 2
        else:
            if change_head :
                score = T[et2] + R[r] - T[et]
            else :
                score = T[et] + R[r] - T[et2]
            score = score.reshape (1, -1)
            score = score ** 2

        return torch.sum (score, dim=1)

class Evaluator(object):
    def __init__(self, test_triplet, logger=None):
        self.logger = logger
        self.xs = test_triplet
        self.tot = len(test_triplet)
        self.pos = None
        self.fpos = None

    def __call__(self, model):
        pos_v, fpos_v = self.positions(model)
        self.pos = pos_v
        self.fpos = fpos_v
        fmrr = self.p_ranking_scores(pos_v, fpos_v, model.epoch, 'VALID')
        return fmrr

    def positions(self, mdl):
        raise NotImplementedError

    def p_ranking_scores(self, pos, fpos, epoch, txt):
        rpos = [p for k in pos.keys() for p in pos[k]]
        frpos = [p for k in fpos.keys() for p in fpos[k]]
        
        fmrr = self._print_pos(
            np.array(rpos),
            np.array(frpos),
            epoch, txt)
        return fmrr

    def _print_pos(self, pos, fpos, epoch, txt):
        mrr, mean_pos, hits = self.compute_scores(pos)
        fmrr, fmean_pos, fhits = self.compute_scores(fpos)
        if self.logger:
            self.logger.info(
                f"[{epoch: 3d}] {txt}: MRR = {mrr:.4f}/{fmrr:.4f}, "
                f"Mean Rank = {mean_pos:.2f}/{fmean_pos:.2f}, "
                f"Hits@1 = {hits[0]:.4f}/{fhits[0]:.4f}, "
                f"Hits@3 = {hits[1]:.4f}/{fhits[1]:.4f}, "
                f"Hits@10 = {hits[2]:.4f}/{fhits[2]:.4f}"
            )
        else:
            print(
                f"[{epoch: 3d}] {txt}: MRR = {mrr:.2f}/{fmrr:.2f}, "
                f"Mean Rank = {mean_pos:.2f}/{fmean_pos:.2f}, "
                f"Hits@1 = {hits[0]:.2f}/{fhits[0]:.2f}, "
                f"Hits@3 = {hits[1]:.2f}/{fhits[1]:.2f}, "
                f"Hits@10 = {hits[2]:.2f}/{fhits[2]:.2f}"
            )
        return fmrr

    def compute_scores(self, pos, hits=None):
        if hits is None:
            hits = [1, 3, 10]
        mrr = np.mean(1.0 / pos)
        mean_pos = np.mean(pos)
        hits_results = []
        for h in range(0, len(hits)):
            k = np.mean(pos <= hits[h])
            print(k)
            k2 = k.sum()
            hits_results.append(k2 * 100)
        return mrr, mean_pos, hits_results

    def save_prediction(self, path="output/pos", fpath="output/fpos_e2t_trt"):
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path, "pos"), 'wb') as f:
            pickle.dump(self.pos, f)
        with open(os.path.join(path, "fpos"), 'wb') as f:
            pickle.dump(self.fpos, f)

    def load_prediction(self,path="output/pos", fpath="output/fpos_e2t_trt"):
        with open(path, 'rb') as f:
            self.pos = pickle.load(f)

        with open(fpath, 'rb') as f:
            self.fpos = pickle.load(f)


class Type_Evaluator(Evaluator):
    def __init__(self, xs, true_tuples, logger=None):
        super(Type_Evaluator, self).__init__(xs, logger)
        self.idx = defaultdict(list)   # defaultdict
        self.tt = defaultdict(list)  # true tuples
        self.sz = len(xs)

        for e,_, t in xs:
            if t not in self.idx[e]:
                self.idx[e].append(t)

        for e,_, t in true_tuples:
            if t not in self.tt[e]:
                self.tt[e].append(t)

        self.idx = dict(self.idx)
        self.tt = dict(self.tt)

    def __call__(self, model, path=None, fpath=None, corpus_e2t=None):
        if path and fpath:
            self.load_prediction(path, fpath)
        else:
            pos_v, fpos_v = self.positions(model, corpus_e2t=corpus_e2t)
            self.pos = pos_v
            self.fpos = fpos_v

        return self.et_ranking_scores(self.pos, self.fpos, 0, 'VALID')

    def et_ranking_scores(self, pos, fpos, epoch, txt):
        tpos = [p for k in pos.keys() for p in pos[k]['type']]
        tfpos = [p for k in fpos.keys() for p in fpos[k]['type']]
        fmrr = self._print_pos(
            np.array(tpos),
            np.array(tfpos),
            epoch, txt)
        return fmrr

    def positions(self, mdl, corpus_e2t=None):
        pos = {}    # Raw Positions
        fpos = {}   # Filtered Positions
        cnt = 0
        for e, ts in tqdm(self.idx.items()):

            ppos = {'type': []}
            pfpos = {'type': []}

            for t in ts:
                scores_t = mdl._scores_et(e).flatten()
                sortidx_t = torch.argsort(torch.argsort(scores_t))                    
                ppos['type'].append(sortidx_t[t].cpu().item() + 1)

                rm_idx = self.tt[e]
                rm_idx = [i for i in rm_idx if i != t]
                scores_t[rm_idx] = float('inf')
                sortidx_t = torch.argsort(torch.argsort(scores_t))
                pfpos['type'].append(sortidx_t[t].cpu().item() + 1)
                lis = {"/m/05yg8kx", "/m/0zb8", "/m/0jbk", "/m/06bpx", "/m/07c72", "/m/0fhp9"}
                if corpus_e2t is not None and corpus_e2t.id2entity[e] in lis:
                    ets = torch.argsort(scores_t)[:10]
                    ets_names = [corpus_e2t.id2type[i.cpu().item()] for i in ets]
                    print("Sample Prediction: {}: {}".format(corpus_e2t.id2entity[e], ets_names))
                    ets_names = [corpus_e2t.id2type[i] for i in self.tt[e]]
                    print("Truth: {}: {}".format(corpus_e2t.id2entity[e], ets_names))
                    cnt += 1
                    if cnt == 3:
                        corpus_e2t = None
            pos[e] = ppos
            fpos[e] = pfpos

        return pos, fpos


class Type_Evaluator_trt (Type_Evaluator) :
    def __init__(self, xs, true_tuples, train_triplets, logger=None, p=0.02) :
        super ().__init__ (xs, true_tuples, logger)
        self.tt_h_l, self.tt_t_l = self.convert_triple_into_dict (
            train_triplets)  # {head: {tail: [relation1, relation2, ...]}}
        self.p = p

    def convert_triple_into_dict(self, triplet) :
        h_l_dict = {}
        t_l_dict = {}
        count = 1
        for head, label, tail in triplet :
            if head in h_l_dict.keys () :
                if label in h_l_dict[head].keys () and tail not in h_l_dict[head][label] :
                    h_l_dict[head][label].append (tail)
                    count += 1
                else :
                    h_l_dict[head][label] = [tail]
                    count += 1
            else :
                h_l_dict[head] = {label : [tail]}

            if tail in t_l_dict.keys () :
                if label in t_l_dict[tail].keys () and tail not in t_l_dict[tail][label] :
                    t_l_dict[tail][label].append (head)
                    count += 1
                else :
                    t_l_dict[tail][label] = [head]
                    count += 1
            else :
                t_l_dict[tail] = {label : [head]}
        print (f"initialization complete: {count} items")
        return h_l_dict, t_l_dict

    def positions(self, mdl):
        pos = {}  # Raw Positions
        fpos = {}  # Filtered Positions
        count = 0
        for e, ts in tqdm(self.idx.items()):

            ppos = {'type' : []}
            pfpos = {'type' : []}

            for t in ts :
                score1, score2 = self._score_trt (mdl, e)
                scores_t = mdl._scores_et (e).flatten ()
                scores_t = scores_t + self.p * score1 + self.p * score2
                sortidx_t = torch.argsort(torch.argsort(scores_t))
                ppos['type'].append (sortidx_t[t].cpu().item() + 1)

                rm_idx = self.tt[e]
                rm_idx = [i for i in rm_idx if i != t]
                scores_t[rm_idx] = np.Inf
                sortidx_t = torch.argsort(torch.argsort(scores_t))
                pfpos['type'].append (sortidx_t[t].cpu().item() + 1)
                count += 1

            pos[e] = ppos
            fpos[e] = pfpos

        return pos, fpos

    def _score_trt(self, mdl, e) :
        # e, r, e2
        scores1 = []
        scores2 = []
        tot = 0
        if e in self.tt_h_l.keys () :
            for r, related_e in self.tt_h_l[e].items () :
                for entity in related_e :
                    if entity in self.tt.keys () :
                        types = self.tt[entity]
                        for et in types :
                            scores1.append (mdl._scores_trt (et, r, change_head=True).unsqueeze(0))
                            tot += 1

        if e in self.tt_t_l.keys () :
            for r, related_e in self.tt_t_l[e].items () :
                for entity in related_e :
                    if entity in self.tt.keys () :
                        types = self.tt[entity]
                        for et in types :
                            scores2.append (mdl._scores_trt (et, r, change_head=False).unsqueeze(0))
                            tot += 1

        if scores1 :
            score1 = torch.mean (torch.cat (scores1), axis=0)
        else :
            score1 = torch.zeros (mdl.t.shape[0])
            if CUDA:
                score1 = score1.cuda()

        if scores2 :
            score2 = torch.mean (torch.cat (scores2), axis=0)
        else :
            score2 = torch.zeros (mdl.t.shape[0])
            if CUDA:
                score2 = score2.cuda()
        return score1, score2
    

class Classification_Evaluator(Type_Evaluator_trt):
    def __init__(self, validate_set, test_set, true_tuples, train_triplets, model, logger=None,
                 save_data=False, load_data=False, load_path=None, save_path=None):
        super().__init__(test_set, true_tuples, train_triplets, logger)
        self.validate = validate_set
        self.test = test_set
        self.model = model
        self.name = model.name
        self.type_size = self.model.get_type_size()
        self.load_data = load_data
        self.load_path = load_path
        self.save_data = save_data
        self.save_path = save_path
        self.logger.info(f"Start to handel {self.name} model")

    def sample(self, p_sample):
        n_sample = []
        for e, _ in p_sample:
            p_et = np.random.randint(0, self.type_size)

            # while p_et in self.tt[e]:
            #     p_et = np.random.randint(0, self.type_size)
            n_sample.append((e, p_et))
        return n_sample

    def __call__(self):
        saving_data = {}
        # validate
        if self.load_data:
            results1 = self.load(self.load_path, type='valid')
        else:
            results1 = self.evaluate(self.validate)

        best_dis, best_score = self.get_optimal_distance(results1)
        if self.save_data:
            saving_data = {'valid': results1}

        self.logger.info("%s: The best distance/accuracy in VALID data is %f/%f"%(self.name, best_dis, best_score))

        # test
        if self.load_data:
            results = self.load(self.load_path, type='test')

        else:
            results = self.evaluate(self.test)

        if self.save_data:
            saving_data['test'] = results
            self.save(saving_data, path=self.save_path)

        acc = self.cal_accuracy(results, best_dis)
        self.logger.info("%s: The accuracy in TEST is %f"%(self.name, acc))
        return results

    @classmethod
    def save(cls, data, path):
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path, type):
        with open(path, "rb") as f:
            return pickle.load(f)[type]
    @classmethod
    def normalize(cls, data):
        max_ = max([d for d, _ in data])
        return [(1-d/max_, flag) for d, flag in data]

    def get_optimal_distance(self, results):
        distances = []
        y_true = []
        for d, pos in results:
            distances.append(d)
            y_true.append(pos)

        distances = np.array(distances)
        y_true = np.array(y_true)
        best_accuracy = 0
        best_distance = distances[0]
        for dis in tqdm(distances):
            y_pred = np.ones_like(y_true)
            y_pred[distances>=dis] = -1
            acc = accuracy_score(y_true, y_pred)
            if acc > best_accuracy:
                best_accuracy = acc
                best_distance = dis

        return best_distance, best_accuracy

    def cal_accuracy(self, results, dis):
        distances = []
        y_true = []
        for d, pos in results:
            distances.append(d)
            y_true.append(pos)

        distances = np.array(distances)
        y_true = np.array(y_true)
        y_pred = np.ones_like(y_true)
        y_pred[distances >= dis] = -1
        acc = accuracy_score(y_true, y_pred)
        return acc 

    def evaluate(self, p_sample):
        n_sample = self.sample(p_sample)
        p_sample = [(k, 1) for k in p_sample]
        n_sample = [(k, -1) for k in n_sample]
        tot_sample = p_sample + n_sample
        tot_scores = []
        count = 0
        for m, flag in tot_sample:
            e, et = m
            scores_t = self.model._scores_et(e, et).flatten()[0]
            score1, score2 = self._score_trt(self.model, e, et)
            trt_score = 0.5*score1[0] + 0.5*score2[0]
            avg_score = scores_t + trt_score
            tot_scores.append((avg_score, flag))
            count += 1

        return tot_scores


    def _score_trt(self, mdl, e, et2):
        # e, r, e2
        scores1 = []
        scores2 = []
        if e in self.tt_h_l.keys():
            for r, related_e in self.tt_h_l[e].items():
                for entity in related_e:
                    if entity in self.tt.keys():
                        types = self.tt[entity]
                        for et in types:
                            scores1.append(mdl._scores_trt(et, r, change_head=True, et2=et2))


        if e in self.tt_t_l.keys():
            for r, related_e in self.tt_t_l[e].items():
                for entity in related_e:
                    if entity in self.tt.keys():
                        types = self.tt[entity]
                        for et in types:
                            scores2.append(mdl._scores_trt(et, r, change_head=False, et2=et2))

        if scores1:
            score1 = np.vstack(scores1).mean(axis=0)
        else:
            score1 = np.zeros(mdl.typeMat.shape[0])

        if scores2:
            score2 = np.vstack(scores2).mean(axis=0)
        else:
            score2 = np.zeros(mdl.typeMat.shape[0])

        return score1, score2

class Classification_Evaluator_E2T(Classification_Evaluator):
    def evaluate(self, p_sample):
        p_sample = [(et, r) for et, _, r in p_sample]
        n_sample = self.sample(p_sample)
        p_sample = [(k, 1) for k in p_sample]
        n_sample = [(k, -1) for k in n_sample]
        tot_sample = p_sample + n_sample
        tot_scores = []
        count = 0
        for m, flag in tqdm(tot_sample):
            e, et = m
            scores_t = self.model._scores_et(e, et).flatten()[0]
            avg_score = scores_t.cpu().numpy()
            tot_scores.append((avg_score, flag))
            count += 1

        return tot_scores