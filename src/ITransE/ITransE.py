import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../TransE'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../common'))
from utils import random_orthogonal_matrix
from TransE import TransE
import numpy as np
from numpy import linalg as LA
from numpy import random as RD
import time
import multiprocessing
from multiprocessing import Process, Value, Lock, Manager
import pickle
import heapq as HP
from scipy import spatial as SP

class ITransE(object):
    def __init__(self, dim = 100, save_dir = 'model_ITransE.bin'):
        self.dim = dim
        self.languages = []
        self.rate = 0.01 #learning rate
        self.trained_epochs = 0
        self.save_dir = save_dir
        #single-language models of each language
        self.models = {}
        self.triples = {}
        # cross-lingual linear transfer
        self.transfer = {}
        #intersect graph
        self.intersect_triples = np.array([0])
        #alignment seeds for ITransE
        self.seeds = {}
        self.seeds_pair = set([])
        #shuffle index for intersect triples
        self.intersect_index = np.array([0])
    
    def initialize_models(self, languages=['en', 'fr'], graphs=['../person/P_en_v3.csv','../person/P_fr_v3.csv'], intersect_graph='../person/P_en_fr_v3.csv', save_dirs = ['model_en.bin','model_fr.bin'], splitter='@@@', line_end='\n', rate=0.01):
        self.languages = languages
        if len(languages) != len(graphs):
            print "#Graph files are not correct"
            return None
        if len(languages) != len(save_dirs):
            print "#Save dirs for submodels are not correct"
            return None
        for i in range(len(languages)):
            lan = self.languages[i]
            self.models[lan] = TransE(dim=self.dim, save_dir=save_dirs[i])
            self.models[lan].rate = rate
            self.models[lan].generate_vocab(graphs[i], splitter, line_end)
            self.models[lan].initialize_vectors()
            triples = []
            for line in open(graphs[i]):
                line = line.rstrip(line_end).split(splitter)
                tmp_model = self.models[lan]
                triples.append([tmp_model.e2vec[line[0]], tmp_model.r2vec[line[1]], tmp_model.e2vec[line[2]]])
            triples = np.array(triples, dtype=np.int)
            self.triples[lan] = triples
            print "Loaded graph for [",lan,"] size ",len(self.triples[lan])
        triples = []
        for l in languages:
            self.seeds[l] = {}
        for line in open(intersect_graph):
            line = line.rstrip(line_end).split(splitter)
            tmp_line = []
            suc = True
            for i in range(len(languages)):
                lan = languages[i]
                h = self.models[lan].e2vec.get(line[i * 3])
                r = self.models[lan].r2vec.get(line[i * 3 + 1])
                t = self.models[lan].e2vec.get(line[i * 3 + 2])
                if h == None or r == None or t == None:
                    suc = False
                    break
                tmp_line.append([h, r, t])
            if suc == True: #every thing in this line are contained in the vocabs of models of corresponding languages
                triples.append(tmp_line)
                self.seeds_pair.add((tmp_line[0][0], tmp_line[1][0]))
                self.seeds[self.languages[0]][tmp_line[0][0]] = tmp_line[1][0]
                self.seeds[self.languages[1]][tmp_line[1][0]] = tmp_line[0][0]
                #self.seeds.add((tmp_line[0][2], tmp_line[1][2]))
                self.seeds[self.languages[0]][tmp_line[0][2]] = tmp_line[1][2]
                self.seeds[self.languages[1]][tmp_line[1][2]] = tmp_line[0][2]
        self.intersect_triples = np.array(triples, dtype=np.int)
        #self.intersect_index = np.array(range(len([x for x in self.seeds[self.languages[0]].iteritems()])), dtype=np.int)
        for pair in self.seeds_pair:
            self.models[self.languages[0]].vec_e[pair[0]] = self.models[self.languages[1]].vec_e[pair[1]]
        print "Loaded intersect graph. size ",len(self.intersect_triples)

    def Train_MT(self, epochs=100, tol=50.0, rate=0.05, save_every_epochs=0, save_dir = None, languages=['en', 'fr'], graphs=['../person/P_en_v3.csv','../person/P_fr_v3.csv'], intersect_graph='../person/P_en_fr_v3.csv', save_dirs = ['model_en.bin','model_fr.bin'], splitter='@@@', line_end='\n', split_rate=True, L1_flag=False):
        if save_dir != None:
            self.save_dir = save_dir
        self.rate = rate
        self.initialize_models(languages, graphs, intersect_graph, save_dirs, splitter, line_end, rate)
        shuffle_index = {}
        const_decays = np.ones(len(self.languages) + 1)
        if split_rate:
            total = 0.0
            for lan in self.languages:
                total += len(self.triples[lan])
            total += len(self.intersect_triples)
            for i in range(len(self.languages)):
                const_decays[i] *= len(self.triples[self.languages[i]]) / total
            const_decays[-1] *= len(self.intersect_triples) / total
        for lan in self.languages:
            shuffle_index[lan] = np.array(range(len(self.triples[lan])), dtype=np.int)
        # train on intersect graph once first
        t1 = time.time()
        for i in range(self.trained_epochs + 1, self.trained_epochs + epochs + 1):
            self.trained_epochs = i
            print "Epoch ",i,":"
            for lan in self.languages:
                RD.shuffle(shuffle_index[lan])
            sum2 = 0.
            for x in range(len(self.languages)):
                model = self.models[self.languages[x]]
                model.train_1epoch_shared_param(self.triples[self.languages[x]], shuffle_index[self.languages[x]], self.seeds[self.languages[x]], self.models[self.languages[1-x]], None, None, const_decays[x], L1_flag)
            sum3 = []
            for lan in self.languages:
                sum3.append( self.models[lan].get_loss(self.triples[lan]) )
            sum4 = sum(sum3)
            print "Current sum of loss is ",' + '.join(map(str, sum3))," + ",sum2," = ",sum4 + sum2,". Time used ",time.time() - t1
            if save_every_epochs > 0 and i % save_every_epochs == 0:
                self.save(self.save_dir)
            if sum4 + sum2 < tol:
                print "Converged at epoch [", i,"]"
                self.save(self.save_dir)
                break
                
    def MT_helper_function(self, model, triples, shuffle_index, const_decay):
        model.train_1epoch(triples, shuffle_index, None, None, const_decay)

    #return k nearest entities to a given vector (np.array). method 1 uses cosine distance, otherwise L-2 norm. self_id means the vector corresponds to certain entity, and you want to get rid of it from the topk result
    def kNN_entity(self, vec, tgt_lan='en', topk=10, method=0, self_vec_id=None, replace_q=True):
        q = []
        model = self.models.get(tgt_lan)
        if model == None:
            print "Model for language", tgt_lan," does not exist."
            return None
        for i in range(len(model.vec_e)):
            #skip self
            if self_vec_id != None and i == self_vec_id:
                continue
            if method == 1:
                dist = SP.distance.cosine(vec, model.vec_e[i])
            else:
                dist = LA.norm(vec - model.vec_e[i])
            if (not replace_q) or len(q) < topk:
                HP.heappush(q, model.index_dist(i, dist))
            else:
                #indeed it fetches the biggest
                tmp = HP.nsmallest(1, q)[0]
                if tmp.dist > dist:
                    HP.heapreplace(q, model.index_dist(i, dist) )
        rst = []
        if replace_q:
            while len(q) > 0:
                item = HP.heappop(q)
                rst.insert(0, (model.vocab_e[model.vec2e[item.index]], item.dist))
        else:
            while len(q) > topk:
                HP.heappop(q)
            while len(q) > 0:
                item = HP.heappop(q)
                rst.insert(0, (model.vocab_e[model.vec2e[item.index]], item.dist))
        return rst

    #given entity name, find kNN
    def kNN_entity_name(self, name, src_lan='en', tgt_lan='fr', topk=10, method=0, replace=True):
        model = self.models.get(src_lan)
        if model == None:
            print "Model for language", src_lan," does not exist."
            return None
        id = model.e2vec.get(name)
        if id == None:
            print name," is not in vocab"
            return None
        if src_lan != tgt_lan:#if you're not quering the kNN in the same language, then no need to get rid of the "self" point. However, transfer the vector.          
            pass_vec = model.vec_e[id]###
            pass_vec /= LA.norm(pass_vec)
            return self.kNN_entity(pass_vec, tgt_lan, topk, method, self_vec_id=None, replace_q=replace)
        return self.kNN_entity(model.vec_e[id], tgt_lan, topk, method, self_vec_id=id, replace_q=replace)
    
    #return k nearest relations to a given vector (np.array)
    def kNN_relation(self, vec, tgt_lan='fr', topk=10, method=0, self_vec_id=None):
        q = []
        model = self.models.get(tgt_lan)
        if model == None:
            print "Model for language", tgt_lan," does not exist."
            return None
        for i in range(len(model.vec_r)):
            #skip self
            if self_vec_id != None and i == self_vec_id:
                continue
            if method == 1:
                dist = SP.distance.cosine(vec, model.vec_r[i])
            else:
                dist = LA.norm(vec - model.vec_r[i])
            if len(q) < topk:
                HP.heappush(q, model.index_dist(i, dist))
            else:
                #indeed it fetches the biggest
                tmp = HP.nsmallest(1, q)[0]
                if tmp.dist > dist:
                    HP.heapreplace(q, model.index_dist(i, dist) )
        rst = []
        while len(q) > 0:
            item = HP.heappop(q)
            rst.insert(0, (model.vocab_r[model.vec2r[item.index]], item.dist))
        return rst

    #given relation name, find kNN
    def kNN_relation_name(self, name, src_lan='en', tgt_lan='fr', topk=10, method=0):
        model = self.models.get(src_lan)
        if model == None:
            print "Model for language", src_lan," does not exist."
            return None
        id = model.r2vec.get(name)
        if id == None:
            print name," is not in vocab"
            return None
        if src_lan != tgt_lan:#if you're not quering the kNN in the same language, then no need to get rid of the "self" point    
            pass_vec = model.vec_r[id]
            pass_vec /= LA.norm(pass_vec)
            return self.kNN_relation(pass_vec, tgt_lan, topk, method, self_vec_id=None)
        return self.kNN_relation(model.vec_r[id], tgt_lan, topk, method, self_vec_id=id)

    #entity name to vector
    def entity_vec(self, name, src_lan='en'):
        model = self.models.get(src_lan)
        if model == None:
            print "Model for language", src_lan," does not exist."
            return None
        e = model.e2vec.get(name)
        if e == None:
            return None
        return model.vec_e[e]

    #entity name to vector
    def relation_vec(self, name, src_lan='en'):
        model = self.models.get(src_lan)
        if model == None:
            print "Model for language", src_lan," does not exist."
            return None
        r = model.r2vec.get(name)
        if r == None:
            return None
        return model.vec_r[r]
    
    def entity_rank(self, vec_s, vec_t, src_lan='en'):
        model = self.models.get(src_lan)
        if model == None:
            print "Model for language", src_lan," does not exist."
            return None
        rst = 1
        t_dist = LA.norm(vec_s - vec_t)
        for vec in model.vec_e:
            dist = LA.norm(vec - vec_t)
            if dist > 0 and dist < t_dist:
                rst += 1
        return rst
    
    def relation_rank(self, vec_s, vec_t, src_lan='en'):
        model = self.models.get(src_lan)
        if model == None:
            print "Model for language", src_lan," does not exist."
            return None
        rst = 1
        t_dist = LA.norm(vec_s - vec_t)
        for vec in model.vec_r:
            dist = LA.norm(vec - vec_t)
            if dist > 0 and dist < t_dist:
                rst += 1
        return rst

    def entity_transfer_vec(self, name, src_lan='en', tgt_lan='fr'):
        model = self.models.get(src_lan)
        if model == None:
            print "Model for language", src_lan," does not exist."
            return None
        id = model.e2vec.get(name)
        if id == None:
            print name," is not in vocab"
            return None
        return model.vec_e[id]
    
    def relation_transfer_vec(self, name, src_lan='en', tgt_lan='fr'):
        model = self.models.get(src_lan)
        if model == None:
            print "Model for language", src_lan," does not exist."
            return None
        id = model.r2vec.get(name)
        if id == None:
            print name," is not in vocab"
            return None
        return model.vec_r[id]
    
    
    def save(self, filename):
        #save invert matrices
        f = open(filename,'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print "Save ITransE model as ", filename
    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print "Loaded ITransE model from", filename
