import numpy as np
from numpy import linalg as LA
from numpy import random as RD
import time
import multiprocessing
from multiprocessing import Process, Value, Lock, Manager
import pickle
import heapq as HP
from scipy import spatial as SP
log_file = "transE.log"

t0 = time.time()

try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

class TransE(object):
    
    def __init__(self, dim = 100, save_dir = 'model_test.bin'):
        self.dim = dim
        #vocabulary of h,r,t
        self.vocab_e = []
        self.vocab_r = []
        #vectors of h,r,t
        self.vec_e = np.array([0]) #empty
        self.vec_r = np.array([0]) #empty
        #vocab to vector indices
        self.e2vec = {}
        self.r2vec = {}
        #vector indices to vocab indices
        self.vec2e = np.array([0], dtype=np.int)
        self.vec2r = np.array([0], dtype=np.int)
        #parameters
        self.rate = 0.01 #learning rate
        self.trained_epochs = 0
        self.save_dir = save_dir

    #scan triple file once and generate vocabularies for h,r,t respectively
    def generate_vocab(self, file_dir, splitter='@@@', line_end = '\n'):
        distinct_e = {}
        distinct_r = {}
        count = 0
        print "Begin generating vocab from file: ",file_dir,", column splitted by '",splitter,"'"
        for line in open(file_dir):
            count += 1
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 3:
                continue
            if distinct_e.get(line[0]) == None:
                distinct_e[line[0]] = 1
                self.vocab_e.append(line[0])
            if distinct_r.get(line[1]) == None:
                distinct_r[line[1]] = 1
                self.vocab_r.append(line[1])
            if distinct_e.get(line[2]) == None:
                distinct_e[line[2]] = 1
                self.vocab_e.append(line[2])
        print "Finish generating vocab on ",len(self.vocab_e)," entities and ",len(self.vocab_r)," relations."
    
    #initialize vectors for words in each vocab after generating vocabs
    def initialize_vectors(self):
        self.vec2e = np.zeros(len(self.vocab_e), dtype=np.int)
        for i in range(len(self.vocab_e)):
            w = self.vocab_e[i]
            self.e2vec[w] = i
            self.vec2e[i] = i
        #randomize and normalize the initial vector
        self.vec_e = RD.randn(len(self.vocab_e), self.dim)
        for i in range(len(self.vec_e)):
            self.vec_e[i] /= LA.norm(self.vec_e[i])
        self.vec2r = np.zeros(len(self.vocab_r), dtype=np.int)
        for i in range(len(self.vocab_r)):
            w = self.vocab_r[i]
            #randomize and normalize the initial vector
            self.r2vec[w] = i
            self.vec2r[i] = i
        self.vec_r = RD.randn(len(self.vocab_r), self.dim)
        for i in range(len(self.vec_r)):
            self.vec_r[i] /= LA.norm(self.vec_r[i])
        print "Initialized vectors for all entities and relations."
    
    #GD of h + r - t
    def gradient_decent(self,index_e1,index_r,index_e2,const_decay=1.0, L1=False):
        diff = 2.0 * (self.vec_e[index_e2] - self.vec_e[index_e1] - self.vec_r[index_r]) * self.rate * const_decay
        if L1:
            for i in range(len(diff)):
                if diff[i] > 0:
                    diff[i] = self.rate
                else:
                    diff[i] = -self.rate
        self.vec_e[index_e1] += diff #-1.0 * diff * self.rate
        self.vec_r[index_r] += diff
        self.vec_e[index_e2] -= diff
        self.vec_e[index_e1] /= LA.norm(self.vec_e[index_e1])
        self.vec_e[index_e2] /= LA.norm(self.vec_e[index_e2])
        # return the current L-2 norm of h + r - t
        return LA.norm(self.vec_e[index_e1] + self.vec_r[index_r] - self.vec_e[index_e2])
    
    class index_dist:
        def __init__(self, index, dist):
            self.dist = dist
            self.index = index
            return
        def __lt__(self, other):
            return self.dist > other.dist
        #def __cmp__(self, other):
        #    return -cmp(self.dist, other.dist)

    #return k nearest entities to a given vector (np.array). method 1 uses cosine distance, otherwise L-2 norm. self_id means the vector corresponds to certain entity, and you want to get rid of it from the topk result
    def kNN_entity(self, vec, topk=10, method=0, self_vec_id=None):
        q = []
        for i in range(len(self.vec_e)):
            #skip self
            if self_vec_id != None and i == self_vec_id:
                continue
            if method == 1:
                dist = SP.distance.cosine(vec, self.vec_e[i])
            else:
                dist = LA.norm(vec - self.vec_e[i])
            if len(q) < topk:
                HP.heappush(q, self.index_dist(i, dist))
            else:
                #indeed it fetches the biggest
                tmp = HP.nsmallest(1, q)[0]
                if tmp.dist > dist:
                    HP.heapreplace(q, self.index_dist(i, dist) )
        rst = []
        while len(q) > 0:
            item = HP.heappop(q)
            rst.insert(0, (self.vocab_e[self.vec2e[item.index]], item.dist))
        return rst

    #given entity name, find kNN
    def kNN_entity_name(self, name, topk=10, method=0):
        id = self.e2vec.get(name)
        if id == None:
            print name," is not in vocab"
            return None
        return self.kNN_entity(self.vec_e[id], topk, method, self_vec_id=id)
    
    #return k nearest relations to a given vector (np.array)
    def kNN_relation(self, vec, topk=10, method=0, self_vec_id=None):
        q = []
        for i in range(len(self.vec_r)):
            #skip self
            if self_vec_id != None and i == self_vec_id:
                continue
            if method == 1:
                dist = SP.distance.cosine(vec, self.vec_r[i])
            else:
                dist = LA.norm(vec - self.vec_r[i])
            if len(q) < topk:
                HP.heappush(q, self.index_dist(i, dist))
            else:
                #indeed it fetches the biggest
                tmp = HP.nsmallest(1, q)[0]
                if tmp.dist > dist:
                    HP.heapreplace(q, self.index_dist(i, dist) )
        rst = []
        while len(q) > 0:
            item = HP.heappop(q)
            rst.insert(0, (self.vocab_r[self.vec2r[item.index]], item.dist))
        return rst

    #given relation name, find kNN
    def kNN_relation_name(self, name, topk=10, method=0):
        id = self.r2vec.get(name)
        if id == None:
            print name," is not in vocab"
            return None
        return self.kNN_relation(self.vec_r[id], topk, method, self_vec_id=id)

    #entity name to vector
    def entity_vec(self, name):
        e = self.e2vec.get(name)
        if e == None:
            return None
        return self.vec_e[e]

    #entity name to vector
    def relation_vec(self, name):
        r = self.r2vec.get(name)
        if r == None:
            return None
        return self.vec_r[r]

    #Single-thread training. if save_every_epochs > 0 then model will be saved every save_every_epochs epochs
    def Train_ST(self, file_dir, splitter='@@@', line_end='\n', epochs=100, tol=50.0, rate=0.01, save_every_epochs=0, const_decay=1.0, save_dir = None, valid_dir = None, L1_flag=False):
        if save_dir != None:
            self.save_dir = save_dir
        self.rate = rate
        self.generate_vocab(file_dir, splitter, line_end)
        self.initialize_vectors()
        triples = []
        
        for line in open(file_dir):
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 3:
                continue
            e1 = self.e2vec.get(line[0])
            r = self.r2vec.get(line[1])
            e2 = self.e2vec.get(line[2])
            if e1 == None or e2 == None or r == None:
                continue
            triples.append([e1,r,e2])
        triples = np.array(triples, dtype=np.int)
        valid = []
        sum_hist = []
        if valid_dir != None:
            for line in open(valid_dir):
                line = line.rstrip(line_end).split(splitter)
                if len(line) != 3:
                    continue
                e1 = self.e2vec.get(line[0])
                r = self.r2vec.get(line[1])
                e2 = self.e2vec.get(line[2])
                if e1 == None or e2 == None or r == None:
                    continue
                valid.append([e1,r,e2])
        shuffle_index = np.array(range(len(triples)), dtype=np.int)
        print "Loaded triples.",len(triples)
        print "Begin training. epochs=[",epochs,"] learning rate=[",self.rate,"] tol=[", tol, "]dim=[",self.dim,']'
        for i in range(self.trained_epochs + 1, self.trained_epochs + epochs + 1):
            self.trained_epochs = i
            print "Epoch ",i,":"
            sum = 0.0
            RD.shuffle(shuffle_index)
            count = 0
            for i in shuffle_index:
                line = triples[i]
                count += 1
                if count % 100000 == 0:
                    print "Scanned ",count," time used ",time.time() - t0
                if valid_dir  == None:
                    sum += self.gradient_decent(line[0], line[1], line[2], const_decay, L1_flag)
            if valid_dir  == None:
                print "\nCurrent sum of loss: ",sum," time used ",time.time() - t0,'\n'
            else:
                for v in valid:
                    sum += LA.norm(self.vec_e[v[0]] + self.vec_r[v[1]] - self.vec_e[v[2]])
                print "\nCurrent sum of loss on validation set: ",sum," time used ",time.time() - t0,'\n'
                sum_hist.append(sum)
            if save_every_epochs > 0 and i % save_every_epochs == 0:
                self.save(self.save_dir)
            if valid_dir ==None and sum < tol:
                print "Converged at epoch [", i,"]"
                self.save(self.save_dir)
                break
            elif valid != None and len(sum_hist) > 25 and (sum >= sum_hist[len(sum_hist) - 26] or sum < tol):
                print "Stopped at epoch [", i,"]"
                self.save(self.save_dir)
                break
    
    def train_1epoch(self, triples, shuffle_index, valid_dir=None, valid=[], const_decay=1.0, L1=False):
        count = 0
        sum = 0.0
        for i in shuffle_index:
            line = triples[i]
            count += 1
            if count % 100000 == 0:
                print "Scanned ",count
            if valid_dir  == None:
                sum += self.gradient_decent(line[0], line[1], line[2], const_decay, L1)
            else:
                self.gradient_decent(line[0], line[1], line[2], const_decay, L1)
        if valid_dir  != None:
            for v in valid:
                sum += LA.norm(self.vec_e[v[0]] + self.vec_r[v[1]] - self.vec_e[v[2]], L1)
        return sum

    def train_1epoch_shared_param(self, triples, shuffle_index, seed, counterpart_model, valid_dir=None, valid=[], const_decay=1.0, L1=False):
        count = 0
        sum = 0.0
        for i in shuffle_index:
            line = triples[i]
            count += 1
            if count % 100000 == 0:
                print "Scanned ",count
            if valid_dir  == None:
                sum += self.gradient_decent(line[0], line[1], line[2], const_decay, L1)
                s0 = seed.get(line[0])
                s1 = seed.get(line[2])
                if s0 != None:
                    counterpart_model.vec_e[s0] = self.vec_e[line[0]]
                if s1 != None:
                    counterpart_model.vec_e[s1] = self.vec_e[line[2]]
            else:
                self.gradient_decent(line[0], line[1], line[2], const_decay, L1)
                s0 = seed.get(line[0])
                s1 = seed.get(line[2])
                if s0 != None:
                    counterpart_model.vec_e[s0] = self.vec_e[line[0]]
                if s1 != None:
                    counterpart_model.vec_e[s1] = self.vec_e[line[2]]
        if valid_dir  != None:
            for v in valid:
                sum += LA.norm(self.vec_e[v[0]] + self.vec_r[v[1]] - self.vec_e[v[2]], L1)
        return sum
    
    def entity_rank(self, vec_s, vec_t, src_lan='en'):
        rst = 1
        t_dist = LA.norm(vec_s - vec_t)
        for vec in self.vec_e:
            dist = LA.norm(vec - vec_t)
            if dist > 0 and dist < t_dist:
                rst += 1
        return rst
    
    def relation_rank(self, vec_s, vec_t, src_lan='en'):
        rst = 1
        t_dist = LA.norm(vec_s - vec_t)
        for vec in self.vec_r:
            dist = LA.norm(vec - vec_t)
            if dist > 0 and dist < t_dist:
                rst += 1
        return rst
    

    #current sum of loss
    def get_loss(self, triples):
        sum = 0.0
        for line in triples:
            sum += LA.norm(self.vec_e[line[0]] + self.vec_r[line[1]] - self.vec_e[line[2]])
        return sum
    
    def save(self, filename):
        f = open(filename,'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print "Save model as ", filename
    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print "Loaded model from", filename

    '''def save_model(obj, filename):
        f = open(filename,'wb')
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print "Save model as ", filename

    def load_model(filename):
        f = open(filename,'rb')
        obj = pickle.load(f)
        print "Loaded model from", filename
        return obj'''