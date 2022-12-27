import copy
from itertools import zip_longest
import json
from encoder.graph2vec import featrue_extract_by_graph
from nondo_sort import NonDominatedSorting
import numpy as np
from population import Population
from ranknet import RankNet
from utils import archlist2archcode, build_train_sample, prepare_eval_folder

import torch
from individual import Individual
import logging
import random
from preprocessing.gen_graphs import build_mat_encoding, sample_arch
from darts.cnn.genotypes import Genotype
from darts.cnn.model import NetworkCIFAR as Network

from config import OPS, NUM_VERTICES
from darts.cnn.train_search import Train
from multiprocessing import Pool
import os
import subprocess


def tournament_select(pop,n_sample=2):
    pop_size  = len(pop)
    idxs = random.sample(range(pop_size),k=n_sample)
    scores_selected = [pop[i].score for i in idxs]
    id  = idxs[np.argmax(scores_selected)]
    return pop[id]

def convert_normal(arch):
    normal_name = []
    for node in arch:
        normal_name.append((str(node[0]),OPS[node[1]]))
    return normal_name

def crossover_ind(p1,p2,p_c=0.5):
    n = len(p1.X)//2
    probs = np.random.uniform(0.,1.,size=(n,))<p_c
    r1,r2 = [],[]
    for i in range(n):
        if probs[i]:
            r1.extend(copy.deepcopy(p2.X[i*2:i*2+2]))
            r2.extend(copy.deepcopy(p1.X[i*2:i*2+2]))
        else:
            r1.extend(copy.deepcopy(p1.X[i*2: i*2+2]))
            r2.extend(copy.deepcopy(p2.X[i*2:i*2+2]))
    ind1 = Individual(X=r1,age=0)
    ind2 = Individual(X=r2, age=0)
    return ind1, ind2

def mutate_ind(p,p_m=0.2):
    r = copy.deepcopy(p.X)
    n = len(r)//2
    probs = np.random.uniform(0,1,size=(n,))<p_m
    for i in range(n):
        if probs[i]:
            ops = np.random.choice(range(len(OPS)),NUM_VERTICES)
            nodes_in_normal = np.random.choice(range(i+2),2,replace=False)
            r[i*2], r[i*2+1] = (nodes_in_normal[0], ops[0]), (nodes_in_normal[1],ops[1])
    ind = Individual(X=r,age=0)
    return ind

def target(trainer,p,gpu_id,seed,ratio):
    normal_name = convert_normal(p.X)
    print(normal_name)
    rewards, rewards_test = trainer.main(seed,normal_name,train_portion=ratio,gpu=gpu_id)
    val_sum = 0
    for epoch,val_acc in rewards:
        val_sum+=val_acc
    val_avg = val_sum/len(rewards)
    return val_avg


def cos_dis(a, b):
    return 1-np.matmul(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

class ENAS(object):
    def __init__(self,nasspace,g2v_model,args):
        self.pop_size = args.pop_size
        self.total_gen = args.total_gen
        self.total_eval = args.total_eval
        self.nasspace = nasspace
        self.args = args
        self.seed = args.seed
        self.p_c = args.p_c
        self.p_m = args.p_m

        self.n_gen = 0
        self.n_eval = 0
        self.best_F = 0.
        self.best_FS = []
        self.test_F = 0.
        self.test_FS = []
        self.pop = []
        self.archive = []
        self.hash_visited = {}
        self.n_feature = 32
        self.g2v_model = g2v_model
        self.n_cluster = 20
        self.K = 10
        self.W = 4
        self.M = 6
        
        self.ranknet=None
        self.gpr = None
        self.exe_path='./tmp'
        self.n_gpus = 3

        self.trainer=Train(args.data_path)


    def initialize(self):
        while len(self.archive)<self.pop_size:
            arch = sample_arch()[0]
            hash_arch = str(arch)
            if hash_arch in self.hash_visited:
                continue
            else: self.hash_visited[hash_arch]=1
            ind = Individual(X=arch,age=0,F=None,score=None,code=self.encode_g2v(arch))
            self.archive.append(ind)
            self.n_eval+=1
        self._evaluate(self.archive)
        for p in self.archive:
            p.score = p.F
        print([p.F for p in self.archive])
        for p in self.archive:
            if p.F > self.best_F:
                self.best_F = p.F
            self.best_FS.append(self.best_F)

        #--- 构建ranknet训练的sample ---
        self.train_surrogate(self.archive)
        self.archive = sorted(self.archive,key=lambda x:x.F,reverse=True)

        #--- 构建archive_pop 和 pop
        self.archive_pop = copy.deepcopy(self.archive)
        self.pop = copy.deepcopy(self.archive_pop)
    
    def _evaluate(self,pop):
        gen_dir = os.path.join(self.exe_path,"iter_{}".format(self.n_gen))
        archs = [convert_normal(p.X) for p in pop]
        prepare_eval_folder(gen_dir,archs,n_gpus=self.n_gpus)
        subprocess.call("sh {}/run_bash.sh".format(gen_dir),shell=True)
        for i in range(len(pop)):
            with open(os.path.join(gen_dir,'net_{}.stats'.format(i)),'r') as fp:
                state = json.load(fp)
            pop[i].F = state['top1']
        

    def has_next(self):
        return self.n_eval<self.total_eval
    
    def encode_g2v(self,arch):
        edges,features = [], {}
        matrix,ops = build_mat_encoding(arch,None,return_encode=True)
        xs,ys = np.where(np.array(matrix)==1)
        hash_info = str(arch)
        xs = xs.tolist()
        ys = ys.tolist()
        for x,y in zip(xs,ys):
            edges.append([x,y])
        for id in range(len(ops)):
            features[str(id)] = str(ops[id])
        g={"edges":edges,'features':features}
        doc = featrue_extract_by_graph(g,name=hash_info)[0]
        arch_code = self.g2v_model.infer_vector(doc)
        return arch_code

    
    def solve(self):
        self.initialize()
        while self.has_next():
            self.next()
        return self.best_FS,self.pop
    
    def pop_diversity(self,pop):
        cand_X = [ind.code for ind in pop]
        n_cand = len(cand_X)
        total_dis,k=0.,0
        if len(pop)==1:
            return 0.
        for i in range(n_cand):
            for j in range(i+1,n_cand):
                dis = cos_dis(cand_X[i],cand_X[j])
                total_dis+=dis
                k+=1
        return total_dis/k
    
    def train_surrogate(self,pop):

        ## ranknet
        model_pool = [(ind.code,ind.F) for ind in pop]
        random.shuffle(model_pool)
        samples = build_train_sample(model_pool)
        self.ranknet = RankNet(self.n_feature)
        self.ranknet.fit(*samples)

    
    def predict(self,pop,return_std=False):
        scores = []
        xembedding = [ind.code for ind in pop]
        predicted= np.squeeze(self.ranknet.predict(xembedding).detach().cpu().numpy())
        if return_std:
            for _ in range(5):
                xembedding = [self.encode_g2v(ind.X) for ind in pop]
                score = np.squeeze(self.ranknet.predict(xembedding).detach().cpu().numpy())
                scores.append(score)
            return predicted,np.std(scores,axis=0)
        else:
            return predicted

    def gen_offspring(self):
        offspring = []
        hash_visited = copy.deepcopy(self.hash_visited)
        logging.info("size of pop:{}".format(len(self.pop)))
        num_mutated = 0
        offspring_size = self.pop_size * self.M
        for ind in self.pop:
            hash_ind = str(ind.X)
            if hash_ind not in hash_visited:
                hash_visited[hash_ind]=1
        logging.info("size of hash_visited:{}".format(len(hash_visited)))

        # generate the offspring
        patience = 100
        num_mutated = 0
        while len(offspring)<self.pop_size*self.M:
            if patience==0:
                break
            p1,p2 = tournament_select(self.pop,n_sample=2),tournament_select(self.pop,n_sample=2)
            p1,p2 = crossover_ind(p1,p2,self.p_c)
            p1 = mutate_ind(p1,self.p_m)
            p2 = mutate_ind(p2,self.p_m)
            hash_p1,hash_p2 = str(p1.X), str(p2.X)
            if (hash_p1 in hash_visited) and (hash_p2 in hash_visited):
                patience-=1
            if hash_p1 not in hash_visited:
                offspring.append(p1)
                hash_visited[hash_p1]=1
                patience=100
            if hash_p2 not in hash_visited:
                offspring.append(p2)
                hash_visited[hash_p2]=1
                patience=100
            num_mutated+=1
        if len(offspring)<self.pop_size*self.M:
            logging.info("The number of offspring is insufficient, uniform sample")
        while len(offspring)<self.pop_size*self.M:
            arch = sample_arch()[0]
            hash_arch = str(arch)
            if hash_arch not in hash_visited:
                offspring.append(Individual(X=arch,age=0))
                hash_visited[hash_arch]=1
        logging.info("offspring if ready")
        return offspring

    def env_select(self,offspring):
        mixed = Population.merge(self.pop,offspring)
        for ind in mixed:
            if ind.code is None:
                ind.code = self.encode_g2v(ind.X)
        scores = self.predict(mixed)
        for i in range(len(mixed)):
            mixed[i].score = scores[i]
        diss = np.full((self.pop_size,self.pop_size*self.M),np.inf)
        n_update = 0
        for i in range(self.pop_size):
            for j in range(self.pop_size*self.M):
                diss[i,j] = cos_dis(self.pop[i].code, offspring[j].code)
        associate_stat = np.full((self.pop_size,),0)
        associate_list = [[] for i in range(self.pop_size)]
        for _ in range(self.pop_size*self.M):
            xs,ys  = np.where(diss==np.min(diss))
            x,y =  xs[0],ys[0]
            associate_stat[x]+=1
            associate_list[x].append(y)
            diss[:,y]=np.inf
            if associate_stat[x]==self.M:
                diss[x,:]=np.inf
        for i in range(self.pop_size):
            associate =  [offspring[j] for j in associate_list[i]]
            associate_scores = [ind.score for ind in associate]
            best_id = np.argmax(associate_scores)
            if self.n_gen%self.W==0:
                self.pop[i] = associate[best_id]
                n_update+=1
            elif self.pop[i].score<associate[best_id].score:
                self.pop[i] = associate[best_id]
                n_update+=1
        logging.info("gen:{} n_update:{}".format(self.n_gen+1,n_update))
        logging.info("scores:{}".format(np.sort(-scores)[:self.pop_size]))

    def infill_all(self):
        self.pop = sorted(self.pop,key=lambda x:x.score, reverse=True)
        scores_infill,uncerit_infill = self.predict(self.pop,return_std=True)
        logging.info("socres_infill:{}".format(scores_infill))
        logging.info("unverit_info:{}".format(uncerit_infill))
        for i in range(len(self.pop)):
            self.pop[i].score = scores_infill[i]
            self.pop[i].uncerit = uncerit_infill[i]
        
        ids_sorted = self.infill(self.pop)
        k=0
        new_candidate = []
        to_eval_pop = []
        for id in ids_sorted:
            ind = self.pop[id]
            if ind.F is None:
                to_eval_pop.append(ind)
                self.archive.append(ind)
                new_candidate.append(ind)
                self.hash_visited[str(ind.X)]=1
                self.n_eval+=1
                k+=1
            if k>=self.K:
                break
        self._evaluate(to_eval_pop)
        self.train_surrogate(self.archive)

        ## 更新下一代种群
        num_resample = len(new_candidate)
        self.archive_pop = sorted(self.archive_pop,key=lambda x:x.F,reverse=True)
        n_absolate = self.pop_size -  num_resample
        diss = np.full((num_resample,num_resample),0.)
        for i in range(num_resample):
            for j in range(num_resample):
                diss[i,j] = cos_dis(self.archive_pop[n_absolate+i].code, new_candidate[j].code)
        n_update = 0
        for i in range(num_resample):
            xs,ys = np.where(diss==np.min(diss))
            x,y = xs[0], ys[0]
            if self.archive_pop[n_absolate+x].F <= new_candidate[y].F:
                self.archive_pop[n_absolate+x] = new_candidate[y]
                n_update+=1
            diss[:,y] = np.inf
            diss[x,:] = np.inf
        self.pop = copy.deepcopy(self.archive_pop)
        for ind in self.pop:
            ind.score = ind.F
        
        logging.info("gen:{} archive_pop n_update:{}".format(self.n_gen+1,n_update))

        ## ---显示搜索到的最优个体
        self.archive = sorted(self.archive,key=lambda x:x.F, reverse=True)
        self.best_F = self.archive[0].F
        self.best_FS.extend([self.best_F]*k)
        logging.info("gen:{} n_eval:{} best_F:{}".format(self.n_gen, self.n_eval, self.best_F))
    
    def next(self):
        offspring = self.gen_offspring()
        self.env_select(offspring)
        if (self.n_gen+1)%self.W==0:
            self.infill_all()
        self.n_gen+=1
    
    def infill(self,pop):
        scores = [ind.score for ind in pop]
        uncerit = [ind.uncerit for ind in pop]
        su = np.array([scores, uncerit])
        F, rank = NonDominatedSorting(su)
        selected_id = []
        logging.info("nondo_sort F:{}".format(F))
        for i in range(len(F)):
            selected_id.extend(F[i])
        return selected_id
        


