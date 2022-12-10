
import json
import os
import logging
import argparse
from utils import archlist2archcode
from evolution import ENAS, crossover_ind
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from nasspace import Nasbench301
from scipy.stats import kendalltau
from gensim.models.doc2vec import Doc2Vec


logging.basicConfig(level=logging.INFO,
                    filename='log.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(levelname)s: %(message)s')

cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = True

parser = argparse.ArgumentParser("nasbench")
parser.add_argument("--classes",type=int,default=10)
parser.add_argument("--layers",type=int,default=2)
parser.add_argument("--channels",type=int,default=32) #32
parser.add_argument("--nodes",type=int,default=4)
parser.add_argument("--init_lr",type=float,default=0.025)
parser.add_argument("--momentum",type=float,default=0.9)
parser.add_argument("--weight_decay",type=float,default=3e-4)
parser.add_argument("--epochs",type=int,default=150)
parser.add_argument("--root",type=str,default='/data/Fanliang/data')
# parser.add_argument("--batch_size",type=int,default=64) # train
parser.add_argument("--batch_size",type=int,default=256) # kd
parser.add_argument("--seed",type=int,default=0) # train
# parser.add_argument("--seed",type=int,default=2020) # kd
parser.add_argument("--ratio",type=float,default=0.8)
parser.add_argument("--keep_prob",type=float,default=1.0)
parser.add_argument("--drop_path_keep_prob",type=float,default=0.9)
parser.add_argument('--use_aux_head', action='store_true', default=False)

parser.add_argument("--n_kd_sample",type=int,default=20)

parser.add_argument("--n_sample",type=int,default=3)
parser.add_argument("--pop_size",type=int,default=20) #m
parser.add_argument("--total_gen",type=int,default=80) #m
parser.add_argument("--total_eval",type=int,default=300)
parser.add_argument("--p_c",type=float,default=0.5)
parser.add_argument("--p_m",type=float,default=0.2)
parser.add_argument("--pool_limit",type=int,default=50)

parser.add_argument("--up_epochs",type=int,default=5)
parser.add_argument("--up_lr",type=float,default=0.001)
parser.add_argument("--data-path",type=str,default="data/cifar-10")

args = parser.parse_args()
# graph2vec
g2v_model =  Doc2Vec.load("g2v_model/doc2vec_model_dim32.model")


seed = np.random.randint()
logging.info("seed:{}".format(seed))
nasspace = Nasbench301()
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

enas = ENAS(nasspace=nasspace,g2v_model=g2v_model,args=args)
best_FS,pop = enas.solve()
with open("trace.json".format(seed),'w') as fp:
    json.dump(best_FS,fp)
torch.save({"pop":pop},'best_pop.pt')

    