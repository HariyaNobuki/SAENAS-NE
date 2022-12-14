#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_0.subnet --save ./tmp\iter_0\net_0.stats --net_id 0 &
CUDA_VISIBLE_DEVICES=1 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_1.subnet --save ./tmp\iter_0\net_1.stats --net_id 1 &
CUDA_VISIBLE_DEVICES=2 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_2.subnet --save ./tmp\iter_0\net_2.stats --net_id 2 &
CUDA_VISIBLE_DEVICES=3 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_3.subnet --save ./tmp\iter_0\net_3.stats --net_id 3 &
CUDA_VISIBLE_DEVICES=4 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_4.subnet --save ./tmp\iter_0\net_4.stats --net_id 4 &
CUDA_VISIBLE_DEVICES=5 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_5.subnet --save ./tmp\iter_0\net_5.stats --net_id 5 &
CUDA_VISIBLE_DEVICES=6 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_6.subnet --save ./tmp\iter_0\net_6.stats --net_id 6 &
CUDA_VISIBLE_DEVICES=7 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_7.subnet --save ./tmp\iter_0\net_7.stats --net_id 7 &
wait
CUDA_VISIBLE_DEVICES=0 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_8.subnet --save ./tmp\iter_0\net_8.stats --net_id 8 &
CUDA_VISIBLE_DEVICES=1 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_9.subnet --save ./tmp\iter_0\net_9.stats --net_id 9 &
CUDA_VISIBLE_DEVICES=2 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_10.subnet --save ./tmp\iter_0\net_10.stats --net_id 10 &
CUDA_VISIBLE_DEVICES=3 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_11.subnet --save ./tmp\iter_0\net_11.stats --net_id 11 &
CUDA_VISIBLE_DEVICES=4 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_12.subnet --save ./tmp\iter_0\net_12.stats --net_id 12 &
CUDA_VISIBLE_DEVICES=5 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_13.subnet --save ./tmp\iter_0\net_13.stats --net_id 13 &
CUDA_VISIBLE_DEVICES=6 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_14.subnet --save ./tmp\iter_0\net_14.stats --net_id 14 &
CUDA_VISIBLE_DEVICES=7 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_15.subnet --save ./tmp\iter_0\net_15.stats --net_id 15 &
wait
CUDA_VISIBLE_DEVICES=0 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_16.subnet --save ./tmp\iter_0\net_16.stats --net_id 16 &
CUDA_VISIBLE_DEVICES=1 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_17.subnet --save ./tmp\iter_0\net_17.stats --net_id 17 &
CUDA_VISIBLE_DEVICES=2 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_18.subnet --save ./tmp\iter_0\net_18.stats --net_id 18 &
CUDA_VISIBLE_DEVICES=3 python darts/cnn/evaluation.py --subnet ./tmp\iter_0\net_19.subnet --save ./tmp\iter_0\net_19.stats --net_id 19 &
wait
