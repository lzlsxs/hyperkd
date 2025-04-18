#!/bin/sh
nohup python3 -u train_life_long.py --eval_on_train --nepochs=150 --approach=our --num-exemplars-per-class=5 --exemplar-selection=ssgd --batch_size=128 --seed=0 --patches=7 > myout 2>&1 &
