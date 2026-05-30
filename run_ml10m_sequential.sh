#!/usr/bin/env bash

nohup ./run_hpt_parallel.sh movielens_10m p4_n1 4 200 2>&1 | tee logs/seq_p4n1.log
nohup ./run_hpt_parallel.sh movielens_10m p3_n2 4 200 2>&1 | tee logs/seq_p3n2.log
nohup ./run_hpt_parallel.sh movielens_10m p5_n1 4 200 2>&1 | tee logs/seq_p5n1.log
nohup ./run_hpt_parallel.sh movielens_20m p3_n2 4 200 2>&1 | tee logs/seq_p3n2.log
nohup ./run_hpt_parallel.sh movielens_20m p5_n1 4 200 2>&1 | tee logs/seq_p5n1.log
