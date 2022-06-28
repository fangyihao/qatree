#!/bin/bash
#SBATCH --account=general
#SBATCH --partition=Aurora,Combined
#SBATCH --gres=gpu:rtx8000

CUDA_VISIBLE_DEVICES="1" ./run_qa_experiment.sh csqa --data_dir data/ -mbs 1 -hlrr "1.0" -lr "5e-6" --tree_width 1 --max_node_num 8 --max_seq_len 72
