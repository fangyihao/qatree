#!/bin/bash
#SBATCH --account=general
#SBATCH --partition=Aurora,Combined
#SBATCH --gres=gpu:rtx8000

CUDA_VISIBLE_DEVICES="2" ./run_qa_experiment.sh csqa --data_dir data/ -mbs 2 -hlrr "1.0" --tree_width 4 --encoder "roberta-large"
