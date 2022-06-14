#!/bin/bash
#SBATCH --account=general
#SBATCH --partition=Aurora,Combined
#SBATCH --gres=gpu:rtx8000

CUDA_VISIBLE_DEVICES="0" ./run_qa_experiment.sh obqa --data_dir data/ -mbs 2 -hlrr "1.0" --tree_width 4
