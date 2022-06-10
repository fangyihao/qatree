#!/bin/bash
#SBATCH --account=general
#SBATCH --partition=Aurora,Combined
#SBATCH --gres=gpu:rtx8000

CUDA_VISIBLE_DEVICES="0" ./run_qa_experiment.sh csqa --data_dir data/ -mbs 3 -hlrr "0.75" --tree_width 4
