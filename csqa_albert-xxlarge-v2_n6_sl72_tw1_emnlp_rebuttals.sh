#!/bin/bash
#SBATCH --account=general
#SBATCH --partition=Aurora,Combined
#SBATCH --gres=gpu:rtx8000

CUDA_VISIBLE_DEVICES="1" ./run_qa_experiment.sh csqa --data_dir data/ -mbs 1 -hlrr "1.0" -lr "8e-6" --tree_width 1 --max_node_num 6 --max_seq_len 72 --inhouse True --random_walk True --random_walk_sample_rate 0.8

#CUDA_VISIBLE_DEVICES="1" ./run_qa_experiment.sh csqa --data_dir data/ -mbs 8 -hlrr "1.0" -lr "8e-6" --tree_width 1 --max_node_num 6 --max_seq_len 72 --inhouse True --mode eval --load_model_path run/csqa/20220901-090732_csqa_albert-xxlarge-v2-emnlp-rebuttals/model.pt.2.79.0 --voting_method entropy --random_walk True --random_walk_sample_rate 0.8
