#!/bin/bash
#SBATCH --account=general
#SBATCH --partition=Aurora,Combined
#SBATCH --gres=gpu:rtx8000

CUDA_VISIBLE_DEVICES="1" ./run_qa_experiment.sh csqa --data_dir data/ -mbs 1 -hlrr "1.0" -lr "8e-6" --tree_width 1 --max_node_num 6 --max_seq_len 72 --inhouse True

#CUDA_VISIBLE_DEVICES="1" ./run_qa_experiment.sh csqa --data_dir data/ --mode eval --load_model_path /home/yfang/workspace/qatree/run/csqa/20220621-195515_csqa_albert-xxlarge-v2_emnlp/model.pt.1 -mbs 8 -hlrr "1.0" -lr "8e-6" --tree_width 1 --max_node_num 6 --max_seq_len 72 --inhouse True
