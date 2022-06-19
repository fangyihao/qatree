#!/bin/bash
export TOKENIZERS_PARALLELISM=true
dt=`date '+%Y%m%d_%H%M%S'`

dataset=$1
shift

args=$@

seed=5
lr_schedule=fixed

ent_emb=tzw

resume_checkpoint=None
resume_id=None
random_ent_emb=false

###### Training ######
python3 -u qa_experiment.py \
    --dataset $dataset \
    --seed $seed \
    --resume_checkpoint ${resume_checkpoint} --resume_id ${resume_id} --random_ent_emb ${random_ent_emb} --ent_emb ${ent_emb//,/ } --lr_schedule ${lr_schedule} \
    --inhouse False \
    $args \

