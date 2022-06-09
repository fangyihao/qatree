#!/bin/bash
export TOKENIZERS_PARALLELISM=true
dt=`date '+%Y%m%d_%H%M%S'`


dataset=$1
shift
#encoder='bert-base-uncased'
encoder='roberta-base'
#encoder='roberta-large'
#encoder='google/mobilebert-uncased'
args=$@


bs=256
mbs=8

nc=5
max_node_num=2
seed=5
lr_schedule=fixed
psl=32

if [ ${dataset} = obqa ]
then
  n_epochs=70
  max_epochs_before_stop=10
else
  n_epochs=300
  max_epochs_before_stop=10
fi

max_seq_len=48
ent_emb=tzw

resume_checkpoint=None
resume_id=None
random_ent_emb=false

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $encoder"
echo "batch_size: $bs mini_batch_size: $mbs"
echo "learning_rate: $lr"
echo "******************************"

save_dir_pref='runs'
mkdir -p $save_dir_pref

run_name=TreeLM__ds_${dataset}__enc_${encoder}__sd${seed}__${dt}
# log=logs/train_${dataset}__${run_name}.log.txt

###### Training ######
python3 -u qa_experiment.py \
    --dataset $dataset \
    --encoder $encoder -bs $bs --seed $seed -mbs ${mbs} -sl ${max_seq_len} --max_node_num ${max_node_num} \
    --n_epochs $n_epochs --max_epochs_before_stop ${max_epochs_before_stop} \
    --save_dir ${save_dir_pref}/${dataset}/${run_name} \
    --run_name ${run_name}\
    --resume_checkpoint ${resume_checkpoint} --resume_id ${resume_id} --random_ent_emb ${random_ent_emb} --ent_emb ${ent_emb//,/ } --lr_schedule ${lr_schedule} \
    --pre_seq_len ${psl} --num_choices ${nc} --inhouse False \
    $args \
# > ${log} 2>&1 &
# echo log: ${log}
