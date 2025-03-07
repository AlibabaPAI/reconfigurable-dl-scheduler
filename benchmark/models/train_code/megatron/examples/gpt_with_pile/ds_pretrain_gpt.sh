#!/bin/bash





dir=`pwd`
###############################################################################
### Main configs
### The main configs are from Megatron-LM paper
### https://arxiv.org/abs/1909.08053. Choose based on your desired model size
### or build your own configs.
seq_len=1024
global_batch_size=16
batch_size=2
lr=0.00015
min_lr=1e-5

num_layers=16
hidden_size=1280
num_attn_heads=16
init_std=0.02

mp_size=1
pp_size=1
no_pp="true"

## ZeRO stage
zero_stage=2


###############################################################################
### Misc configs
log_interval=1
eval_iters=10
eval_interval=1000

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="true"
###############################################################################
### Output and data configs
host="${HOSTNAME}"

## Public the Pile dataset, see prepare_pile_data.py in the same directory
## about how to download and preprocess the data.


data_path="/mnt/nasdata/wikidata/my-gpt2_text_document"

data_options=" \
    --vocab-file /mnt/nasdata/wikidata/gpt2-vocab.json \
    --merge-file /mnt/nasdata/wikidata/gpt2-merges.txt \
    --data-path ${data_path} \
    --data-impl mmap"

megatron_options=" \
    --override-lr-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --tensor-model-parallel-size 1 \
    --lr-decay-iters 320000 \
    --micro-batch-size ${batch_size} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --seq-length ${seq_len} \
    --max-position-embeddings ${seq_len} \
    --train-iters 500000 \
    --lr ${lr} \
    --lr-decay-style cosine \
    --min-lr ${min_lr} \
    --split 949,50,1 \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --lr-warmup-fraction .01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 "

if [ "${activation_checkpoint}" = "true" ]; then
megatron_options="${megatron_options}"
fi

if [ "${log_optimizer_state}" = "true" ]; then
megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi
#    --cpu-optimizer \
deepspeed_options=" \
--cpu-optimizer \

    --deepspeed \
    --deepspeed_config /mnt/nasdata/megatron/examples/gpt_with_pile/ds_config_gpt.json \
    --zero-stage ${zero_stage} \
    --no-pipeline-parallel "

if [[ "${no_pp}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
    --no-pipeline-parallel"
fi

if [ "${activation_checkpoint}" = "true" ]; then
deepspeed_options="${deepspeed_options}   --deepspeed-activation-checkpointing"
fi


GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost #172.17.105.70 #172.21.65.68 # #172.21.64.132 #
MASTER_PORT=9001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/mnt/nasdata/wikidata/my-gpt2_text_document
CHECKPOINT_PATH=.

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


# CUDA_VISIBLE_DEVICES=0,1,2,3 
OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=4 taskset -c 0-63 python -m torch.distributed.launch $DISTRIBUTED_ARGS  /mnt/nasdata/megatron/pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options}