#!/bin/bash
dir=`pwd`
###############################################################################
### Main configs
### The main configs are from Megatron-LM paper
### https://arxiv.org/abs/1909.08053. Choose based on your desired model size
### or build your own configs.
seq_len=1024
global_batch_size=16
batch_size=1
lr=0.0001
min_lr=0.00001


## BERT 336M (same config as original BERT-Large model)
num_layers=16
hidden_size=1280
num_attn_heads=16
init_std=0.02

train_iters_in_million=2
train_iters=$((${train_iters_in_million} * 1000000))
###############################################################################
lr_decay_style="linear"
###############################################################################
### Parallelism configs
## Model parallelism, 1 is no MP
mp_size=1

## Pipeline parallelism. To disable PP, set pp_size to 1 and no_pp to true.
## Currently pipeline parallelism is not supported for BERT model: DeepSpeed's
## pipeline parallelism is only integrated with the GPT case, and currently
## DeepSpeed is not integrated with Megatron's own pipeline parallelism.
pp_size=1
no_pp="true"

## ZeRO stage
zero_stage=2

###############################################################################
### Misc configs
log_interval=1
eval_iters=1
eval_interval=50


## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="true"

data_path="/mnt/nasdata/megatron/my-bert_text_sentence"

vocab_path="/mnt/nasdata/wikidata/bert-vocab.txt"

num_workers=4

###############################################################################
data_options=" \
    --vocab-file ${vocab_path} \
    --data-path ${data_path} \
    --data-impl mmap"

megatron_options=" \
    --override-lr-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --tensor-model-parallel-size 1 \
    --lr-decay-iters 1000000 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 768 \
    --decoder-seq-length 768 \
    --micro-batch-size ${batch_size} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --max-position-embeddings ${seq_len} \
    --train-iters ${train_iters} \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --lr-decay-style ${lr_decay_style} \
    --split 949,50,1 \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --vocab-extra-ids 100 \
    --num-workers ${num_workers} "

if [ "${activation_checkpoint}" = "true" ]; then
megatron_options="${megatron_options} \
    --checkpoint-activations"
fi

if [ "${log_optimizer_state}" = "true" ]; then
megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi

# --cpu-optimizer \
deepspeed_options=" \
    --deepspeed \
    --deepspeed_config /mnt/nasdata/megatron/examples/t5_with_pile/ds_config_t5.json \
    --zero-stage ${zero_stage} \
    --no-pipeline-parallel "

if [[ "${no_pp}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
    --no-pipeline-parallel"
fi

if [ "${activation_checkpoint}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
    --deepspeed-activation-checkpointing"
fi


GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=172.22.112.83 #172.17.105.70 #172.21.65.113 #172.21.65.68 # #172.21.64.132 #
MASTER_PORT=9001
NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/mnt/nasdata/wikidata/my-gpt2_text_document
CHECKPOINT_PATH=.

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


# CUDA_VISIBLE_DEVICES=0,1  OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=4 taskset -c 0-63  
OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1 taskset -c 0-15  python -m torch.distributed.launch $DISTRIBUTED_ARGS  /mnt/nasdata/megatron/pretrain_t5.py ${megatron_options} ${data_options} ${deepspeed_options}
#CUDA_VISIBLE_DEVICES=1 OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=4 taskset -c 8-15  python -m torch.distributed.launch $DISTRIBUTED_ARGS  /mnt/nasdata/megatron/pretrain_t5.py ${megatron_options} ${data_options} ${deepspeed_options}
#CUDA_VISIBLE_DEVICES=2 OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=4 taskset -c 16-23  python -m torch.distributed.launch $DISTRIBUTED_ARGS  /mnt/nasdata/megatron/pretrain_t5.py ${megatron_options} ${data_options} ${deepspeed_options}
