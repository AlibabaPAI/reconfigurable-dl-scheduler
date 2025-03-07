#!/bin/bash

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=172.22.112.83 #172.17.105.70 #172.21.65.113 #172.21.65.68 # #172.21.64.132 #
MASTER_PORT=9001
NNODES=2
NODE_RANK=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_PATH=my-bert_text_sentence
VOCAB_FILE=/mnt/nasdata/wikidata/bert-vocab.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 16 \
       --num-layers 16 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --micro-batch-size 1 \
       --global-batch-size 16 \
       --seq-length 512 \
       --max-position-embeddings 514 \
       --train-iters 1000000 \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --eval-interval 50 \
       --eval-iters 1 \
       # --checkpoint-activations \