#!/bin/bash


GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=172.22.112.83 #172.17.105.70 #172.21.65.113 #172.21.65.68 # #172.21.64.132 #
MASTER_PORT=9001
NNODES=2
NODE_RANK=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=my-bert_text_sentence
VOCAB_FILE=/mnt/nasdata/wikidata/bert-vocab.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_t5.py \
       --tensor-model-parallel-size 16 \
       --pipeline-model-parallel-size 1 \
       --num-layers 16 \
       --hidden-size 1280 \
       --num-attention-heads 16 \
       --kv-channels 64 \
       --ffn-hidden-size 3072 \
       --encoder-seq-length 768 \
       --decoder-seq-length 768 \
       --vocab-extra-ids 100 \
       --micro-batch-size 16 \
       --global-batch-size 16 \
       --max-position-embeddings 1024 \
       --train-iters 1000000 \
       --lr-decay-iters 1000000 \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 1 \
       --eval-interval 5000 \
       --eval-iters 1 \
       --checkpoint-activations \


