python -m torch.distributed.launch --nproc_per_node=8 --master_port 9996 test_allreduce.py \
--global_tp_deg 8 \
--global_tp_consec 1 \
--pp_deg 1 \
--local_batch_size 8