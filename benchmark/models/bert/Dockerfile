FROM nvcr.io/nvidia/pytorch:20.03-py3
WORKDIR /root

COPY benchmark/models/bert/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt



COPY benchmark/models/train_code/megatron megatron
WORKDIR /root/megatron

ENV PYTHONUNBUFFERED=true
