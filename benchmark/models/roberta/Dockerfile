FROM nvcr.io/nvidia/pytorch:20.03-py3
WORKDIR /root

COPY benchmark/models/roberta/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt



COPY benchmark/models/roberta roberta
WORKDIR /root/roberta

ENV PYTHONUNBUFFERED=true
