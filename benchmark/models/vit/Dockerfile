FROM nvcr.io/nvidia/pytorch:20.03-py3
WORKDIR /root

COPY benchmark/models/vit/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt



COPY benchmark/models/vit vit
WORKDIR /root/vit

ENV PYTHONUNBUFFERED=true
