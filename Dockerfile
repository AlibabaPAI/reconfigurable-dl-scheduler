
FROM python:3.9.12

WORKDIR /workspace/morphling

COPY . /workspace/morphling

RUN pip install --upgrade pip && pip install -r requirements.txt && pip install notebook