FROM nvcr.io/nvidia/pytorch:23.05-py3
LABEL authors="kindroach"

COPY requirements.txt .

RUN python -m pip install -r requirements.txt
