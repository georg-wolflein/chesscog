FROM python:3

RUN pip install --upgrade pip
RUN mkdir -p /chess /data
WORKDIR /chess
ENV DATA_DIR /data

