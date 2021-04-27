# FROM ubuntu:18.04
FROM nvidia/cuda:11.2.2-base-ubuntu20.04

RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN apt-get update
RUN apt-get install -y python3.6 python3-pip
RUN python3 -m pip install --upgrade pip

ENV PYTHONUNBUFFERED 1
ENV PYTHONIOENCODING utf-8

RUN pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

COPY download_pretrained_model.py /download_pretrained_model.py
RUN python3 download_pretrained_model.py

RUN pip3 install stanza-batch

RUN pip3 install watchdog==0.8.3
RUN pip3 install PyYAML==3.12
RUN pip3 install argh==0.26.2

COPY app /app

ENV PYTHONPATH=/app
WORKDIR /app

# RUN cd /usr/local/lib/python3.6/dist-packages && \
#     python3 /app/setup.py develop
RUN pip3 install -e .

# CMD ["pserve", "development.ini", "--reload"]
