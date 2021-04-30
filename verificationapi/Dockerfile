# FROM ubuntu:18.04
FROM nvidia/cuda:11.2.2-base-ubuntu20.04

RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN apt-get update && apt-get install -y python3.8 python3-pip
RUN python3 -m pip install pip==21.0.1

ENV PYTHONUNBUFFERED 1
ENV PYTHONIOENCODING utf-8

RUN pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

COPY download_pretrained_model.py /download_pretrained_model.py
RUN python3 download_pretrained_model.py

# flower does not work with celery 5.0.5 at the moment -> upgrade later
# RUN pip3 install celery==5.0.5

# RUN pip3 install https://github.com/mher/flower/zipball/master#egg=flower
COPY app /app

ENV PYTHONPATH=/app
WORKDIR /app

# RUN python3 --version
# RUN cd /usr/local/lib/python3.8/dist-packages && \
#     python3 /app/setup.py develop
RUN pip3 install -e .

# CMD ["pserve", "development.ini", "--reload"]
