FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y python3.6 python3-pip

RUN pip3 install torch torchvision
RUN pip3 install stanza regex sentence_transformers

RUN python3 -c "import stanza; stanza.download('ru')"
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')"

ENV LANG C.UTF-8

COPY . / 