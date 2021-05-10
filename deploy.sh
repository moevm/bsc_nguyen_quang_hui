#!/bin/sh

# update git
# sudo add-apt-repository ppa:git-core/ppa -y
# sudo apt update
# sudo apt install git

# install git large file storage
sudo apt install git-lfs -y
git lfs install

## update repo with models
git lfs pull

## get submodule code
git submodule init
git submodule update

## launch sw-auto
cd sw-auto
git checkout mls_analysis_service

cd SciencePaperAnalyzer

echo "Building SW-Auto"
docker-compose -f docker-compose-mlss.yml build

echo "Running SW-Auto"
docker-compose -f docker-compose-mlss.yml up -d

## launch verificationapi
cd ../../verificationapi

echo "Building MLS Analysis Service"
docker-compose -f docker-compose-sw-auto.yml build

echo "Running MLS Analysis Service"
docker-compose -f docker-compose-sw-auto.yml up -d
