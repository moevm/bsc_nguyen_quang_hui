#!/bin/sh

cd sw-auto

cd SciencePaperAnalyzer

echo "Stopping SW-Auto"
docker-compose -f docker-compose-mlss.yml stop || true

## launch verificationapi
cd ../../verificationapi

echo "Stopping MLS Analysis Service"
docker-compose -f docker-compose-sw-auto.yml stop || true
