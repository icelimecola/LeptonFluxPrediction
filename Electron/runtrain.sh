#!/bin/bash

mkdir -p jobs JSUB/error JSUB/output JSUB/JSUBs Data/model Data/hyperpara Figure/lstmtrain

for i in 0 1 2 3 4 5 6 7; do
  sed "s/NUM/$i/g" jsub_train.sh > JSUB/JSUBs/jsub_train_$i.sh
  chmod +x JSUB/JSUBs/jsub_train_$i.sh
  jsub < JSUB/JSUBs/jsub_train_$i.sh
  echo "Submitted paras_$i.yaml"
done
