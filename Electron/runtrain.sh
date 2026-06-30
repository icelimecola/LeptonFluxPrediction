#!/bin/bash

mkdir -p jobs JSUB/error JSUB/output JSUBs Model Figure/lstm

for i in 0 1 2 3 4 5 6 7; do
  sed "s/NUM/$i/g" jsub_train.sh > JSUBs/jsub_train_$i.sh
  chmod +x JSUBs/jsub_train_$i.sh
  jsub < JSUBs/jsub_train_$i.sh
  echo "Submitted paras_$i.yaml"
done
