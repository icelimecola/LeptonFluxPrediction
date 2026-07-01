#!/bin/bash

mkdir -p jobs JSUB/error JSUB/output JSUB/JSUBs Data/modelw Data/hyperpara Figure/lstm

for i in 0 1 2 3 4 5 6 7; do
  sed "s/NUM/$i/g" jsub_train_w.sh > JSUB/JSUBs/jsub_train_w_$i.sh
  chmod +x JSUB/JSUBs/jsub_train_w_$i.sh
  jsub < JSUB/JSUBs/jsub_train_w_$i.sh
  echo "Submitted weighted paras_$i.yaml"
done
