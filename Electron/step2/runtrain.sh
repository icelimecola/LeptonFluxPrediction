#!/bin/bash

mkdir -p jobs JSUB/error JSUB/output JSUB/JSUBs Data/model Data/hyperpara Figure/lstmtrain

for config_path in $(find Data/hyperpara -maxdepth 1 -name 'paras_*.yaml' | sort -V); do
  config_file=${config_path##*/}
  i=${config_file#paras_}
  i=${i%.yaml}
  sed "s/NUM/$i/g" jsub_train.sh > JSUB/JSUBs/jsub_train_$i.sh
  chmod +x JSUB/JSUBs/jsub_train_$i.sh
  jsub < JSUB/JSUBs/jsub_train_$i.sh
  echo "Submitted paras_$i.yaml"
done
