#!/bin/bash

mkdir -p jobs JSUB/error JSUB/output JSUB/JSUBs Data/modelw Data/hyperpara Figure/lstmtrainw

for config_path in $(find Data/hyperpara -maxdepth 1 -name 'paras_*.yaml' | sort -V); do
  config_file=${config_path##*/}
  i=${config_file#paras_}
  i=${i%.yaml}
  sed "s/NUM/$i/g" jsub_train_w.sh > JSUB/JSUBs/jsub_train_w_$i.sh
  chmod +x JSUB/JSUBs/jsub_train_w_$i.sh
  jsub < JSUB/JSUBs/jsub_train_w_$i.sh
  echo "Submitted sun-only weighted paras_$i.yaml"
done
