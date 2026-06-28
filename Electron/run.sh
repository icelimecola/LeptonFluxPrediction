#!/bin/bash

mkdir -p JSUB/error JSUB/output Model Figure/lstm

for i in 0 1 2 3 4 5 6 7; do
  sed "s/NUM/$i/g" jsub_lstm_elec.sh > jsub_lstm_elec_$i.sh
  chmod +x jsub_lstm_elec_$i.sh
  jsub < jsub_lstm_elec_$i.sh
  echo "Submitted paras_$i.yaml"
done
