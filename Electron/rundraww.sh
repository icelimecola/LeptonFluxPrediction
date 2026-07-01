#!/bin/bash

mkdir -p jobs JSUB/error JSUB/output JSUB/JSUBs Data/lstm2 Figure/lstm2

sed "s/NUM/0/g" jsub_draw_w.sh > JSUB/JSUBs/jsub_draw_w_0.sh
chmod +x JSUB/JSUBs/jsub_draw_w_0.sh
jsub < JSUB/JSUBs/jsub_draw_w_0.sh
echo "Submitted weighted draw job"
