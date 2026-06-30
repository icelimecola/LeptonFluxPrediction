#!/bin/bash

mkdir -p jobs JSUB/error JSUB/output JSUBs Data/lstm2 Figure/lstm2

sed "s/NUM/0/g" jsub_draw.sh > JSUBs/jsub_draw_0.sh
chmod +x JSUBs/jsub_draw_0.sh
jsub < JSUBs/jsub_draw_0.sh
echo "Submitted draw job"
