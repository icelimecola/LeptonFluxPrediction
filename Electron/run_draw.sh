#!/bin/bash

mkdir -p JSUB/error JSUB/output Data/lstm2 Figure/lstm2

sed "s/NUM/0/g" jsub_draw.sh > jsub_draw_0.sh
chmod +x jsub_draw_0.sh
jsub < jsub_draw_0.sh
echo "Submitted draw job"
