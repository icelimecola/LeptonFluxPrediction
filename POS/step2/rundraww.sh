#!/bin/bash

mkdir -p jobs JSUB/error JSUB/output JSUB/JSUBs Data/modelw Data/lstmdraww Figure/lstmdraww

flux_source="${FLUX_SOURCE:-baseline}"

sed \
  -e "s#NUM#0#g" \
  -e "s#FLUX_SOURCE_DEFAULT=\"baseline\"#FLUX_SOURCE_DEFAULT=\"${flux_source}\"#g" \
  jsub_draw_w.sh > JSUB/JSUBs/jsub_draw_w_0.sh
chmod +x JSUB/JSUBs/jsub_draw_w_0.sh
jsub < JSUB/JSUBs/jsub_draw_w_0.sh
echo "Submitted weighted draw job with FLUX_SOURCE=${flux_source}"
