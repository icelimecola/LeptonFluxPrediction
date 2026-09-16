#!/bin/bash
#JSUB -q gpu
#JSUB -e JSUB/error/error.%J
#JSUB -o JSUB/output/output.%J
#JSUB -J saits_impute_15st
#JSUB -gpgpu 1
#JSUB -n 1

source /public/jhinno/unischeduler/conf/jobstarter/unisched

cd /public/home/wxu.ams/LeptonFluxPrediction/NM

mkdir -p logs JSUB/error JSUB/output

exec > logs/saits_impute_15st.log 2>&1
set -x
hostname
pwd
date

PY=/public/home/wxu.ams/miniconda3/envs/prediction/bin/python
STATIONS=TERA,SOPB,SOPO,FSMT,INVK,NAIN,PWNK,THUL,APTY,OULU,YKTK,NEWK,LMKS,JUNG
# STATIONS=TERA,SOPB,SOPO,FSMT,INVK,NAIN,PWNK,THUL,APTY,OULU,YKTK,NEWK,LMKS,JUNG,MXCO
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

$PY -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

$PY nmdb_impute_saits.py --window 365 --epochs 100 --mask-rate 0.3 --seed 17 --stations $STATIONS

set +x
date
rm -f .hostfile*
