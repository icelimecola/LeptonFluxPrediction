#!/bin/bash
#JSUB -q gpu
#JSUB -e JSUB/error/error.%J
#JSUB -o JSUB/output/output.%J
#JSUB -J pos_draw_w
#JSUB -gpgpu 1
#JSUB -n 1

source /public/jhinno/unischeduler/conf/jobstarter/unisched

export LD_LIBRARY_PATH=/public/soft/cuda-12.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/public/soft/cuda-12.2
module load cuda-12.2

cd /public/home/wxu.ams/LeptonFluxPrediction/Positron/step2

FLUX_SOURCE_DEFAULT="baseline"
FLUX_SOURCE="${FLUX_SOURCE:-$FLUX_SOURCE_DEFAULT}"
if [ "$FLUX_SOURCE" = "imputed" ]; then
  export POSITRON_FLUX_PATH=../step1/Data/flux/positron_flux_sun_imputed.npy
  export POSITRON_ERROR_PATH=../step1/Data/flux/positron_err_sun_imputed.npy
fi

~/miniconda3/envs/prediction/bin/python lstm_draw_w.py

rm .hostfile*
