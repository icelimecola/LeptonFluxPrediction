#!/bin/bash
#JSUB -q gpu
#JSUB -e JSUB/error/error.%J
#JSUB -o JSUB/output/output.%J
#JSUB -J elec_draw
#JSUB -gpgpu 1
#JSUB -n 1

source /public/jhinno/unischeduler/conf/jobstarter/unisched

export LD_LIBRARY_PATH=/public/soft/cuda-12.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/public/soft/cuda-12.2
module load cuda-12.2

cd /public/home/wxu.ams/LeptonFluxPrediction/Electron

~/miniconda3/envs/prediction/bin/python draw_lstm_sun5_all_mbin.py

rm .hostfile*
