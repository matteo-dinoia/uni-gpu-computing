#!/usr/bin/bash
module load cuda/12.1;

nvcc --gpu-architecture=sm_80 -m64 -o compiled.exec $1 || exit;
sbatch gpu-run.sbatch;

sleep 3;
cat result.out;
