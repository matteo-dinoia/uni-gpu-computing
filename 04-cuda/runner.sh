#!/usr/bin/bash
module load CUDA/12.3.2 || exit;

nvcc --gpu-architecture=sm_80 -m64 -o compiled.exec $1 || exit;

rm result.err
rm result.out

sbatch gpu-run.sbatch || exit;

#echo "WAITING: Put in queue."
#sleep 5;
until [ -e ./result.out ]; do sleep 1; done
cat ./result.out
