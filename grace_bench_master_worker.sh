#!/bin/bash
touch grace_master_worker.csv

echo "size_1, gflops" > grace_master_worker.csv

for i in 2 4 8 16 32 64 72 128 180 256 384
do
/home/justus/mpich/bin/mpirun -n 2 --bind-to numa --map-by numa -env LIBXSMM_TARGET=aarch64 -env OMP_NUM_THREADS=70 ./build/bench_binary_mpi $i 70 >> grace_master_worker.csv
done
