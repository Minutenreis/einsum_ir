#!/bin/bash
touch grace_mw_comp_c0.csv

echo "size_1, gflops" > grace_mw_comp_c0.csv

for i in 2 4 8 16 32 64 72 128 180 256 384
do
      for j in {1..10}
      do
            /home/justus/mpich/bin/mpirun -n 2 --bind-to numa --map-by numa -env LIBXSMM_TARGET=aarch64 -env OMP_NUM_THREADS=70 ./build/bench_tree_mpi_dist $i 70 >> grace_mw_comp_c0.csv
      done
done