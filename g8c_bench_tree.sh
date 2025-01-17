#!/bin/bash
touch g8c_dist_c.csv
touch g8c_dist_m_n.csv
touch g8c_dist_k.csv
touch g8c_local_192.csv
touch g8c_local_96_0.csv
touch g8c_local_96_1.csv

echo "size_1, gflops" > g8c_dist_c.csv
echo "size_1, gflops" > g8c_dist_m_n.csv
echo "size_1, gflops" > g8c_dist_k.csv
echo "size_1, gflops" > g8c_local_192.csv
echo "size_1, gflops" > g8c_local_96_0.csv
echo "size_1, gflops" > g8c_local_96_1.csv

for i in 2 4 8 16 32 64 96 128 180 256
do
	for j in {1..10}
	do
      /home/fedora/mpich/bin/mpirun -n 2 --bind-to numa --map-by numa -env LIBXSMM_TARGET=aarch64 -env OMP_NUM_THREADS=94 ./build/bench_tree_mpi_dist $i 94 c >> g8c_dist_c.csv
      /home/fedora/mpich/bin/mpirun -n 2 --bind-to numa --map-by numa -env LIBXSMM_TARGET=aarch64 -env OMP_NUM_THREADS=94 ./build/bench_tree_mpi_dist $i 94 m >> g8c_dist_m_n.csv
      /home/fedora/mpich/bin/mpirun -n 2 --bind-to numa --map-by numa -env LIBXSMM_TARGET=aarch64 -env OMP_NUM_THREADS=94 ./build/bench_tree_mpi_dist $i 94 k >> g8c_dist_k.csv
      LIBXSMM_TARGET=aarch64 OMP_NUM_THREADS=192 ./build/bench_tree_mpi_local $i 94 >> g8c_local_192.csv
      LIBXSMM_TARGET=aarch64 OMP_NUM_THREADS=96 OMP_PLACES={0:96} ./build/bench_tree_mpi_local $i 94 >> g8c_local_96_0.csv
      LIBXSMM_TARGET=aarch64 OMP_NUM_THREADS=96 OMP_PLACES={96:96} ./build/bench_tree_mpi_local $i 94 >> g8c_local_96_1.csv
      #todo: test that 94 doesn't perform much worse than 96
	done
done
