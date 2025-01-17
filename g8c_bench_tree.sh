#!/bin/bash
touch dist_c.csv
touch dist_m_n.csv
touch dist_k.csv
touch local_192.csv
touch local_96_0.csv
touch local_96_1.csv

echo "size_1, gflops" > dist_c.csv
echo "size_1, gflops" > dist_m_n.csv
echo "size_1, gflops" > dist_k.csv
echo "size_1, gflops" > local_192.csv
echo "size_1, gflops" > local_96_0.csv
echo "size_1, gflops" > local_96_1.csv

for i in 2 4 8 16 32 64 96 128 180 256
do
	for j in {1..10}
	do
      /home/fedora/mpich/bin/mpirun -n 2 --bind-to numa --map-by numa -env OMP_NUM_THREADS=94 ./build/bench_tree_mpi_dist $i 94 c >> dist_c.csv
      /home/fedora/mpich/bin/mpirun -n 2 --bind-to numa --map-by numa -env OMP_NUM_THREADS=94 ./build/bench_tree_mpi_dist $i 94 m >> dist_m_n.csv
      /home/fedora/mpich/bin/mpirun -n 2 --bind-to numa --map-by numa -env OMP_NUM_THREADS=94 ./build/bench_tree_mpi_dist $i 94 k >> dist_k.csv
      OMP_NUM_THREADS=192 ./build/bench_tree_mpi_local $i 70 >> local_192.csv
      OMP_NUM_THREADS=96 OMP_PLACES={0:96} ./build/bench_tree_mpi_local $i 70 >> local_96_0.csv
      OMP_NUM_THREADS=96 OMP_PLACES={96:96} ./build/bench_tree_mpi_local $i 70 >> local_96_1.csv
	done
done
