#!/bin/bash
touch grace_dist_c.csv
touch grace_dist_m_n.csv
touch grace_dist_k.csv
touch grace_local_144.csv
touch grace_local_72_0.csv
touch grace_local_72_1.csv

echo "size_1, gflops" > grace_dist_c.csv
echo "size_1, gflops" > grace_dist_m_n.csv
echo "size_1, gflops" > grace_dist_k.csv
echo "size_1, gflops" > grace_local_144.csv
echo "size_1, gflops" > grace_local_72_0.csv
echo "size_1, gflops" > grace_local_72_1.csv

for i in 2 4 8 16 32 64 72 128 180 256
do
	for j in {1..10}
	do
      /home/justus/mpich/bin/mpirun -n 2 --bind-to numa --map-by numa -env LIBXSMM_TARGET=aarch64 -env OMP_NUM_THREADS=70 ./build/bench_tree_mpi_dist $i 70 c >> grace_dist_c.csv
      /home/justus/mpich/bin/mpirun -n 2 --bind-to numa --map-by numa -env LIBXSMM_TARGET=aarch64 -env OMP_NUM_THREADS=70 ./build/bench_tree_mpi_dist $i 70 m >> grace_dist_m_n.csv
      /home/justus/mpich/bin/mpirun -n 2 --bind-to numa --map-by numa -env LIBXSMM_TARGET=aarch64 -env OMP_NUM_THREADS=70 ./build/bench_tree_mpi_dist $i 70 k >> grace_dist_k.csv
      LIBXSMM_TARGET=aarch64 OMP_NUM_THREADS=144 ./build/bench_tree_mpi_local $i 70 >> grace_local_144.csv
      LIBXSMM_TARGET=aarch64 OMP_NUM_THREADS=72 OMP_PLACES={0:72} ./build/bench_tree_mpi_local $i 70 >> grace_local_72_0.csv
      LIBXSMM_TARGET=aarch64 OMP_NUM_THREADS=72 OMP_PLACES={72:72} ./build/bench_tree_mpi_local $i 70 >> grace_local_72_1.csv
      #todo: test that 70 doesn't perform much worse than 72
	done
done
