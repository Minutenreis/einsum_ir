PATH=/home/justus/mpich/bin:\$PATH
NUM_THREADS=144
HALF_THREADS=$((NUM_THREADS/2))

mpirun -n 2 --bind-to numa --map-by numa -env OMP_NUM_THREADS=$HALF_THREADS ./build/bench_binary_mpi