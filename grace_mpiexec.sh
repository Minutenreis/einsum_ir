# export PATH=/home/justus/Linux_aarch64/24.11/comm_libs/mpi/bin:\$PATH
PATH=/home/justus/mpich/bin:\$PATH
NUM_THREADS=144
COMM_THREADS_PER_PROCESS=3
THREADS_PER_NODE=$((NUM_THREADS/2-COMM_THREADS_PER_PROCESS))

mpirun -n 2 --bind-to numa --map-by numa -env OMP_NUM_THREADS=$THREADS_PER_NODE ./build/bench_binary_mpi