NUM_THREADS=20
HALF_THREADS=$((NUM_THREADS/2))
mpirun -n 1 -env OMP_NUM_THREADS=$HALF_THREADS -env OMP_PLACES={0:$HALF_THREADS}             ./build/bench_binary_mpi 32 70 :\
       -n 1 -env OMP_NUM_THREADS=$HALF_THREADS -env OMP_PLACES={$HALF_THREADS:$HALF_THREADS} ./build/bench_binary_mpi 32 70