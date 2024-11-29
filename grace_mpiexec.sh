PATH=/home/justus/mpich/bin:\$PATH
NUM_THREADS=144
HALF_THREADS=$((NUM_THREADS/2))
mpirun -n 1 -env OMP_PROC_BIND true -env OMP_NUM_THREADS=$HALF_THREADS -env OMP_PLACES={0:$HALF_THREADS}             ./build/bench_binary_mpi :\
       -n 1 -env OMP_PROC_BIND true -env OMP_NUM_THREADS=$HALF_THREADS -env OMP_PLACES={$HALF_THREADS:$HALF_THREADS} ./build/bench_binary_mpi

# echo mpirun -n 2 -bind-to core:$HALF_THREADS -map-by socket -genv OMP_PROC_BIND true -genv OMP_NUM_THREADS=$HALF_THREADS ./build/bench_binary_mpi