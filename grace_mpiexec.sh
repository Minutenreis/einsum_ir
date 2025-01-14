COMP_THREADS=70

/home/justus/mpich/bin/mpirun -n 2 --bind-to numa --map-by numa -env OMP_NUM_THREADS=$COMP_THREADS ./build/bench_binary_mpi