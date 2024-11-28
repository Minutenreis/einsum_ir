mpirun -n 1 -genv OMP_NUM_THREADS=72 -genv OMP_PLACES={0:72}  ./build/bench_binary_mpi :\
       -n 1 -genv OMP_NUM_THREADS=72 -genv OMP_PLACES={72:72} ./build/bench_binary_mpi