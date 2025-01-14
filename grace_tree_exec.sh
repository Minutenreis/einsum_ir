COMP_THREADS=70

if [[ -z $2 ]]
then
    /home/justus/mpich/bin/mpirun -n 2 --bind-to numa --map-by numa -env OMP_NUM_THREADS=$COMP_THREADS ./build/bench_tree_mpi $1
else
    /home/justus/mpich/bin/mpirun -n 2 --bind-to numa --map-by numa -env OMP_NUM_THREADS=$COMP_THREADS ./build/bench_tree_mpi $1 $2
fi
