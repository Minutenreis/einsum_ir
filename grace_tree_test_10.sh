#!/bin/bash
touch test.csv
echo "size_1, gflops, gflops_mpi, speedup" > test.csv
for i in 2 4 8 16 32 64 96 128 180 256
do
	for j in {1..10}
	do
    		bash grace_tree_exec.sh $i 70 >> test.csv
	done
done
