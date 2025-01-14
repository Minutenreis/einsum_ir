#!/bin/bash
touch test.csv
echo "size_1, gflops, gflops_mpi, speedup" > test.csv
for i in {2..128..2}
do
	for j in {1..100}
	do
    		bash grace_tree_exec.sh $i 70 >> test.csv
	done
done
