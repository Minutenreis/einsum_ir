echo "threads, gflops" > test.csv
for i in {20..144..4}
do
	for j in {1..50}
	do
    		bash grace_bench_binary.sh $i >> test.csv
	done
done