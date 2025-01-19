import numpy as np
import matplotlib.pyplot as plt
import sys

    
filename = "bench_binary.csv"

titleGflops = "binary contraction einsum_ir"

threads_arr = []
gflops_arr = []
gflops_10_arr = []
gflops_90_arr = []

with open(filename, "r") as f:
    lines = f.readlines()
    lines = lines[1:]
    lines = [line.strip().split(",") for line in lines]
    for i in range(20,146,4):
        intermediateArray = []
        for j in range (50):
            threads = int(lines[j][0])
            gflops = float(lines[j][1])
            intermediateArray.append([threads, gflops])
        
        sortedArrayByGflops = sorted(intermediateArray, key=lambda x: x[1])
        
        threads = sortedArrayByGflops[0][0]
        gflops = sortedArrayByGflops[25][1]
        gflops_10 = sortedArrayByGflops[5][1]
        gflops_90 = sortedArrayByGflops[45][1]
        threads_arr.append(threads)
        gflops_arr.append(gflops)
        gflops_10_arr.append(gflops_10)
        gflops_90_arr.append(gflops_90)
        lines = lines[50:]

plt.figure(figsize=(10, 5))
plt.plot(threads_arr, gflops_arr, 'r-+', label="Median einsum_ir")
plt.fill_between(threads_arr, gflops_10_arr, gflops_90_arr, color='r', alpha=0.5, label="einsum_ir 10th and 90th percentile")
plt.axvline(x=72, color='b', linestyle='--', label="72 threads")
plt.xlabel("number of threads")
plt.ylabel("GFLOPS")
plt.grid(axis="both")

plt.legend()
plt.title(titleGflops)
plt.savefig("gflops_threads.png", pad_inches=0.1, bbox_inches='tight')
