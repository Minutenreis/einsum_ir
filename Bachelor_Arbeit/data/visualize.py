import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print("Please provide a filename as argument.")
    sys.exit(1)
    
filename = sys.argv[1]

title = filename if len(sys.argv) < 3 else sys.argv[2]
titleGflops = title + " GFLOPS"
titleSpeedup = title + " Speedup"

size_1_arr = []
gflops_arr = []
gflops_10_arr = []
gflops_90_arr = []
gflops_mpi_arr = []
gflops_mpi_10_arr = []
gflops_mpi_90_arr = []
speedup_arr = []
speedup_10_arr = []
speedup_90_arr = []

with open(filename, "r") as f:
    lines = f.readlines()
    lines = lines[1:]
    lines = [line.strip().split(",") for line in lines]
    for i in range(2,130,2):
        intermediateArray = []
        for j in range (100):
            size = int(lines[j][0])
            gflops = float(lines[j][1])
            gflops_mpi = float(lines[j][2])
            speedup = float(lines[j][3])
            
            intermediateArray.append([size, gflops, gflops_mpi, speedup])
        
        sortedArrayByGflops = sorted(intermediateArray, key=lambda x: x[1])
        sortedArrayByGflopsMPI = sorted(intermediateArray, key=lambda x: x[2])
        sortedArrayBySpeedup = sorted(intermediateArray, key=lambda x: x[3])
        
        size_1 = sortedArrayByGflops[0][0]
        # gflops = sum([x[1] for x in sortedArrayByGflops]) / 100
        gflops = sortedArrayByGflops[50][1]
        gflops_10 = sortedArrayByGflops[10][1]
        gflops_90 = sortedArrayByGflops[90][1]
        # gflops_mpi = sum([x[2] for x in sortedArrayByGflopsMPI]) / 100
        gflops_mpi = sortedArrayByGflopsMPI[50][2]
        gflops_mpi_10 = sortedArrayByGflopsMPI[10][2]
        gflops_mpi_90 = sortedArrayByGflopsMPI[90][2]
        speedup = sortedArrayBySpeedup[50][3]
        speedup_10 = sortedArrayBySpeedup[10][3]
        speedup_90 = sortedArrayBySpeedup[90][3]
        size_1_arr.append(size_1)
        gflops_arr.append(gflops)
        gflops_10_arr.append(gflops_10)
        gflops_90_arr.append(gflops_90)
        gflops_mpi_arr.append(gflops_mpi)
        gflops_mpi_10_arr.append(gflops_mpi_10)
        gflops_mpi_90_arr.append(gflops_mpi_90)
        speedup_arr.append(speedup)
        speedup_10_arr.append(speedup_10)
        speedup_90_arr.append(speedup_90)
        lines = lines[100:]

plt.figure(figsize=(10, 5))
plt.plot(size_1_arr, gflops_arr, 'r-+', label="Median einsum_ir")
plt.fill_between(size_1_arr, gflops_10_arr, gflops_90_arr, color='r', alpha=0.5, label="einsum_ir 10th and 90th percentile")
plt.plot(size_1_arr, gflops_mpi_arr, 'b-+', label="Median einsum_ir_mpi")
plt.fill_between(size_1_arr, gflops_mpi_10_arr, gflops_mpi_90_arr, color='b', alpha=0.5, label="einsum_ir_mpi 10th and 90th percentile")
plt.xlabel("m0 = n0 = k0")
plt.ylabel("GFLOPS")
plt.legend()
plt.title(titleGflops)
plt.savefig("gflops.png")

plt.figure(figsize=(10, 5))
plt.plot(size_1_arr, speedup_arr, "b-+" ,label="Median Speedup")
plt.fill_between(size_1_arr, speedup_10_arr, speedup_90_arr, color='b', alpha=0.5, label="Speedup 10th and 90th percentile")
plt.axhline(y=1, color='red', linestyle='--', label="einsum_ir")
plt.axhline(y=2, color='black', linestyle='--', label="maximum Speedup")
plt.xlabel("m0 = n0 = k0")
plt.ylabel("Speedup")
plt.legend()
plt.title(titleSpeedup)
plt.savefig("speedup.png")
