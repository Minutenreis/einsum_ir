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
gflops_mpi_arr = []
speedup_arr = []

with open(filename, "r") as f:
    lines = f.readlines()
    lines = lines[1:]
    lines = [line.strip().split(",") for line in lines]
    for i in {2,4,8,16,32,64, 96,128, 180, 256}:
        intermediateArray = []
        for j in range (10):
            size = int(lines[j][0])
            gflops = float(lines[j][1])
            gflops_mpi = float(lines[j][2])
            speedup = float(lines[j][3])
            
            intermediateArray.append([size, gflops, gflops_mpi, speedup])
        

        sortedArrayByGflops = sorted(intermediateArray, key=lambda x: x[1])
        sortedArrayByGflopsMPI = sorted(intermediateArray, key=lambda x: x[2])
        sortedArrayBySpeedup = sorted(intermediateArray, key=lambda x: x[3])
        
        size_1 = sortedArrayByGflops[0][0]
        gflops = sum([x[1] for x in sortedArrayByGflops]) / 10
        gflops_mpi = sum([x[2] for x in sortedArrayByGflopsMPI]) / 10
        speedup = gflops_mpi / gflops
        size_1_arr.append(size_1)
        gflops_arr.append(gflops)
        gflops_mpi_arr.append(gflops_mpi)
        speedup_arr.append(speedup)
        lines = lines[10:]

plt.figure(figsize=(10, 5))
plt.plot(size_1_arr, gflops_arr, 'r-+', label="Median einsum_ir")
plt.plot(size_1_arr, gflops_mpi_arr, 'b-+', label="Median einsum_ir_mpi")
plt.xlabel("m0 = n0 = k0")
plt.ylabel("GFLOPS")
plt.legend()
plt.title(titleGflops)
plt.savefig("gflops.png")

plt.figure(figsize=(10, 5))
plt.plot(size_1_arr, speedup_arr, "b-+" ,label="Median Speedup")
plt.axhline(y=1, color='red', linestyle='--', label="einsum_ir")
plt.axhline(y=2, color='black', linestyle='--', label="maximum Speedup")
plt.xlabel("m0 = n0 = k0")
plt.ylabel("Speedup")
plt.legend()
plt.title(titleSpeedup)
plt.savefig("speedup.png")
