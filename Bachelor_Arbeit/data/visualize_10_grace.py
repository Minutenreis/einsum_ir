import numpy as np
import matplotlib.pyplot as plt
import sys

results = [[],[],[],[],[],[],[]]
# 0 = size_1, 1 = 144, 2 = 72_0, 3=72_1, 4=mpi_c, 5=mpi_k, 6 = mpi_m_n


dist_c = "grace_dist_c.csv"
dist_k = "grace_dist_k.csv"
dist_m_n = "grace_dist_m_n.csv"
local_144 = "grace_local_144.csv"
local_72_0 = "grace_local_72_0.csv"
local_72_1 = "grace_local_72_1.csv"
files = [local_144, local_72_0, local_72_1, dist_c, dist_k, dist_m_n]

column = 1
for filename in files:
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = lines[1:]
        lines = [line.strip().split(",") for line in lines]
        resultRange = [2,4,8,16,32,64, 72,128, 180, 256, 384]
        for i in resultRange:
            gflops = 0
            for j in range (10):
                size = int(lines[j][0])
                gflops += float(lines[j][1])
            gflops /= 10

            if column == 1:
                results[0].append(size)
            results[column].append(gflops)
            lines = lines[10:]
    column+= 1

size_1_arr = results[0]
gflops_144_arr = results[1]
gflops_72_0_arr = results[2]
gflops_72_1_arr = results[3]
gflops_mpi_c = results[4]
gflops_mpi_k = results[5]
gflops_mpi_m_n = results[6]

plt.figure(figsize=(10, 5))
plt.plot(size_1_arr, gflops_144_arr, '-o', label="einsum_ir 144 cores")
plt.plot(size_1_arr, gflops_72_0_arr, '-v', label="einsum_ir 72 cores socket 0")
plt.plot(size_1_arr, gflops_72_1_arr, '-^', label="einsum_ir 72 cores socket 1")
plt.plot(size_1_arr, gflops_mpi_c, '-s', label="einsum_ir 144 cores distributed c0")
plt.plot(size_1_arr, gflops_mpi_m_n, '-*', label="einsum_ir 144 cores distributed m0 and n0")
plt.plot(size_1_arr, gflops_mpi_k, '-P', label="einsum_ir 144 cores distributed k0")
plt.grid(axis="both")

plt.xlabel("$m_0 = n_0 = k_0$")
plt.ylabel("GFLOPS")
plt.legend()
plt.title("Average Performance on NVIDIA Grace CPU Superchip")
plt.savefig("gflops_grace.png", pad_inches=0.1, bbox_inches='tight')

plt.figure(figsize=(10, 5))
plt.plot(size_1_arr, [new/base for new, base in zip(gflops_144_arr, gflops_72_0_arr)], '-o', label="einsum_ir 192 cores")
plt.plot(size_1_arr, [new/base for new, base in zip(gflops_mpi_c, gflops_72_0_arr)], '-s', label="einsum_ir 192 cores distributed c0")
plt.plot(size_1_arr, [new/base for new, base in zip(gflops_72_1_arr, gflops_72_0_arr)], '-^', label="einsum_ir 72 cores socket 1")
plt.plot(size_1_arr, [new/base for new, base in zip(gflops_mpi_m_n, gflops_72_0_arr)], '-*', label="einsum_ir 192 cores distributed m0 and n0")
plt.plot(size_1_arr, [new/base for new, base in zip(gflops_mpi_k, gflops_72_0_arr)], '-P', label="einsum_ir 192 cores distributed k0")
plt.grid(axis="both")

plt.xlabel("$m_0 = n_0 = k_0$")
plt.ylabel("Speedup")
plt.legend()
plt.title("Average Speedup on AWS c8g.metal-48xl compared to einsum_ir 72 cores socket 0")
plt.savefig("speedup_grace.png", pad_inches=0.1, bbox_inches='tight')
