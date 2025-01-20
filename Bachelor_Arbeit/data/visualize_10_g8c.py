import numpy as np
import matplotlib.pyplot as plt
import sys

results = [[],[],[],[],[],[],[]]
# 0 = size_1, 1 = 192, 2 = 96_0, 3=96_1, 4=mpi_c, 5=mpi_k, 6 = mpi_m_n


dist_c = "g8c_dist_c.csv"
dist_k = "g8c_dist_k.csv"
dist_m_n = "g8c_dist_m_n.csv"
local_192 = "g8c_local_192.csv"
local_96_0 = "g8c_local_96_0.csv"
local_96_1 = "g8c_local_96_1.csv"
files = [local_192, local_96_0, local_96_1, dist_c, dist_k, dist_m_n]

column = 1
for filename in files:
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = lines[1:]
        lines = [line.strip().split(",") for line in lines]
        for i in {2, 4, 8, 16, 32, 64, 96, 128, 180, 256, 384}:
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
gflops_192_arr = results[1]
gflops_96_0_arr = results[2]
gflops_96_1_arr = results[3]
gflops_mpi_c = results[4]
gflops_mpi_k = results[5]
gflops_mpi_m_n = results[6]

plt.figure(figsize=(10, 5))
plt.plot(size_1_arr, gflops_192_arr, '-o', label="einsum_ir 192 cores")
plt.plot(size_1_arr, gflops_96_0_arr, '-v', label="einsum_ir 96 cores socket 0")
plt.plot(size_1_arr, gflops_96_1_arr, '-^', label="einsum_ir 96 cores socket 1")
plt.plot(size_1_arr, gflops_mpi_c, '-s', label="einsum_ir 192 cores distributed c0")
plt.plot(size_1_arr, gflops_mpi_m_n, '-*', label="einsum_ir 192 cores distributed m0 and n0")
plt.plot(size_1_arr, gflops_mpi_k, '-P', label="einsum_ir 192 cores distributed k0")
plt.grid(axis="both")

plt.xlabel("$m_0 = n_0 = k_0$")
plt.ylabel("GFLOPS")
plt.legend()
plt.title("Average Performance on AWS c8g.metal-48xl")
plt.savefig("gflops_g8c.png", pad_inches=0.1, bbox_inches='tight')

plt.figure(figsize=(10, 5))
plt.plot(size_1_arr, [new/base for new, base in zip(gflops_192_arr, gflops_96_0_arr)], '-o', label="einsum_ir 192 cores")
plt.plot(size_1_arr, [new/base for new, base in zip(gflops_mpi_c, gflops_96_0_arr)], '-s', label="einsum_ir 192 cores distributed c0")
plt.plot(size_1_arr, [new/base for new, base in zip(gflops_96_1_arr, gflops_96_0_arr)], '-^', label="einsum_ir 96 cores socket 1")
plt.plot(size_1_arr, [new/base for new, base in zip(gflops_mpi_m_n, gflops_96_0_arr)], '-*', label="einsum_ir 192 cores distributed m0 and n0")
plt.plot(size_1_arr, [new/base for new, base in zip(gflops_mpi_k, gflops_96_0_arr)], '-P', label="einsum_ir 192 cores distributed k0")
plt.grid(axis="both")

plt.xlabel("$m_0 = n_0 = k_0$")
plt.ylabel("Speedup")
plt.legend()
plt.title("Average Speedup on AWS c8g.metal-48xl compared to einsum_ir 96 cores socket 0")
plt.savefig("speedup_g8c.png", pad_inches=0.1, bbox_inches='tight')
