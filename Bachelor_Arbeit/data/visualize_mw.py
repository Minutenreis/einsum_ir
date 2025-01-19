import numpy as np
import matplotlib.pyplot as plt
import sys

# 0 = size_1, 1 = 144, 2 = 72_0, 3=72_1, 4=mpi_c, 5=mpi_k, 6 = mpi_m_n
size_1_arr = []
master_worker_arr = []
mw_comp_arr = []


dist_master_worker = "grace_master_worker.csv"
dist_mw_comp_arr = "grace_mw_comp_c0.csv"

with open(dist_master_worker, "r") as f:
    lines = f.readlines()
    lines = lines[1:]
    lines = [line.strip().split(",") for line in lines]
    resultRange = [2,4,8,16,32,64, 72,128, 180, 256, 384]
    for i in resultRange:
        size = int(lines[0][0])
        gflops = float(lines[0][1])
        size_1_arr.append(size)
        master_worker_arr.append(gflops)
        lines = lines[1:]

with open(dist_mw_comp_arr, "r") as f:
    lines = f.readlines()
    lines = lines[1:]
    lines = [line.strip().split(",") for line in lines]
    resultRange = [2,4,8,16,32,64,72,128, 180, 256, 384]
    for i in resultRange:
        gflops = 0
        for j in range (10):
            size = int(lines[j][0])
            gflops += float(lines[j][1])
        gflops /= 10
        mw_comp_arr.append(gflops)
        lines = lines[10:]


plt.figure(figsize=(10, 5))
plt.plot(size_1_arr, master_worker_arr, '-o', label="einsum_ir master-worker")
plt.plot(size_1_arr, mw_comp_arr, '-v', label="einsum_ir distributed c")
plt.grid(axis="both")

plt.xlabel("$m_0 = n_0 = k_0$")
plt.ylabel("GFLOPS")
plt.legend()
plt.title("Average Performance on NVIDIA Grace CPU Superchip")
plt.savefig("gflops_grace_master_worker.png", pad_inches=0.1, bbox_inches='tight')
