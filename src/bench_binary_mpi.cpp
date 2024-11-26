#include <cstdlib>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../../../usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h"
#include "backend/BinaryContractionTpp.h"
#include "backend/EinsumNode.h"
#include <ATen/ATen.h>

void blocked_binary_contraction() {

  int num_runs = 10;

  // Initialize MPI
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_size != 2) {
    std::cerr << "error: world size must be 2" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::cout << "*******************************" << std::endl;
    std::cout << "*** blocked matmul testcase ***" << std::endl;
    std::cout << "*******************************" << std::endl;
  }

  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration<double> l_dur;
  int64_t l_n_flops = 0;
  double l_time_compile = 0;
  double l_time = 0;
  double l_gflops = 0;

  // ./gemm_kernel F32 F32 F32 F32 64 8 24 64 24 64 1 0 0 0 0 0 0 0 0 nopf nobr
  // 1 0 1000000 0
  //
  // C: 256
  // M: 512
  // N: 128
  // K: 768

  int64_t l_size_c0 = 4;
  int64_t l_size_c1 = 8;
  int64_t l_size_c2 = 8;

  int64_t l_size_m0 = 8;
  int64_t l_size_m1 = 64;

  int64_t l_size_n0 = 16;
  int64_t l_size_n1 = 8;

  int64_t l_size_k0 = 2;
  int64_t l_size_k1 = 16;
  int64_t l_size_k2 = 24;

  int64_t l_size_c = l_size_c0 * l_size_c1 * l_size_c2;
  int64_t l_size_m = l_size_m0 * l_size_m1;
  int64_t l_size_n = l_size_n0 * l_size_n1;
  int64_t l_size_k = l_size_k0 * l_size_k1 * l_size_k2;

  l_n_flops = l_size_c * l_size_m * l_size_n * l_size_k * 2;

  std::map<int64_t, int64_t> l_dim_sizes;
  l_dim_sizes.insert(std::pair<int64_t, int64_t>(0, l_size_c0)); // c0
  l_dim_sizes.insert(std::pair<int64_t, int64_t>(1, l_size_c1)); // c1
  l_dim_sizes.insert(std::pair<int64_t, int64_t>(2, l_size_c2)); // c2

  l_dim_sizes.insert(std::pair<int64_t, int64_t>(3, l_size_m0)); // m0
  l_dim_sizes.insert(std::pair<int64_t, int64_t>(4, l_size_m1)); // m1

  l_dim_sizes.insert(std::pair<int64_t, int64_t>(5, l_size_n0)); // n0
  l_dim_sizes.insert(std::pair<int64_t, int64_t>(6, l_size_n1)); // n1

  l_dim_sizes.insert(std::pair<int64_t, int64_t>(7, l_size_k0)); // k0
  l_dim_sizes.insert(std::pair<int64_t, int64_t>(8, l_size_k1)); // k1
  l_dim_sizes.insert(std::pair<int64_t, int64_t>(9, l_size_k2)); // k2

  //                               c0 c1 c2 m0 k0 k1 k2 m1
  int64_t l_dim_ids_in_left[8] = {0, 1, 2, 3, 7, 8, 9, 4};
  //                               c0 c1 c2 n0 k0 k1 n1 k2
  int64_t l_dim_ids_in_right[8] = {0, 1, 2, 5, 7, 8, 6, 9};
  //                               c0 c1 c2 n0 m0 n1 m1
  int64_t l_dim_ids_out[7] = {0, 1, 2, 5, 3, 6, 4};

  at::Tensor l_ten_left;
  at::Tensor l_ten_right;
  at::Tensor l_ten_out;

  if (rank == 0) {
    l_ten_left = at::rand({l_size_c, l_size_k, l_size_m});
    l_ten_right = at::rand({l_size_c, l_size_n, l_size_k});
    l_ten_out = at::zeros({l_size_c, l_size_n, l_size_m});

    /**
     * einsum data
     **/
    l_ten_left = l_ten_left.view({l_size_c0,   // 0
                                  l_size_c1,   // 1
                                  l_size_c2,   // 2
                                  l_size_k0,   // 3
                                  l_size_k1,   // 4
                                  l_size_k2,   // 5
                                  l_size_m0,   // 6
                                  l_size_m1}); // 7
    //                                c0 c1 c2 m0 k0 k1 k2 m1
    l_ten_left = l_ten_left.permute({0, 1, 2, 6, 3, 4, 5, 7}).contiguous();

    l_ten_right = l_ten_right.view({l_size_c0,   // 0
                                    l_size_c1,   // 1
                                    l_size_c2,   // 2
                                    l_size_n0,   // 3
                                    l_size_n1,   // 4
                                    l_size_k0,   // 5
                                    l_size_k1,   // 6
                                    l_size_k2}); // 7
    //                                  c0 c1 c2 n0 k0 k1 n1 k2
    l_ten_right = l_ten_right.permute({0, 1, 2, 3, 5, 6, 4, 7}).contiguous();

    l_ten_out = l_ten_out.view({l_size_c0,   // 0
                                l_size_c1,   // 1
                                l_size_c2,   // 2
                                l_size_m0,   // 3
                                l_size_m1,   // 4
                                l_size_n0,   // 5
                                l_size_n1}); // 6
    //                              c0 c1 c2 n0 m0 n1 m1
    l_ten_out = l_ten_out.permute({0, 1, 2, 5, 3, 6, 4}).contiguous();

    /*
     * einsum_ir
     */
    std::cout << "einsum_ir:" << std::endl;

    einsum_ir::backend::BinaryContractionTpp l_bin_cont;
    l_bin_cont.init(8, 8, 7, &l_dim_sizes, &l_dim_sizes, &l_dim_sizes, nullptr,
                    &l_dim_sizes, l_dim_ids_in_left, l_dim_ids_in_right,
                    l_dim_ids_out, einsum_ir::FP32, einsum_ir::FP32,
                    einsum_ir::FP32, einsum_ir::FP32, einsum_ir::ZERO,
                    einsum_ir::MADD, einsum_ir::UNDEFINED_KTYPE);

    l_tp0 = std::chrono::steady_clock::now();
    l_bin_cont.compile();
    l_tp1 = std::chrono::steady_clock::now();
    l_dur = std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1 -
                                                                      l_tp0);
    l_time_compile = l_dur.count();

// enable threading
#ifdef _OPENMP
    // four times overload
    int64_t l_num_tasks = omp_get_max_threads() * 4;

    l_bin_cont.threading(l_num_tasks);
#endif

    // dry run
    l_bin_cont.contract(l_ten_left.data_ptr(), l_ten_right.data_ptr(),
                        l_ten_out.data_ptr());

    l_tp0 = std::chrono::steady_clock::now();
    // measure multiple times
    for (int i = 0; i < num_runs; i++) {
      l_bin_cont.contract(l_ten_left.data_ptr(), l_ten_right.data_ptr(),
                          l_ten_out.data_ptr());
    }
    l_tp1 = std::chrono::steady_clock::now();
    l_dur = std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1 - l_tp0) / 10;
    l_time = l_dur.count();
    l_gflops = 1.0E-9 * l_n_flops / l_time;

    std::cout << "  time (compile): " << l_time_compile << std::endl;
    std::cout << "  time (contract): " << l_time << std::endl;
    std::cout << "  gflops: " << l_gflops << std::endl;
  }

  auto l_gflops_old = l_gflops;

  {
    /*
     * einsum_ir mpi c0 split
     */
    if (rank == 0) {
      std::cout << "einsum_ir_mpi split:" << std::endl;
    }

    int chunks = 2;

    std::map<int64_t, int64_t> l_dim_sizes_mpi = l_dim_sizes;
    l_dim_sizes_mpi[0] = l_size_c0 / chunks;

    einsum_ir::backend::BinaryContractionTpp l_bin_cont_mpi;
    l_bin_cont_mpi.init(
        8, 8, 7, &l_dim_sizes_mpi, &l_dim_sizes_mpi, &l_dim_sizes_mpi, nullptr,
        &l_dim_sizes_mpi, l_dim_ids_in_left, l_dim_ids_in_right, l_dim_ids_out,
        einsum_ir::FP32, einsum_ir::FP32, einsum_ir::FP32, einsum_ir::FP32,
        einsum_ir::ZERO, einsum_ir::MADD, einsum_ir::UNDEFINED_KTYPE);

    l_tp0 = std::chrono::steady_clock::now();
    l_bin_cont_mpi.compile();
    l_tp1 = std::chrono::steady_clock::now();
    l_dur = std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1 -
                                                                      l_tp0);
    l_time_compile = l_dur.count();

    // enable threading
#ifdef _OPENMP
    // four times overload
    int64_t l_num_tasks = omp_get_max_threads() * 4;

    l_bin_cont_mpi.threading(l_num_tasks);
#endif

    auto l_dur_communication = std::chrono::duration<double>::zero();
    auto l_dur_contract = std::chrono::duration<double>::zero();

    // wait on all MPI processes
    for (int i = 0; i < 11; i++) {
      if (rank == 0) {
        l_tp0 = std::chrono::steady_clock::now();

        // split tensor
        auto l_ten_left_mpi_split = l_ten_left.chunk(chunks, 0);
        auto l_ten_right_mpi_split = l_ten_right.chunk(chunks, 0);
        auto l_ten_out_mpi = at::zeros_like(l_ten_out);
        auto l_ten_out_mpi_split = l_ten_out_mpi.chunk(chunks, 0);

        // broadcast? async?
        // send tensor to other rank
        MPI_Request l_reqs[2];
        MPI_Request l_req_recv;

        // todo: do Waitall in dedicated communication thread(s)
        // todo: send smaller chunks to overlap communication and computation

        MPI_Isend(l_ten_left_mpi_split[1].data_ptr(),
                  l_ten_left_mpi_split[1].numel(), MPI_FLOAT, 1, 0,
                  MPI_COMM_WORLD, &l_reqs[0]);
        MPI_Isend(l_ten_right_mpi_split[1].data_ptr(),
                  l_ten_right_mpi_split[1].numel(), MPI_FLOAT, 1, 0,
                  MPI_COMM_WORLD, &l_reqs[1]);
        MPI_Irecv(l_ten_out_mpi_split[1].data_ptr(),
                  l_ten_out_mpi_split[1].numel(), MPI_FLOAT, 1, 0,
                  MPI_COMM_WORLD, &l_req_recv);
        MPI_Waitall(2, l_reqs, MPI_STATUSES_IGNORE);

        auto l_tpComm1End = std::chrono::steady_clock::now();

        // perform local computation
        l_bin_cont_mpi.contract(l_ten_left_mpi_split[0].data_ptr(),
                                l_ten_right_mpi_split[0].data_ptr(),
                                l_ten_out_mpi_split[0].data_ptr());

        auto l_tpContractEnd = std::chrono::steady_clock::now();

        MPI_Wait(&l_req_recv, MPI_STATUS_IGNORE);

        if (i > 0) {
          l_tp1 = std::chrono::steady_clock::now();
          l_dur_communication += std::chrono::duration_cast<std::chrono::duration<double>>(
              l_tpComm1End - l_tp0 + l_tp1 - l_tpContractEnd);
          l_dur_contract += std::chrono::duration_cast<std::chrono::duration<double>>(
              l_tpContractEnd - l_tpComm1End);

          if (i == 10) {
            l_dur = l_dur_communication + l_dur_contract;
            l_time = l_dur.count() / 10;
            l_gflops = 1.0E-9 * l_n_flops / l_time;

            std::cout << "  time (compile): " << l_time_compile << std::endl;
            std::cout << "  time (contract total): " << l_time << std::endl;
            std::cout << "  time (contract only):" << l_dur_contract.count() / 10 << std::endl;
            std::cout << "  time (communication): " << l_dur_communication.count() / 10 << std::endl;
            std::cout << "  gflops: " << l_gflops << std::endl;
            std::cout << "  Speedup: " << l_gflops / l_gflops_old << std::endl;

            if (!at::allclose(l_ten_out_mpi, l_ten_out)) {
              std::cerr
                  << "error: einsum_ir_mpi solution is not close to einsum_ir!"
                  << std::endl
                  << "max error: " << (l_ten_out_mpi - l_ten_out).abs().max().item<float>() << std::endl;
            }
          }
        }
      } else {
        auto l_ten_left_mpi = at::zeros({
            l_size_c0 / chunks, // 0
            l_size_c1,          // 1
            l_size_c2,          // 2
            l_size_m0,          // 6
            l_size_k0,          // 3
            l_size_k1,          // 4
            l_size_k2,          // 5
            l_size_m1           // 7
        });
        auto l_ten_right_mpi = at::zeros({
            l_size_c0 / chunks, // 0
            l_size_c1,          // 1
            l_size_c2,          // 2
            l_size_n0,          // 3
            l_size_k0,          // 5
            l_size_k1,          // 6
            l_size_n1,          // 4
            l_size_k2           // 7
        });
        auto l_ten_out_mpi = at::zeros({
            l_size_c0 / chunks, // 0
            l_size_c1,          // 1
            l_size_c2,          // 2
            l_size_n0,          // 5
            l_size_m0,          // 3
            l_size_n1,          // 6
            l_size_m1           // 4
        });

        MPI_Request l_reqs[2];
        MPI_Irecv(l_ten_left_mpi.data_ptr(), l_ten_left_mpi.numel(), MPI_FLOAT,
                  0, 0, MPI_COMM_WORLD, &l_reqs[0]);
        MPI_Irecv(l_ten_right_mpi.data_ptr(), l_ten_right_mpi.numel(), MPI_FLOAT,
                  0, 0, MPI_COMM_WORLD, &l_reqs[1]);
        MPI_Waitall(2, l_reqs, MPI_STATUSES_IGNORE);

        l_bin_cont_mpi.contract(l_ten_left_mpi.data_ptr(),
                                l_ten_right_mpi.data_ptr(),
                                l_ten_out_mpi.data_ptr());

        MPI_Send(l_ten_out_mpi.data_ptr(), l_ten_out_mpi.numel(), MPI_FLOAT, 0,
                 0, MPI_COMM_WORLD);
      }
    }
  }
}

int main() {
  int provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::cout << "running bench_binary!" << std::endl;
  }

  blocked_binary_contraction();

  if (rank == 0) {
    std::cout << "finished running bench_binary!" << std::endl;
  }
  MPI_Finalize();
  return EXIT_SUCCESS;
}