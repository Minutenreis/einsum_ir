#include <cstdlib>
#include <iostream>

#include <omp.h>

#include "backend/BinaryContractionTpp.h"
#include "backend/EinsumNode.h"
#include <ATen/ATen.h>
#include <mpi.h>

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

  const int64_t l_size_c0 = 16;
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

  int64_t l_size_total = l_size_c * l_size_m * l_size_n * l_size_k;

  int64_t l_n_flops = l_size_total * 2;

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

    // four times overload
    int64_t l_num_tasks = omp_get_max_threads() * 4;

    l_bin_cont.threading(l_num_tasks);

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

    // has to be a factor of l_size_c0
    const int chunks = 16;
    const int chunks_per_rank = chunks / 2;
    static_assert(chunks % 2 == 0, "chunks has to be a factor of l_size_c0");
    static_assert(l_size_c0 % chunks == 0, "l_size_c0 has to be a factor of chunks");

    std::map<int64_t, int64_t> l_dim_sizes_mpi = l_dim_sizes;

    if (rank == 0) {
      l_dim_sizes_mpi[0] = l_size_c0 / 2;
    } else {
      l_dim_sizes_mpi[0] = l_size_c0 / chunks;
    }

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

    // four times overload
    int64_t l_num_tasks = omp_get_max_threads() * 4;

    l_bin_cont_mpi.threading(l_num_tasks);
    omp_set_nested(true); // allow multiple nested parallel regions

    auto l_dur_communication = std::chrono::duration<double>::zero();
    auto l_dur_contract = std::chrono::duration<double>::zero();
    auto l_dur_dataPrep = std::chrono::duration<double>::zero();
    l_dur = std::chrono::duration<double>::zero();

    // synchronize MPI processes for better time measurement
    MPI_Barrier(MPI_COMM_WORLD);

    // 1st run is warmup, next 10 for getting average times
    for (int i = 0; i < 11; i++) {
      if (rank == 0) {
        l_tp0 = std::chrono::steady_clock::now();

        // split tensor
        auto l_ten_left_mpi_split = l_ten_left.chunk(chunks, 0);
        auto l_ten_right_mpi_split = l_ten_right.chunk(chunks, 0);
        auto l_ten_out_mpi = at::zeros_like(l_ten_out);
        auto l_ten_out_mpi_split = l_ten_out_mpi.chunk(chunks, 0);

        auto l_tp_chunkedSend = std::chrono::steady_clock::now();
        l_dur_dataPrep += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp_chunkedSend - l_tp0);

#pragma omp parallel num_threads(2)
        {
          if (omp_get_thread_num() == 0) {
            auto l_tp0_contract = std::chrono::steady_clock::now();
            l_bin_cont_mpi.contract(l_ten_left_mpi_split[0].data_ptr(),
                                    l_ten_right_mpi_split[0].data_ptr(),
                                    l_ten_out_mpi_split[0].data_ptr());
            auto l_tp1_contract = std::chrono::steady_clock::now();
            if (i > 0)
              l_dur_contract += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1_contract - l_tp0_contract);
          } else {
            auto l_tp0_communication = std::chrono::steady_clock::now();

            MPI_Request l_reqs[3];
            MPI_Isend(l_ten_left_mpi_split[chunks_per_rank].data_ptr(),
                      l_ten_left_mpi_split[chunks_per_rank].numel(), MPI_FLOAT, 1, 0,
                      MPI_COMM_WORLD, &l_reqs[0]);
            MPI_Isend(l_ten_right_mpi_split[chunks_per_rank].data_ptr(),
                      l_ten_right_mpi_split[chunks_per_rank].numel(), MPI_FLOAT, 1, 1,
                      MPI_COMM_WORLD, &l_reqs[1]);
            MPI_Waitall(2, l_reqs, MPI_STATUSES_IGNORE);

            for (int j = chunks_per_rank + 1; j < chunks; j++) {
              MPI_Irecv(l_ten_out_mpi_split[j - 1].data_ptr(),
                        l_ten_out_mpi_split[j - 1].numel(), MPI_FLOAT, 1, 0,
                        MPI_COMM_WORLD, &l_reqs[2]);

              MPI_Isend(l_ten_left_mpi_split[j].data_ptr(),
                        l_ten_left_mpi_split[j].numel(), MPI_FLOAT, 1, 0,
                        MPI_COMM_WORLD, &l_reqs[0]);
              MPI_Isend(l_ten_right_mpi_split[j].data_ptr(),
                        l_ten_right_mpi_split[j].numel(), MPI_FLOAT, 1, 1,
                        MPI_COMM_WORLD, &l_reqs[1]);
              MPI_Waitall(3, l_reqs, MPI_STATUSES_IGNORE);
            }

            MPI_Recv(l_ten_out_mpi_split[chunks - 1].data_ptr(),
                     l_ten_out_mpi_split[chunks - 1].numel(), MPI_FLOAT, 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            auto l_tp1_communication = std::chrono::steady_clock::now();
            if (i > 0)
              l_dur_communication += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1_communication - l_tp0_communication);
          }
        }

        if (i > 0) {
          l_tp1 = std::chrono::steady_clock::now();
          l_dur += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1 - l_tp0);

          if (i == 10) {
            l_time = l_dur.count() / 10;
            l_gflops = 1.0E-9 * l_n_flops / l_time;

            double l_times_r1[4] = {0.0, 0.0, 0.0, 0.0};
            MPI_Recv(l_times_r1, 4, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

            std::cout << "  num_chunks: " << chunks << std::endl;
            std::cout << "  time (compile): " << l_time_compile << std::endl;
            std::cout << "  rank 0" << std::endl;
            std::cout << "    time (data prep): " << l_dur_dataPrep.count() / 10 << std::endl;
            std::cout << "    time (contract):  " << l_dur_contract.count() / 10 << std::endl;
            std::cout << "    time (comm):      " << l_dur_communication.count() / 10 << std::endl;
            std::cout << "    time (total):     " << l_time << std::endl;
            std::cout << "    time (control):   " << (l_dur_dataPrep.count() + std::max(l_dur_contract.count(), l_dur_communication.count())) / 10 << std::endl;
            std::cout << "  rank 1" << std::endl;
            std::cout << "    time (data prep): " << l_times_r1[2] << std::endl;
            std::cout << "    time (contract):  " << l_times_r1[0] << std::endl;
            std::cout << "    time (comm):      " << l_times_r1[1] << std::endl;
            std::cout << "    time (total):     " << l_times_r1[3] << std::endl;
            std::cout << "    time (control):   " << l_times_r1[2] + std::max(l_times_r1[0], l_times_r1[1]) << std::endl;
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
        l_tp0 = std::chrono::steady_clock::now();

        std::array<at::Tensor, 2> l_ten_left_mpi = {at::zeros({l_size_c / chunks,
                                                               l_size_k,
                                                               l_size_m}),
                                                    at::zeros({l_size_c / chunks,
                                                               l_size_k,
                                                               l_size_m})};
        std::array<at::Tensor, 2> l_ten_right_mpi = {at::zeros({l_size_c / chunks,
                                                                l_size_n,
                                                                l_size_k}),
                                                     at::zeros({l_size_c / chunks,
                                                                l_size_n,
                                                                l_size_k})};
        std::array<at::Tensor, 2> l_ten_out_mpi = {at::zeros({l_size_c / chunks,
                                                              l_size_n,
                                                              l_size_m}),
                                                   at::zeros({l_size_c / chunks,
                                                              l_size_n,
                                                              l_size_m})};

        auto l_tp0_comm = std::chrono::steady_clock::now();
        if (i > 0)
          l_dur_dataPrep += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp0_comm - l_tp0);

        MPI_Request l_reqs[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        MPI_Irecv(l_ten_left_mpi[0].data_ptr(),
                  l_ten_left_mpi[0].numel(), MPI_FLOAT, 0, 0,
                  MPI_COMM_WORLD, &l_reqs[0]);
        MPI_Irecv(l_ten_right_mpi[0].data_ptr(),
                  l_ten_right_mpi[0].numel(), MPI_FLOAT, 0, 1,
                  MPI_COMM_WORLD, &l_reqs[1]);
        MPI_Waitall(2, l_reqs, MPI_STATUSES_IGNORE);

        auto l_tp1_comm = std::chrono::steady_clock::now();
        if (i > 0)
          l_dur_communication += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1_comm - l_tp0_comm);

#pragma omp parallel num_threads(2)
        {
          for (int j = 1; j < chunks_per_rank; j++) {

            if (omp_get_thread_num() == 0) {
              auto l_tp0_contract = std::chrono::steady_clock::now();

              l_bin_cont_mpi.contract(l_ten_left_mpi[(j - 1) % 2].data_ptr(),
                                      l_ten_right_mpi[(j - 1) % 2].data_ptr(),
                                      l_ten_out_mpi[(j - 1) % 2].data_ptr());

              auto l_tp1_contract = std::chrono::steady_clock::now();
              if (i > 0)
                l_dur_contract += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1_contract - l_tp0_contract);
            } else {
              l_tp0_comm = std::chrono::steady_clock::now();

              MPI_Irecv(l_ten_left_mpi[j % 2].data_ptr(),
                        l_ten_left_mpi[j % 2].numel(), MPI_FLOAT, 0, 0,
                        MPI_COMM_WORLD, &l_reqs[0]);
              MPI_Irecv(l_ten_right_mpi[j % 2].data_ptr(),
                        l_ten_right_mpi[j % 2].numel(), MPI_FLOAT, 0, 1,
                        MPI_COMM_WORLD, &l_reqs[1]);
              MPI_Waitall(3, l_reqs, MPI_STATUSES_IGNORE);

              l_tp1_comm = std::chrono::steady_clock::now();
              if (i > 0)
                l_dur_communication += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1_comm - l_tp0_comm);
            }

#pragma omp barrier

            if (omp_get_thread_num() != 0) {
              l_tp0_comm = std::chrono::steady_clock::now();

              MPI_Isend(l_ten_out_mpi[(j - 1) % 2].data_ptr(),
                        l_ten_out_mpi[(j - 1) % 2].numel(), MPI_FLOAT, 0, 0,
                        MPI_COMM_WORLD, &l_reqs[2]);

              l_tp1_comm = std::chrono::steady_clock::now();
              if (i > 0)
                l_dur_communication += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1_comm - l_tp0_comm);
            }
          }
        }

        auto l_tp0_contract = std::chrono::steady_clock::now();

        l_bin_cont_mpi.contract(l_ten_left_mpi[(chunks_per_rank - 1) % 2].data_ptr(),
                                l_ten_right_mpi[(chunks_per_rank - 1) % 2].data_ptr(),
                                l_ten_out_mpi[(chunks_per_rank - 1) % 2].data_ptr());

        auto l_tp1_contract = std::chrono::steady_clock::now();
        if (i > 0)
          l_dur_contract += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1_contract - l_tp0_contract);

        MPI_Isend(l_ten_out_mpi[(chunks_per_rank - 1) % 2].data_ptr(),
                  l_ten_out_mpi[(chunks_per_rank - 1) % 2].numel(), MPI_FLOAT, 0, 0,
                  MPI_COMM_WORLD, &l_reqs[1]);

        MPI_Waitall(2, &l_reqs[1], MPI_STATUSES_IGNORE);

        l_tp1 = std::chrono::steady_clock::now();
        if (i > 0) {
          l_dur_communication += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1 - l_tp1_contract);
          l_dur += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1 - l_tp0);
          if (i == 10) {
            double l_times_r1[4] = {l_dur_contract.count() / 10, l_dur_communication.count() / 10, l_dur_dataPrep.count() / 10, l_dur.count() / 10};
            MPI_Send(l_times_r1, 4, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
          }
        }
      }
    }
  }
}

int main() {
  int provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
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