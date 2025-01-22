#include <cstdlib>
#include <iostream>

#include <omp.h>

#include "backend/BinaryContractionTpp.h"
#include "backend/EinsumNode.h"
#include <ATen/ATen.h>
#include <mpi.h>

void blocked_binary_contraction(int64_t size_1, int64_t size_2) {

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

  const int64_t l_size_c0 = 2;

  const int64_t l_size_m0 = size_1;
  const int64_t l_size_m1 = size_2;

  const int64_t l_size_n0 = size_1;
  const int64_t l_size_n1 = size_2;

  const int64_t l_size_k0 = size_1;
  const int64_t l_size_k1 = size_2;

  const int64_t l_size_c = l_size_c0;
  const int64_t l_size_m = l_size_m0 * l_size_m1;
  const int64_t l_size_n = l_size_n0 * l_size_n1;
  const int64_t l_size_k = l_size_k0 * l_size_k1;

  const int64_t l_size_total = l_size_c * l_size_m * l_size_n * l_size_k;
  const int64_t l_size_left = l_size_c * l_size_m * l_size_k;
  const int64_t l_size_right = l_size_c * l_size_n * l_size_k;
  const int64_t l_size_out = l_size_c * l_size_n * l_size_m;

  const int64_t l_n_flops = l_size_total * 2;

  int c0 = 0;
  int m0 = 1;
  int m1 = 2;
  int n0 = 3;
  int n1 = 4;
  int k0 = 5;
  int k1 = 6;

  std::map<int64_t, int64_t> l_dim_sizes;
  l_dim_sizes.insert(std::pair<int64_t, int64_t>(c0, l_size_c0));

  l_dim_sizes.insert(std::pair<int64_t, int64_t>(m0, l_size_m0));
  l_dim_sizes.insert(std::pair<int64_t, int64_t>(m1, l_size_m1));

  l_dim_sizes.insert(std::pair<int64_t, int64_t>(n0, l_size_n0));
  l_dim_sizes.insert(std::pair<int64_t, int64_t>(n1, l_size_n1));

  l_dim_sizes.insert(std::pair<int64_t, int64_t>(k0, l_size_k0));
  l_dim_sizes.insert(std::pair<int64_t, int64_t>(k1, l_size_k1));

  std::vector<int64_t> l_dim_ids_in_left({m0, c0, k0, k1, m1});
  std::vector<int64_t> l_dim_ids_in_right({n0, c0, k0, n1, k1});
  std::vector<int64_t> l_dim_ids_out({m0, n0, c0, n1, m1});

  at::Tensor l_ten_left;
  at::Tensor l_ten_right;
  at::Tensor l_ten_out;
  at::Tensor l_ten_out2;

  if (rank == 0) {
    l_ten_left = at::rand({l_size_c, l_size_k, l_size_m});
    l_ten_right = at::rand({l_size_c, l_size_n, l_size_k});
    l_ten_out = at::zeros({l_size_c, l_size_n, l_size_m});

    l_ten_left = l_ten_left.view({
                                     l_size_c0, // 0
                                     l_size_m0, // 1
                                     l_size_m1, // 2
                                     l_size_k0, // 3
                                     l_size_k1, // 4
                                 })
                     //        c0 m0 k0 k1 m1
                     .permute({0, 1, 3, 4, 2})
                     .contiguous();

    l_ten_right = l_ten_right.view({
                                       l_size_c0, // 0
                                       l_size_n0, // 1
                                       l_size_n1, // 2
                                       l_size_k0, // 3
                                       l_size_k1, // 4
                                   })
                      //        c0 n0 k0 n1 k1
                      .permute({0, 1, 3, 2, 4})
                      .contiguous();

    l_ten_out = l_ten_out.view({
                                   l_size_c0, // 0
                                   l_size_n0, // 1
                                   l_size_n1, // 2
                                   l_size_m0, // 3
                                   l_size_m1, // 4
                               })
                    //        c0 m0 n0 n1 m1
                    .permute({0, 3, 1, 4, 2})
                    .contiguous();
  }

  // has to be a factor of l_size_c0
  const int chunks = l_size_c0;
  const int comm_threads = 1;

  const int chunks_r0 = ceil(chunks / 2.0);
  const int chunks_r1 = chunks - chunks_r0;
  static_assert(l_size_c0 % chunks == 0, "chunks has to be a factor of l_size_c0");

  int64_t l_size_l_chunk = l_size_left / chunks;
  int64_t l_size_r_chunk = l_size_right / chunks;
  int64_t l_size_o_chunk = l_size_out / chunks;

  int64_t l_size_l_chunk_comm = l_size_l_chunk / comm_threads;
  int64_t l_size_r_chunk_comm = l_size_r_chunk / comm_threads;
  int64_t l_size_o_chunk_comm = l_size_o_chunk / comm_threads;

  std::map<int64_t, int64_t> l_dim_sizes_mpi = l_dim_sizes;

  if (rank == 0) {
    l_dim_sizes_mpi[0] = l_size_c0 / chunks * chunks_r0;
  } else {
    l_dim_sizes_mpi[0] = l_size_c0 / chunks;
  }

  einsum_ir::backend::BinaryContractionTpp l_bin_cont_mpi;
  l_bin_cont_mpi.init(
      5, 5, 5,
      &l_dim_sizes_mpi, &l_dim_sizes_mpi, &l_dim_sizes_mpi, nullptr, &l_dim_sizes_mpi,
      l_dim_ids_in_left.data(), l_dim_ids_in_right.data(), l_dim_ids_out.data(),
      einsum_ir::FP32, einsum_ir::FP32, einsum_ir::FP32, einsum_ir::FP32,
      einsum_ir::ZERO, einsum_ir::MADD, einsum_ir::UNDEFINED_KTYPE);

  l_tp0 = std::chrono::steady_clock::now();
  l_bin_cont_mpi.compile();

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
      auto l_ten_out_mpi = at::zeros_like(l_ten_out);
      l_tp0 = std::chrono::steady_clock::now();

      // split tensor
      auto l_ten_left_mpi_split = l_ten_left.chunk(chunks, 0);
      auto l_ten_right_mpi_split = l_ten_right.chunk(chunks, 0);
      auto l_ten_out_mpi_split = l_ten_out_mpi.chunk(chunks, 0);

      std::chrono::duration<double> l_dur_temp_comm = std::chrono::duration<double>::zero();
      std::chrono::duration<double> l_dur_temp_contract = std::chrono::duration<double>::zero();
      auto l_tp_chunkedSend = std::chrono::steady_clock::now();
      if (i > 0)
        l_dur_dataPrep += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp_chunkedSend - l_tp0);

#pragma omp parallel num_threads(1 + comm_threads)
#pragma omp reduce(max : l_dur_temp)
      {
        if (omp_get_thread_num() == 0) {
          auto l_tp0_contract = std::chrono::steady_clock::now();

          // on rank_0 the contraction doesn't have to be chunked
          l_bin_cont_mpi.contract(l_ten_left_mpi_split[0].data_ptr(),
                                  l_ten_right_mpi_split[0].data_ptr(),
                                  l_ten_out_mpi_split[0].data_ptr());

          auto l_tp1_contract = std::chrono::steady_clock::now();
          if (i > 0)
            l_dur_contract += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1_contract - l_tp0_contract);
        } else {
          auto l_tp0_comm = std::chrono::steady_clock::now();

          auto thread = omp_get_thread_num() - 1; // -1 to get thread id starting from 0
          auto tag0 = thread;
          auto tag1 = thread + comm_threads;
          int64_t offset_l = thread * l_size_l_chunk_comm;
          int64_t offset_r = thread * l_size_r_chunk_comm;
          int64_t offset_o = thread * l_size_o_chunk_comm;

          MPI_Request l_reqs[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
          // send initial chunk
          MPI_Isend(reinterpret_cast<float *>(l_ten_left_mpi_split[chunks_r0].data_ptr()) + offset_l,
                    l_size_l_chunk_comm, MPI_FLOAT, 1, tag0,
                    MPI_COMM_WORLD, &l_reqs[0]);
          MPI_Isend(reinterpret_cast<float *>(l_ten_right_mpi_split[chunks_r0].data_ptr()) + offset_r,
                    l_size_r_chunk_comm, MPI_FLOAT, 1, tag1,
                    MPI_COMM_WORLD, &l_reqs[1]);
          MPI_Waitall(2, l_reqs, MPI_STATUSES_IGNORE);

          for (int j = chunks_r0 + 1; j < chunks; j++) {

            // send chunk for contraction
            MPI_Isend(reinterpret_cast<float *>(l_ten_left_mpi_split[j].data_ptr()) + offset_l,
                      l_size_l_chunk_comm, MPI_FLOAT, 1, tag0,
                      MPI_COMM_WORLD, &l_reqs[0]);
            MPI_Isend(reinterpret_cast<float *>(l_ten_right_mpi_split[j].data_ptr()) + offset_r,
                      l_size_r_chunk_comm, MPI_FLOAT, 1, tag1,
                      MPI_COMM_WORLD, &l_reqs[1]);
            MPI_Waitall(3, l_reqs, MPI_STATUSES_IGNORE);

            // receive output from last chunk
            MPI_Irecv(reinterpret_cast<float *>(l_ten_out_mpi_split[j - 1].data_ptr()) + offset_o,
                      l_size_o_chunk_comm, MPI_FLOAT, 1, tag0,
                      MPI_COMM_WORLD, &l_reqs[2]);
          }

          MPI_Wait(&l_reqs[2], MPI_STATUS_IGNORE);

          // receive last chunk
          MPI_Recv(reinterpret_cast<float *>(l_ten_out_mpi_split[chunks - 1].data_ptr()) + offset_o,
                   l_size_o_chunk, MPI_FLOAT, 1, tag0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      }

      if (i > 0) {
        l_tp1 = std::chrono::steady_clock::now();
        l_dur += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1 - l_tp0);

        if (i == 10) {
          l_time = l_dur.count() / 10;
          l_gflops = 1.0E-9 * l_n_flops / l_time;

          std::cout << size_1 << "," << l_gflops << std::endl;
        }
      }
    } else {
      l_tp0 = std::chrono::steady_clock::now();
      std::chrono::duration<double> l_dur_temp_comm = std::chrono::duration<double>::zero();
      std::chrono::duration<double> l_dur_temp_contract = std::chrono::duration<double>::zero();

      std::array<at::Tensor, 2> l_ten_left_mpi = {at::zeros({l_size_l_chunk}), at::zeros({l_size_l_chunk})};
      std::array<at::Tensor, 2> l_ten_right_mpi = {at::zeros({l_size_r_chunk}), at::zeros({l_size_r_chunk})};
      std::array<at::Tensor, 2> l_ten_out_mpi = {at::zeros({l_size_o_chunk}), at::zeros({l_size_o_chunk})};

      auto l_tp_dataPrep = std::chrono::steady_clock::now();
      if (i > 0)
        l_dur_dataPrep += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp_dataPrep - l_tp0);

#pragma omp parallel num_threads(1 + comm_threads)
#pragma omp reduce(max : l_dur_temp_contract)
#pragma omp reduce(max : l_dur_temp_comm)
      {
        auto thread = omp_get_thread_num() - 1; // -1 to get thread id starting from 0
        auto tag0 = thread;
        auto tag1 = thread + comm_threads;
        int64_t offset_l = thread * l_size_l_chunk_comm;
        int64_t offset_r = thread * l_size_r_chunk_comm;
        int64_t offset_o = thread * l_size_o_chunk_comm;

        MPI_Request l_reqs[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

        if (omp_get_thread_num() != 0) {
          auto l_tp0_comm = std::chrono::steady_clock::now();

          // receive first chunk
          MPI_Irecv(reinterpret_cast<float *>(l_ten_left_mpi[0].data_ptr()) + offset_l,
                    l_size_l_chunk_comm, MPI_FLOAT, 0, tag0,
                    MPI_COMM_WORLD, &l_reqs[0]);
          MPI_Irecv(reinterpret_cast<float *>(l_ten_right_mpi[0].data_ptr()) + offset_r,
                    l_size_r_chunk_comm, MPI_FLOAT, 0, tag1,
                    MPI_COMM_WORLD, &l_reqs[1]);
          MPI_Waitall(2, l_reqs, MPI_STATUSES_IGNORE);

          auto l_tp1_comm = std::chrono::steady_clock::now();
          l_dur_temp_comm = std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1_comm - l_tp0_comm);
        }

#pragma omp barrier

        for (int j = 1; j < chunks_r1; j++) {
          if (omp_get_thread_num() == 0) {
            auto l_tp0_contract = std::chrono::steady_clock::now();

            // this will spawn multiple openmp threads internally
            l_bin_cont_mpi.contract(l_ten_left_mpi[(j - 1) % 2].data_ptr(),
                                    l_ten_right_mpi[(j - 1) % 2].data_ptr(),
                                    l_ten_out_mpi[(j - 1) % 2].data_ptr());

            auto l_tp1_contract = std::chrono::steady_clock::now();
            l_dur_temp_contract += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1_contract - l_tp0_contract);
          } else {
            auto l_tp0_comm = std::chrono::steady_clock::now();

            // preload data for second contraction
            MPI_Irecv(reinterpret_cast<float *>(l_ten_left_mpi[j % 2].data_ptr()) + offset_l,
                      l_size_l_chunk_comm, MPI_FLOAT, 0, tag0,
                      MPI_COMM_WORLD, &l_reqs[0]);
            MPI_Irecv(reinterpret_cast<float *>(l_ten_right_mpi[j % 2].data_ptr()) + offset_r,
                      l_size_r_chunk_comm, MPI_FLOAT, 0, tag1,
                      MPI_COMM_WORLD, &l_reqs[1]);
            MPI_Waitall(3, l_reqs, MPI_STATUSES_IGNORE);

            auto l_tp1_comm = std::chrono::steady_clock::now();
            l_dur_temp_comm += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1_comm - l_tp0_comm);
          }

#pragma omp barrier

          if (omp_get_thread_num() != 0) {
            auto l_tp0_comm = std::chrono::steady_clock::now();

            // queue send into next receive
            MPI_Isend(reinterpret_cast<float *>(l_ten_out_mpi[(j - 1) % 2].data_ptr()) + offset_o,
                      l_size_o_chunk_comm, MPI_FLOAT, 0, tag0,
                      MPI_COMM_WORLD, &l_reqs[2]);

            auto l_tp1_comm = std::chrono::steady_clock::now();
            l_dur_temp_comm += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1_comm - l_tp0_comm);
          }
        }

        if (omp_get_thread_num() == 0) {
          auto l_tp0_contract = std::chrono::steady_clock::now();

          // contract last chunk
          l_bin_cont_mpi.contract(l_ten_left_mpi[(chunks_r1 - 1) % 2].data_ptr(),
                                  l_ten_right_mpi[(chunks_r1 - 1) % 2].data_ptr(),
                                  l_ten_out_mpi[(chunks_r1 - 1) % 2].data_ptr());

          auto l_tp1_contract = std::chrono::steady_clock::now();
          l_dur_temp_contract += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1_contract - l_tp0_contract);
        } else {
          auto l_tp0_comm = std::chrono::steady_clock::now();

          // send second to last chunk back
          MPI_Wait(&l_reqs[2], MPI_STATUS_IGNORE);

          auto l_tp1_comm = std::chrono::steady_clock::now();
          l_dur_temp_comm += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1_comm - l_tp0_comm);
        }

#pragma omp barrier

        if (omp_get_thread_num() != 0) {
          auto l_tp0_comm = std::chrono::steady_clock::now();

          // send last chunk
          MPI_Send(reinterpret_cast<float *>(l_ten_out_mpi[(chunks_r1 - 1) % 2].data_ptr()) + offset_o,
                   l_size_o_chunk_comm, MPI_FLOAT, 0, tag0,
                   MPI_COMM_WORLD);

          auto l_tp1_comm = std::chrono::steady_clock::now();
          l_dur_temp_comm += std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1_comm - l_tp0_comm);
        }
      }

      l_tp1 = std::chrono::steady_clock::now();
    }
  }
}

int main(int argc, char const *argv[]) {

  int provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);

  omp_set_nested(true); // allow multiple nested parallel regions

  if (argc < 2 || argc > 4) {
    blocked_binary_contraction(96, 86);
  } else if (argc == 2) {
    int size_1 = atoi(argv[1]);
    blocked_binary_contraction(size_1, 86);
  } else if (argc == 3) {
    int size_1 = atoi(argv[1]);
    int size_2 = atoi(argv[2]);
    blocked_binary_contraction(size_1, size_2);
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}
