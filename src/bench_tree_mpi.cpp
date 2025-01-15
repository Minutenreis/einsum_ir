#include <cstdlib>
#include <iostream>

#include <omp.h>

#include "backend/BinaryContractionTpp.h"
#include "backend/EinsumNode.h"
#include <ATen/ATen.h>
#include <mpi.h>

const auto datatypeEinsum = einsum_ir::FP32;
const auto datatypeMPI = MPI_FLOAT;
using datatype = float;

// expect rank0 to always have the "lower" half and rank1 to have the "upper" half
struct Tensor {
  std::vector<int64_t> dim_ids; // ids of the tensor dimensions
  int64_t size;                 // in #elements
  datatype *data;               // pointer to data
};

/**
 * IMPORTANT: After each function call, the data of the input tensor is arbitrary.
 */

// expect dim_sizes to be fitting the distributed tensor and not the original
void contract_distributed_c(Tensor &left, Tensor &right, Tensor &out, std::map<int64_t, int64_t> dim_sizes) {
  einsum_ir::backend::BinaryContractionTpp bin_cont;

  bin_cont.init(left.dim_ids.size(), right.dim_ids.size(), out.dim_ids.size(),
                &dim_sizes, &dim_sizes, &dim_sizes, nullptr, &dim_sizes,
                left.dim_ids.data(), right.dim_ids.data(), out.dim_ids.data(),
                datatypeEinsum, datatypeEinsum, datatypeEinsum, datatypeEinsum,
                einsum_ir::ZERO, einsum_ir::MADD, einsum_ir::UNDEFINED_KTYPE);

  bin_cont.compile();
  bin_cont.threading(omp_get_max_threads() * 4);

  bin_cont.contract(left.data, right.data, out.data);
}

// expect dim_sizes to be fitting the distributed tensor and not the original
// cmk = left, cnk = right, cmn = out (not in that order)
// expects m to be the outer most dimension in the output
void contract_distributed_m_n_out_n(Tensor &left, Tensor &right, Tensor &out, std::map<int64_t, int64_t> dim_sizes) {
  einsum_ir::backend::BinaryContractionTpp bin_cont;

  bin_cont.init(left.dim_ids.size(), right.dim_ids.size(), out.dim_ids.size(),
                &dim_sizes, &dim_sizes, &dim_sizes, nullptr, &dim_sizes,
                left.dim_ids.data(), right.dim_ids.data(), out.dim_ids.data(),
                datatypeEinsum, datatypeEinsum, datatypeEinsum, datatypeEinsum,
                einsum_ir::ZERO, einsum_ir::MADD, einsum_ir::UNDEFINED_KTYPE);

  bin_cont.compile();
  bin_cont.threading(omp_get_max_threads() * 4);

  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int previous = (rank - 1 + num_ranks) % num_ranks; // + num_ranks to avoid negative values
  int next = (rank + 1) % num_ranks;

  int64_t chunk_size = out.size / num_ranks;

  MPI_Request reqs[2];

  datatype *new_buffer = new datatype[left.size];

  datatype *calc_buffer = left.data;
  datatype *recv_buffer = new_buffer;

#pragma omp parallel num_threads(2)
  {
    for (int i = 0; i < num_ranks - 1; i++) {
      if (omp_get_thread_num() == 0) {
        MPI_Isend(calc_buffer, left.size, datatypeMPI, previous, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recv_buffer, left.size, datatypeMPI, next, 0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
      } else {
        bin_cont.contract(calc_buffer, right.data, out.data + ((rank + i) % num_ranks) * chunk_size);
      }
      // might need barrier here?
#pragma omp single
      std::swap(calc_buffer, recv_buffer);
    }
  }

  bin_cont.contract(calc_buffer, right.data, out.data + ((rank + num_ranks - 1) % num_ranks) * chunk_size);

  delete[] new_buffer;
}

// expect dim_sizes to be fitting the distributed tensor and not the original
// cmk = left, cnk = right, cmn = out (not in that order)
// expects n to be the outer most dimension in the output
void contract_distributed_m_n_out_m(Tensor &left, Tensor &right, Tensor &out, std::map<int64_t, int64_t> dim_sizes) {
  einsum_ir::backend::BinaryContractionTpp bin_cont;

  bin_cont.init(left.dim_ids.size(), right.dim_ids.size(), out.dim_ids.size(),
                &dim_sizes, &dim_sizes, &dim_sizes, nullptr, &dim_sizes,
                left.dim_ids.data(), right.dim_ids.data(), out.dim_ids.data(),
                datatypeEinsum, datatypeEinsum, datatypeEinsum, datatypeEinsum,
                einsum_ir::ZERO, einsum_ir::MADD, einsum_ir::UNDEFINED_KTYPE);

  bin_cont.compile();
  bin_cont.threading(omp_get_max_threads() * 4);

  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // + num_ranks to avoid negative values
  int previous = (rank - 1 + num_ranks) % num_ranks;
  int next = (rank + 1) % num_ranks;

  int64_t chunk_size = out.size / num_ranks;

  MPI_Request reqs[2];

  datatype *new_buffer = new datatype[right.size];

  datatype *calc_buffer = right.data;
  datatype *recv_buffer = new_buffer;

#pragma omp parallel num_threads(2)
  {
    for (int i = 0; i < num_ranks - 1; i++) {
      if (omp_get_thread_num() == 0) {
        MPI_Isend(calc_buffer, right.size, datatypeMPI, previous, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recv_buffer, right.size, datatypeMPI, next, 0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
      } else {
        bin_cont.contract(left.data, calc_buffer, out.data + ((rank + i) % num_ranks) * chunk_size);
      }
      // might need barrier here?
#pragma omp single
      std::swap(calc_buffer, recv_buffer);
    }
  }
  bin_cont.contract(left.data, calc_buffer, out.data + ((rank + num_ranks - 1) % num_ranks) * chunk_size);

  delete[] new_buffer;
}

datatype *getOffsetRight(datatype *data, int rank, int num_ranks, int64_t chunk_size, int64_t step) {
  return data + ((rank + 1 + step / 2) % num_ranks) * chunk_size;
}

datatype *getOffsetLeft(datatype *data, int64_t chunk_size, int64_t step) {
  return data + (step % 2) * chunk_size;
}

void rotate(datatype *&send_buffer, datatype *&calc_buffer, datatype *&recv_buffer) {
  datatype *tmp = send_buffer;
  send_buffer = calc_buffer;
  calc_buffer = recv_buffer;
  recv_buffer = tmp;
}

// expects outer most dimension of right to be an "n" dimension and divisible by num_ranks
// expects outer most dimension of out and left to be an "m" dimension and divisible 2
void contract_distributed_k(Tensor &left, Tensor &right, Tensor &out, std::map<int64_t, int64_t> dim_sizes) {
  einsum_ir::backend::BinaryContractionTpp bin_cont;

  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int previous = (rank - 1 + num_ranks) % num_ranks;
  int next = (rank + 1) % num_ranks;

  auto chunk_dim_n = right.dim_ids[0];
  auto chunk_dim_m = left.dim_ids[0];
  // outer most dimension of left and out has to be the same (and "m")
  assert(out.dim_ids[0] == left.dim_ids[0]);
  // the distributed dimension of right and out has to be divisible by num_ranks
  assert(dim_sizes[chunk_dim_n] % num_ranks == 0);
  // the distributed dimension of left has to be divisible by 2
  assert(dim_sizes[chunk_dim_m] % 2 == 0);
  // the outer most dimension of right has to be "n"
  assert(std::find(left.dim_ids.begin(), left.dim_ids.end(), chunk_dim_n) == left.dim_ids.end());
  // the outer most dimension of left and out has to be "m"
  assert(std::find(right.dim_ids.begin(), right.dim_ids.end(), chunk_dim_m) == right.dim_ids.end());

  dim_sizes[chunk_dim_n] /= num_ranks;
  dim_sizes[chunk_dim_m] /= 2;

  // i_k_type_first_touch -> zero = 0 initialisation, undefined = no initialisation (taken as is)
  bin_cont.init(left.dim_ids.size(), right.dim_ids.size(), out.dim_ids.size(),
                &dim_sizes, &dim_sizes, &dim_sizes, nullptr, &dim_sizes,
                left.dim_ids.data(), right.dim_ids.data(), out.dim_ids.data(),
                datatypeEinsum, datatypeEinsum, datatypeEinsum, datatypeEinsum,
                einsum_ir::UNDEFINED_KTYPE, einsum_ir::MADD, einsum_ir::UNDEFINED_KTYPE);

  bin_cont.compile();
  bin_cont.threading(omp_get_max_threads() * 4);

  int64_t buffer_size = out.size / 2;
  int64_t chunk_size_right = right.size / num_ranks;
  int64_t chunk_size_left = left.size / 2;

  datatype *new_buffer = new datatype[buffer_size]{};

  datatype *send_buffer = out.data;
  datatype *calc_buffer = out.data + buffer_size;
  datatype *recv_buffer = new_buffer;

  /**
   * at the end send_buffer has to be out.data and calc_buffer has to be out.data + buffer_size
   * in each step they rotate (recv -> calc -> send -> recv)
   * they get rotated 2*num_ranks times
   * after 3 rotations the data is the same
   * -> prerotate buffers so the data is in the right place at the end
   */
  for (int i = 0; i < (num_ranks + 1) % 3; i++) {
    rotate(send_buffer, calc_buffer, recv_buffer);
  }

  MPI_Request reqs[2];
  bin_cont.contract(
      getOffsetLeft(left.data, chunk_size_left, 0),
      getOffsetRight(right.data, rank, num_ranks, chunk_size_right, 0),
      calc_buffer);

  rotate(send_buffer, calc_buffer, recv_buffer);

#pragma omp parallel num_threads(2)
  {
    for (int i = 1; i < num_ranks * 2 - 1; i++) {
      if (omp_get_thread_num() == 0) {
        MPI_Isend(send_buffer, buffer_size, datatypeMPI, previous, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recv_buffer, buffer_size, datatypeMPI, next, 0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
      } else {
        bin_cont.contract(
            getOffsetLeft(left.data, chunk_size_left, i),
            getOffsetRight(right.data, rank, num_ranks, chunk_size_right, i),
            calc_buffer);
      }
// might need barrier here?
#pragma omp single
      rotate(send_buffer, calc_buffer, recv_buffer);
    }
  }
  bin_cont.contract(
      getOffsetLeft(left.data, chunk_size_left, num_ranks * 2 - 1),
      getOffsetRight(right.data, rank, num_ranks, chunk_size_right, num_ranks * 2 - 1),
      calc_buffer);

  delete[] new_buffer;
}

void benchmark(int64_t size_1, int64_t size_2) {
  std::chrono::steady_clock::time_point tp0, tp1;
  std::chrono::duration<double> dur, dur_mpi;

  const int64_t l_size_c0 = 2;

  const int64_t l_size_m0 = size_1;
  const int64_t l_size_m1 = size_2;

  const int64_t l_size_n0 = size_2;
  const int64_t l_size_n1 = size_2;

  const int64_t l_size_k0 = size_1;
  const int64_t l_size_k1 = size_1;

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

  Tensor left;
  Tensor right;
  Tensor out;

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if (rank == 0) {
    // std::cout << "einsum_ir:" << std::endl;

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
                     //        m0 c0 k0 k1 m1
                     .permute({1, 0, 3, 4, 2})
                     .contiguous();

    l_ten_right = l_ten_right.view({
                                       l_size_c0, // 0
                                       l_size_n0, // 1
                                       l_size_n1, // 2
                                       l_size_k0, // 3
                                       l_size_k1, // 4
                                   })
                      //        n0 c0 k0 n1 k1
                      .permute({1, 0, 3, 2, 4})
                      .contiguous();

    l_ten_out = l_ten_out.view({
                                   l_size_c0, // 0
                                   l_size_n0, // 1
                                   l_size_n1, // 2
                                   l_size_m0, // 3
                                   l_size_m1, // 4
                               })
                    //        m0 n0 c0 n1 m1
                    .permute({3, 1, 0, 4, 2})
                    .contiguous();

    left = {l_dim_ids_in_left, l_size_left, l_ten_left.data_ptr<datatype>()};
    right = {l_dim_ids_in_right, l_size_right, l_ten_right.data_ptr<datatype>()};
    out = {l_dim_ids_out, l_size_out, l_ten_out.data_ptr<datatype>()};

    tp0 = std::chrono::steady_clock::now();
    einsum_ir::backend::BinaryContractionTpp bin_cont;
    bin_cont.init(left.dim_ids.size(), right.dim_ids.size(), out.dim_ids.size(),
                  &l_dim_sizes, &l_dim_sizes, &l_dim_sizes, nullptr, &l_dim_sizes,
                  left.dim_ids.data(), right.dim_ids.data(), out.dim_ids.data(),
                  datatypeEinsum, datatypeEinsum, datatypeEinsum, datatypeEinsum,
                  einsum_ir::ZERO, einsum_ir::MADD, einsum_ir::UNDEFINED_KTYPE);

    // std::cout << "  compile" << std::endl;
    bin_cont.compile();
    bin_cont.threading(omp_get_max_threads() * 4);

    // std::cout << "  contract" << std::endl;

    bin_cont.contract(left.data, right.data, out.data);
    tp1 = std::chrono::steady_clock::now();
    dur = std::chrono::duration_cast<std::chrono::duration<double>>(tp1 - tp0);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  {
    Tensor left_destributed;
    Tensor right_destributed;
    Tensor out_destributed;

    auto chunk_size_left = l_size_left / num_ranks;
    auto chunk_size_right = l_size_right / num_ranks;
    auto chunk_size_out = l_size_out / num_ranks;

    std::vector<at::Tensor> left_mpi;
    std::vector<at::Tensor> right_mpi;
    std::vector<at::Tensor> out_mpi;

    int einsum_split_left_dim = m0;
    int einsum_split_right_dim = n0;
    int einsum_split_out_dim = n0;

    auto dim_sizes = l_dim_sizes;
    dim_sizes[m0] /= num_ranks;
    dim_sizes[n0] /= num_ranks;

    int ten_split_left_dim, ten_split_right_dim, ten_split_out_dim;

    // predistribute data
    if (rank == 0) {
      // torch id of dimension to split
      auto it = std::find(left.dim_ids.begin(), left.dim_ids.end(), einsum_split_left_dim);
      ten_split_left_dim = std::distance(left.dim_ids.begin(), it);
      it = std::find(right.dim_ids.begin(), right.dim_ids.end(), einsum_split_right_dim);
      ten_split_right_dim = std::distance(right.dim_ids.begin(), it);
      it = std::find(out.dim_ids.begin(), out.dim_ids.end(), einsum_split_out_dim);
      ten_split_out_dim = std::distance(out.dim_ids.begin(), it);

      // std::cout << "einsum_ir_mpi:" << std::endl;

      // std::cout << "  scatter" << std::endl;

      l_ten_out2 = at::zeros_like(l_ten_out);

      left_mpi = l_ten_left.chunk(num_ranks, ten_split_left_dim);
      right_mpi = l_ten_right.chunk(num_ranks, ten_split_right_dim);
      out_mpi = l_ten_out2.chunk(num_ranks, ten_split_out_dim);

      for (int i = 0; i < num_ranks; i++) {
        left_mpi[i] = left_mpi[i].contiguous();
        right_mpi[i] = right_mpi[i].contiguous();
        out_mpi[i] = out_mpi[i].contiguous();
      }

      MPI_Request reqs[(num_ranks - 1) * 2];
      for (int i = 1; i < num_ranks; i++) {
        MPI_Isend(left_mpi[i].data_ptr(), chunk_size_left, datatypeMPI, i, 0, MPI_COMM_WORLD, &reqs[i - 1]);
        MPI_Isend(right_mpi[i].data_ptr(), chunk_size_right, datatypeMPI, i, 1, MPI_COMM_WORLD, &reqs[i - 1 + num_ranks - 1]);
      }

      left_destributed = {l_dim_ids_in_left, chunk_size_left, left_mpi[0].data_ptr<datatype>()};
      right_destributed = {l_dim_ids_in_right, chunk_size_right, right_mpi[0].data_ptr<datatype>()};
      out_destributed = {l_dim_ids_out, chunk_size_out, out_mpi[0].data_ptr<datatype>()};

      MPI_Waitall(2 * (num_ranks - 1), reqs, MPI_STATUSES_IGNORE);

      // std::cout << "  contract" << std::endl;
    } else {
      left_destributed = {l_dim_ids_in_left, chunk_size_left, new datatype[chunk_size_left]};
      right_destributed = {l_dim_ids_in_right, chunk_size_right, new datatype[chunk_size_right]};
      out_destributed = {l_dim_ids_out, chunk_size_out, new datatype[chunk_size_out]};

      MPI_Request reqs[2];

      MPI_Irecv(left_destributed.data, left_destributed.size, datatypeMPI, 0, 0, MPI_COMM_WORLD, &reqs[0]);
      MPI_Irecv(right_destributed.data, right_destributed.size, datatypeMPI, 0, 1, MPI_COMM_WORLD, &reqs[1]);
      MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    }

    // for time measurement
    MPI_Barrier(MPI_COMM_WORLD);
    tp0 = std::chrono::steady_clock::now();
    // contract
    contract_distributed_m_n_out_n(left_destributed, right_destributed, out_destributed, dim_sizes);

    // for time measurement
    MPI_Barrier(MPI_COMM_WORLD);
    tp1 = std::chrono::steady_clock::now();
    dur_mpi = std::chrono::duration_cast<std::chrono::duration<double>>(tp1 - tp0);

    // gather data and cleanup
    if (rank == 0) {
      // std::cout << "  gather" << std::endl;

      MPI_Request reqs[num_ranks - 1];
      for (int i = 1; i < num_ranks; i++) {
        MPI_Irecv(out_mpi[i].data_ptr(), chunk_size_out, datatypeMPI, i, 0, MPI_COMM_WORLD, &reqs[i - 1]);
      }

      MPI_Waitall(num_ranks - 1, reqs, MPI_STATUSES_IGNORE);

      l_ten_out2 = at::cat(out_mpi, ten_split_out_dim).contiguous();

      auto l_gflops = 1.0E-9 * l_n_flops / dur.count();
      auto l_gflops_mpi = 1.0E-9 * l_n_flops / dur_mpi.count();

      std::cout << size_1 << ", " << l_gflops << ", " << l_gflops_mpi << ", " << l_gflops_mpi / l_gflops << std::endl; // size_1
    } else {
      MPI_Send(out_destributed.data, out_destributed.size, datatypeMPI, 0, 0, MPI_COMM_WORLD);

      delete[] left_destributed.data;
      delete[] right_destributed.data;
      delete[] out_destributed.data;
    }
  }
}

int main(int argc, char const *argv[]) {

  int provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);

  omp_set_nested(true); // allow multiple nested parallel regions

  at::set_default_dtype(caffe2::TypeMeta::Make<datatype>()); // set default datatype

  if (argc != 2 && argc != 3) {
    benchmark(96, 86);
  } else if (argc == 2) {
    int size_1 = atoi(argv[1]);
    benchmark(size_1, 86);
  } else if (argc == 3) {
    int size_1 = atoi(argv[1]);
    int size_2 = atoi(argv[2]);
    benchmark(size_1, size_2);
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}
