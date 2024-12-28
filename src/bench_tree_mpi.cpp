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
  Tensor(int64_t size, std::vector<int64_t> dim_ids) : size(size), dim_ids(dim_ids) {
    data = new datatype[size];
  }
  ~Tensor() {
    delete[] data;
  }
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

  int64_t chunk_size = out.size / num_ranks;

  MPI_Request reqs[2];

  datatype *new_buffer = new datatype[left.size];

  datatype *calc_buffer = left.data;
  datatype *recv_buffer = new_buffer;

#pragma omp parallel num_threads(2)
  {
    for (int i = 0; i < num_ranks - 1; i++) {
      if (omp_get_thread_num() == 0) {
        MPI_Isend(calc_buffer, right.size, datatypeMPI, (rank - 1) % num_ranks, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recv_buffer, right.size, datatypeMPI, (rank + 1) % num_ranks, 0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
      } else {
        bin_cont.contract(left.data, calc_buffer, out.data + i * chunk_size);
      }
      // might need barrier here?
#pragma omp single
      std::swap(calc_buffer, recv_buffer);
    }
  }

  bin_cont.contract(left.data, calc_buffer, out.data + (num_ranks - 1) * chunk_size);

  delete[] new_buffer;
}

// expect dim_sizes to be fitting the distributed tensor and not the original
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

  int64_t chunk_size = out.size / num_ranks;

  MPI_Request reqs[2];

  datatype *new_buffer = new datatype[left.size];

  datatype *calc_buffer = left.data;
  datatype *recv_buffer = new_buffer;

#pragma omp parallel num_threads(2)
  {
    for (int i = 0; i < num_ranks - 1; i++) {
      if (omp_get_thread_num() == 0) {
        MPI_Isend(calc_buffer, left.size, datatypeMPI, (rank - 1) % num_ranks, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recv_buffer, left.size, datatypeMPI, (rank + 1) % num_ranks, 0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
      } else {
        bin_cont.contract(calc_buffer, right.data, out.data + i * chunk_size);
      }
      // might need barrier here?
#pragma omp single
      std::swap(calc_buffer, recv_buffer);
    }
  }

  bin_cont.contract(calc_buffer, right.data, out.data + (num_ranks - 1) * chunk_size);

  delete[] new_buffer;
}

datatype *getOffset(datatype *data, int rank, int num_ranks, int64_t chunk_size, int64_t step) {
  return data + (((rank + 1) * 2 + step) % (2 * num_ranks)) * chunk_size;
}

// expects outer most dimension of out and right to be an "n" dimension and divisible by 2 * num_ranks
void contract_distributed_k(Tensor &left, Tensor &right, Tensor &out, std::map<int64_t, int64_t> dim_sizes) {
  einsum_ir::backend::BinaryContractionTpp bin_cont;

  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto chunk_dim_out = out.dim_ids[0];
  auto chunk_dim_right = right.dim_ids[0];

  assert(dim_sizes[chunk_dim_out] % 2 == 0);
  assert(dim_sizes[chunk_dim_right] % 2 == 0);

  dim_sizes[chunk_dim_out] = dim_sizes[chunk_dim_out] / 2;
  dim_sizes[chunk_dim_right] = dim_sizes[chunk_dim_right] / 2;

  bin_cont.init(left.dim_ids.size(), right.dim_ids.size(), out.dim_ids.size(),
                &dim_sizes, &dim_sizes, &dim_sizes, nullptr, &dim_sizes,
                left.dim_ids.data(), right.dim_ids.data(), out.dim_ids.data(),
                datatypeEinsum, datatypeEinsum, datatypeEinsum, datatypeEinsum,
                einsum_ir::ZERO, einsum_ir::MADD, einsum_ir::UNDEFINED_KTYPE);

  bin_cont.compile();
  bin_cont.threading(omp_get_max_threads() * 4);

  int64_t buffer_size = out.size / 2;
  int64_t chunk_size_right = right.size / 2;
  datatype *new_buffer = new datatype[buffer_size];

  int start = ((rank + 1) % num_ranks) * 2;

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
  for (int i = 0; i < (2 * num_ranks) % 3; i++) {
    datatype *tmp = send_buffer;
    send_buffer = calc_buffer;
    calc_buffer = recv_buffer;
    recv_buffer = tmp;
  }

  MPI_Request reqs[2];
  bin_cont.contract(left.data, getOffset(right.data, rank, num_ranks, chunk_size_right, 0), calc_buffer);

  datatype *tmp = send_buffer;
  send_buffer = calc_buffer;
  calc_buffer = recv_buffer;
  recv_buffer = tmp;

#pragma omp parallel num_threads(2)
  {
    for (int i = 1; i < num_ranks * 2 - 1; i++) {
      if (omp_get_thread_num() == 0) {
        MPI_Isend(send_buffer, buffer_size, datatypeMPI, (rank - 1) % num_ranks, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recv_buffer, buffer_size, datatypeMPI, (rank + 1) % num_ranks, 0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
      } else {
        bin_cont.contract(left.data, getOffset(right.data, rank, num_ranks, chunk_size_right, i), calc_buffer);
      }
      // might need barrier here?
#pragma omp single
      {
        tmp = send_buffer;
        send_buffer = calc_buffer;
        calc_buffer = recv_buffer;
        recv_buffer = tmp;
      }
    }
  }
  bin_cont.contract(left.data, getOffset(right.data, rank, num_ranks, chunk_size_right, num_ranks * 2 - 1), calc_buffer);
}

main(int argc, char *argv[]) {
  omp_set_nested(true); // allow multiple nested parallel regions

  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, NULL);

  /* code */
  return 0;
}
