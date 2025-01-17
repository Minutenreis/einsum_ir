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

void benchmark(int64_t size_1, int64_t size_2) {
  std::chrono::steady_clock::time_point tp0, tp1;
  std::chrono::duration<double> dur, dur_mpi;

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

  Tensor left;
  Tensor right;
  Tensor out;

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

  auto l_gflops = 1.0E-9 * l_n_flops / dur.count();

  std::cout << size_1 << ", " << l_gflops << std::endl; // size_1
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
