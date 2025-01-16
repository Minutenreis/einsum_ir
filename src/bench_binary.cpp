#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "backend/BinaryContractionTpp.h"
#include "backend/EinsumNode.h"
#include "frontend/EinsumExpressionAscii.h"
#include <ATen/ATen.h>

void bench_binary(std::map<int64_t, int64_t> &i_dim_sizes_map,
                  std::vector<int64_t> &i_dim_ids_in_left,
                  std::vector<int64_t> &i_dim_ids_in_right,
                  std::vector<int64_t> &i_dim_ids_out,
                  std::vector<int64_t> *i_loop_order,
                  at::ScalarType i_dtype_at,
                  einsum_ir::data_t i_dtype_einsum_ir,
                  std::string i_einsum_string) {
  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration<double> l_dur;
  int64_t l_n_flops = 0;
  int64_t l_repetitions = 1;
  int64_t l_repetitions_warm_up = 1;
  std::vector<int64_t> l_dim_ids_permute_left;
  std::vector<int64_t> l_dim_ids_permute_right;
  double l_time_compile = 0;
  double l_time = 0;
  double l_gflops = 0;

  // create vectors of sizes and einsum string
  std::vector<int64_t> l_sizes_left;
  std::vector<int64_t> l_sizes_right;
  std::vector<int64_t> l_sizes_out;
  for (std::size_t l_di = 0; l_di < i_dim_ids_in_left.size(); l_di++) {
    l_sizes_left.push_back(i_dim_sizes_map[i_dim_ids_in_left[l_di]]);
  }
  for (std::size_t l_di = 0; l_di < i_dim_ids_in_right.size(); l_di++) {
    l_sizes_right.push_back(i_dim_sizes_map[i_dim_ids_in_right[l_di]]);
  }
  for (std::size_t l_di = 0; l_di < i_dim_ids_out.size(); l_di++) {
    l_sizes_out.push_back(i_dim_sizes_map[i_dim_ids_out[l_di]]);
  }

  // number of flops
  l_n_flops = 2;
  for (std::map<int64_t, int64_t>::iterator l_di = i_dim_sizes_map.begin(); l_di != i_dim_sizes_map.end(); l_di++) {
    int64_t l_dim_size = l_di->second;
    l_n_flops *= l_dim_size;
  }

  /*
   * at::einsum
   */

  // create Tensors
  at::Tensor l_ten_left = at::rand(at::IntArrayRef(l_sizes_left.data(), l_sizes_left.size()), i_dtype_at);
  at::Tensor l_ten_right = at::ones(at::IntArrayRef(l_sizes_right.data(), l_sizes_right.size()), i_dtype_at);
  at::Tensor l_ten_out = at::rand(at::IntArrayRef(l_sizes_out.data(), l_sizes_out.size()), i_dtype_at);
  // std::cout << "at::einsum:" << std::endl;

  // warm up
  // at::Tensor l_ten_out_torch;
  // l_tp0 = std::chrono::steady_clock::now();
  // for( int64_t l_rep = 0; l_rep < l_repetitions_warm_up; l_rep++ ){
  //   l_ten_out_torch = at::einsum( i_einsum_string,
  //                                 {l_ten_left, l_ten_right},
  //                                 { {0,1} } );
  // }
  // l_tp1 = std::chrono::steady_clock::now();
  // l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  // l_repetitions = l_repetitions_warm_up / l_dur.count() + 1;

  // // run with repititions
  // l_tp0 = std::chrono::steady_clock::now();
  // for( int64_t l_rep = 0; l_rep < l_repetitions; l_rep++ ){
  //   l_ten_out_torch = at::einsum( i_einsum_string,
  //                                 {l_ten_left, l_ten_right},
  //                                 { {0,1} } );
  // }
  // l_tp1 = std::chrono::steady_clock::now();
  // l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  // l_time = l_dur.count() / l_repetitions;
  // l_gflops = 1.0E-9 * l_n_flops / l_time;

  // std::cout << "  time (contract): " << l_time << std::endl;
  // std::cout << "  gflops: " << l_gflops << std::endl;

  /*
   * einsum_ir
   */
  // std::cout << "einsum_ir:" << std::endl;

  einsum_ir::backend::MemoryManager l_memory;
  einsum_ir::backend::BinaryContractionTpp l_bin_cont;
  l_bin_cont.init(i_dim_ids_in_left.size(),
                  i_dim_ids_in_right.size(),
                  i_dim_ids_out.size(),
                  &i_dim_sizes_map,
                  &i_dim_sizes_map,
                  &i_dim_sizes_map,
                  nullptr,
                  &i_dim_sizes_map,
                  i_loop_order,
                  i_dim_ids_in_left.data(),
                  i_dim_ids_in_right.data(),
                  i_dim_ids_out.data(),
                  l_dim_ids_permute_left.data(),
                  l_dim_ids_permute_right.data(),
                  &l_memory,
                  i_dtype_einsum_ir,
                  i_dtype_einsum_ir,
                  i_dtype_einsum_ir,
                  i_dtype_einsum_ir,
                  einsum_ir::ZERO,
                  einsum_ir::MADD,
                  einsum_ir::UNDEFINED_KTYPE);

  l_tp0 = std::chrono::steady_clock::now();
  l_bin_cont.compile();
  l_memory.alloc_all_memory();
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1 - l_tp0);
  l_time_compile = l_dur.count();

  // enable threading
#ifdef _OPENMP
  // four times overload
  int64_t l_num_tasks = omp_get_max_threads() * 4;

  l_bin_cont.threading(l_num_tasks);
#endif

  // warm up
  l_tp0 = std::chrono::steady_clock::now();
  for (int64_t l_rep = 0; l_rep < l_repetitions_warm_up; l_rep++) {
    l_bin_cont.contract(l_ten_left.data_ptr(),
                        l_ten_right.data_ptr(),
                        l_ten_out.data_ptr());
  }
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1 - l_tp0);
  l_repetitions = l_repetitions_warm_up / l_dur.count() + 1;

  l_tp0 = std::chrono::steady_clock::now();
  for (int64_t l_rep = 0; l_rep < l_repetitions; l_rep++) {
    l_bin_cont.contract(l_ten_left.data_ptr(),
                        l_ten_right.data_ptr(),
                        l_ten_out.data_ptr());
  }
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast<std::chrono::duration<double>>(l_tp1 - l_tp0);
  l_time = l_dur.count() / l_repetitions;
  l_gflops = 1.0E-9 * l_n_flops / l_time;

  // std::cout << "  time (compile): " << l_time_compile << std::endl;
  // std::cout << "  time (contract): " << l_time << std::endl;
  // std::cout << "  gflops: " << l_gflops << std::endl;
  std::cout << omp_get_max_threads() << "," << l_gflops << std::endl;

  // if (!at::allclose(l_ten_out_torch, l_ten_out, 1e-03)) {
  //   std::cerr << "error: einsum_ir solution is not close to aten!" << std::endl;
  //   std::cout << "maximal difference is: " << at::max(l_ten_out_torch - l_ten_out) << std::endl;
  // } else {
  //   std::cout << "  results are close" << std::endl;
  // }
}

int main(int i_argc,
         char *i_argv[]) {
  // std::cout << "running bench_binary!" << std::endl;

  if (i_argc < 3) {
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  bench_binary einsum_string dimension_sizes dtype loop_order" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Arguments:" << std::endl;
    std::cerr << "  * einsum_string:    Einsum string with a binary contraction" << std::endl;
    std::cerr << "  * dimension_sizes:  Dimension sizes have to be in ascending order of the dimension ids." << std::endl;
    std::cerr << "  * dtype:            FP32, FP64, CPX_FP32 or CPX_FP64, default: FP32." << std::endl;
    std::cerr << "  * loop_order:       Loop execution strategy" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Example:" << std::endl;
    std::cerr << "  ./bench_binary \"abc,acdb->abd\" \"32,8,4,2\" FP32 \"a,b,c,d\"" << std::endl;
    return EXIT_FAILURE;
  }

  /**
   * parse expression string
   **/
  std::string l_expression_string_arg(i_argv[1]);
  std::string l_expression_string_std;
  std::string l_expression_string_schar;

  if (l_expression_string_arg[0] == '[') {
    l_expression_string_std = l_expression_string_arg;
    einsum_ir::frontend::EinsumExpressionAscii::standard_to_schar(l_expression_string_std,
                                                                  l_expression_string_schar);
  } else {
    l_expression_string_schar = l_expression_string_arg;
    einsum_ir::frontend::EinsumExpressionAscii::schar_to_standard(l_expression_string_schar,
                                                                  l_expression_string_std);
  }

  /**
   * parse input tensors and output tensors
   **/
  std::vector<std::string> l_tensors;
  einsum_ir::frontend::EinsumExpressionAscii::parse_tensors(l_expression_string_std,
                                                            l_tensors);
  int64_t l_num_tensors = l_tensors.size();

  if (l_num_tensors != 3) {
    std::cerr << "Einsum string is not a binary contraction" << std::endl;
    return EXIT_FAILURE;
  }

  // std::cout << "parsed tensors:" << std::endl;
  // for (int64_t l_te = 0; l_te < l_num_tensors; l_te++) {
  //   std::cout << "  " << l_tensors[l_te] << std::endl;
  // }

  /*
   * create mapping from dimension name to id
   */
  std::map<std::string, int64_t> m_map_dim_name_to_id;
  einsum_ir::frontend::EinsumExpressionAscii::parse_dim_ids(l_expression_string_std,
                                                            m_map_dim_name_to_id);

  // std::cout << "parsed dimension ids:" << std::endl;
  // for (std::map<std::string, int64_t>::iterator l_di = m_map_dim_name_to_id.begin(); l_di != m_map_dim_name_to_id.end(); l_di++) {
  //   std::string l_dim_name = l_di->first;
  //   int64_t l_dim_id = l_di->second;

  //   std::cout << "  " << l_dim_name << ": " << l_dim_id << std::endl;
  // }

  /*
   * parse dimension sizes
   */
  std::string l_dim_sizes_string(i_argv[2]);
  std::vector<int64_t> l_dim_sizes_vec;
  einsum_ir::frontend::EinsumExpressionAscii::parse_dim_sizes(l_dim_sizes_string,
                                                              l_dim_sizes_vec);

  // std::cout << "parsed dimension sizes:" << std::endl;
  // iterate over keys of map dim name to id
  // for (std::map<std::string, int64_t>::iterator l_di = m_map_dim_name_to_id.begin(); l_di != m_map_dim_name_to_id.end(); l_di++) {
  //   std::string l_dim_name = l_di->first;
  //   int64_t l_dim_id = l_di->second;
  //   int64_t l_dim_size = l_dim_sizes_vec[l_dim_id];

  //   std::cout << "  " << l_dim_name << ": " << l_dim_size << std::endl;
  // }

  /*
   * parse dtype
   */
  at::ScalarType l_dtype_at = at::ScalarType::Float;
  einsum_ir::data_t l_dtype_einsum_ir = einsum_ir::FP32;
  if (i_argc > 3) {
    std::string l_arg_dtype = std::string(i_argv[3]);

    if (l_arg_dtype == "FP32") {
      l_dtype_at = at::ScalarType::Float;
      l_dtype_einsum_ir = einsum_ir::FP32;
    } else if (l_arg_dtype == "FP64") {
      l_dtype_at = at::ScalarType::Double;
      l_dtype_einsum_ir = einsum_ir::FP64;
    } else {
      l_dtype_einsum_ir = einsum_ir::UNDEFINED_DTYPE;
    }
  }

  // if (l_dtype_einsum_ir == einsum_ir::FP32) {
  //   std::cout << "dtype: FP32" << std::endl;
  // } else if (l_dtype_einsum_ir == einsum_ir::FP64) {
  //   std::cout << "dtype: FP64" << std::endl;
  // } else {
  //   std::cerr << "failed to determine dtype" << std::endl;
  //   return EXIT_FAILURE;
  // }

  /*
   * parse loop order
   */
  std::vector<int64_t> l_loop_order;
  std::vector<int64_t> *l_loop_order_ptr = nullptr;
  if (i_argc > 4) {
    std::string l_loop_string(i_argv[4]);
    einsum_ir::frontend::EinsumExpressionAscii::parse_loop_order(l_loop_string,
                                                                 m_map_dim_name_to_id,
                                                                 l_loop_order);
    std::cout << "loop order: " << l_loop_string << std::endl;
    l_loop_order_ptr = &l_loop_order;
  }

  /*
   * convert dim_sizes vector to map
   */
  std::map<int64_t, int64_t> l_dim_sizes_map;
  for (std::map<std::string, int64_t>::iterator l_di = m_map_dim_name_to_id.begin(); l_di != m_map_dim_name_to_id.end(); l_di++) {
    int64_t l_dim_id = l_di->second;
    int64_t l_dim_size = l_dim_sizes_vec[l_dim_id];
    l_dim_sizes_map.insert(std::pair<int64_t, int64_t>(l_dim_id, l_dim_size));
  }

  /*
   * Convert tensors to vector of ids
   */
  std::vector<int64_t> l_dim_ids_in_left;
  std::vector<int64_t> l_dim_ids_in_right;
  std::vector<int64_t> l_dim_ids_out;

  std::vector<std::string> l_tensor_dim_names_left;
  std::vector<std::string> l_tensor_dim_names_right;
  std::vector<std::string> l_tensor_dim_names_out;

  einsum_ir::frontend::EinsumExpressionAscii::split_string(l_tensors[0],
                                                           std::string(","),
                                                           l_tensor_dim_names_left);

  for (std::size_t l_na = 0; l_na < l_tensor_dim_names_left.size(); l_na++) {
    std::string l_dim_name = l_tensor_dim_names_left[l_na];
    int64_t l_dim_id = m_map_dim_name_to_id[l_dim_name];
    l_dim_ids_in_left.push_back(l_dim_id);
  }
  einsum_ir::frontend::EinsumExpressionAscii::split_string(l_tensors[1],
                                                           std::string(","),
                                                           l_tensor_dim_names_right);

  for (std::size_t l_na = 0; l_na < l_tensor_dim_names_right.size(); l_na++) {
    std::string l_dim_name = l_tensor_dim_names_right[l_na];
    int64_t l_dim_id = m_map_dim_name_to_id[l_dim_name];
    l_dim_ids_in_right.push_back(l_dim_id);
  }
  einsum_ir::frontend::EinsumExpressionAscii::split_string(l_tensors[2],
                                                           std::string(","),
                                                           l_tensor_dim_names_out);

  for (std::size_t l_na = 0; l_na < l_tensor_dim_names_out.size(); l_na++) {
    std::string l_dim_name = l_tensor_dim_names_out[l_na];
    int64_t l_dim_id = m_map_dim_name_to_id[l_dim_name];
    l_dim_ids_out.push_back(l_dim_id);
  }

  bench_binary(l_dim_sizes_map,
               l_dim_ids_in_left,
               l_dim_ids_in_right,
               l_dim_ids_out,
               l_loop_order_ptr,
               l_dtype_at,
               l_dtype_einsum_ir,
               l_expression_string_schar);

  // std::cout << "finished running bench_binary!" << std::endl;
  return EXIT_SUCCESS;
}