#include "EinsumExpression.h"
#include <deque>
#include <set>
#include <cmath>
#include <string>
#include <sstream>
#ifdef _OPENMP
#include "omp.h"
#endif

void einsum_ir::frontend::EinsumExpression::histogram( int64_t         i_num_dims,
                                                       int64_t         i_string_size,
                                                       int64_t const * i_string_dim_ids,
                                                       int64_t       * o_histogram ) {
  // initialize to zero
  for( int64_t l_di = 0; l_di < i_num_dims; l_di++ ) {
    o_histogram[l_di] = 0;
  }

  // derive histogram
  for( int64_t l_en = 0; l_en < i_string_size; l_en++ ) {
    int64_t l_id = i_string_dim_ids[l_en];
    o_histogram[l_id]++;
  }
}

void einsum_ir::frontend::EinsumExpression::substring_out( int64_t                        i_num_dims_left,
                                                           int64_t                        i_num_dims_right,
                                                           int64_t                const * i_dim_ids_left,
                                                           int64_t                const * i_dim_ids_right,
                                                           int64_t                      * io_histogram,
                                                           std::vector< int64_t >       & o_substring_out ) {
  std::set< int64_t > l_substring;

  // remove left tensor's dimensions from histogram
  for( int64_t l_le = 0; l_le < i_num_dims_left; l_le++ ) {
    int64_t l_id = i_dim_ids_left[l_le];
    io_histogram[l_id]--;
  }

  // remove right tensor's dimensions from histogram
  for( int64_t l_ri = 0; l_ri < i_num_dims_right; l_ri++ ) {
    int64_t l_id = i_dim_ids_right[l_ri];
    io_histogram[l_id]--;
  }

  // add left string's contribution to output
  for( int64_t l_le = 0; l_le < i_num_dims_left; l_le++ ) {
    int64_t l_id = i_dim_ids_left[l_le];
    if( io_histogram[l_id] > 0 ) {
      l_substring.insert( l_id );
    }
  }

  // add right string's contribution to output
  for( int64_t l_ri = 0; l_ri < i_num_dims_right; l_ri++ ) {
    int64_t l_id = i_dim_ids_right[l_ri];
    if( io_histogram[l_id] > 0 ) {
      l_substring.insert( l_id );
    }
  }

  // convert to vector
  o_substring_out = std::vector< int64_t >( l_substring.begin(),
                                            l_substring.end() );

  // add output tensor's dimensions to histogram
  for( std::size_t l_en = 0; l_en < o_substring_out.size(); l_en++ ) {
    int64_t l_id = o_substring_out[l_en];
    io_histogram[l_id]++;
  }
}

void einsum_ir::frontend::EinsumExpression::unique_tensor_ids( int64_t         i_num_conts,
                                                               int64_t const * i_path,
                                                               int64_t       * o_path ) {
  int64_t l_num_tensors = i_num_conts + 1;

  std::deque< int64_t > l_tensor_ids;
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    l_tensor_ids.push_back( l_te );
  }

  for( int64_t l_co = 0; l_co < i_num_conts; l_co++ ) {
    // get tensors' ids
    int64_t l_id_0 = i_path[l_co*2 + 0];
    int64_t l_id_1 = i_path[l_co*2 + 1];

    // add contraction to unique path
    o_path[l_co*2 + 0] = l_tensor_ids[l_id_0];
    o_path[l_co*2 + 1] = l_tensor_ids[l_id_1];

    // remove tensors' ids
    l_tensor_ids.erase( l_tensor_ids.begin() + std::max( l_id_0, l_id_1 ) );
    l_tensor_ids.erase( l_tensor_ids.begin() + std::min( l_id_0, l_id_1 ) );

    // add id of contraction output
    l_tensor_ids.push_back( l_num_tensors );
    l_num_tensors++;
  }
}

void einsum_ir::frontend::EinsumExpression::init( int64_t                 i_num_dims,
                                                  int64_t const         * i_dim_sizes,
                                                  int64_t                 i_num_conts,
                                                  int64_t const         * i_string_num_dims,
                                                  int64_t const         * i_string_dim_ids,
                                                  int64_t const         * i_path,
                                                  complex_t               i_ctype_ext,
                                                  data_t                  i_dtype,
                                                  void          * const * i_data_ptrs ) {
  m_num_dims = i_num_dims;
  m_dim_sizes = i_dim_sizes;
  m_num_conts = i_num_conts;
  m_string_num_dims_ext = i_string_num_dims;
  m_string_dim_ids_ext = i_string_dim_ids;
  m_path_ext = i_path;
  m_ctype_ext = i_ctype_ext;
  m_dtype = i_dtype;
  m_data_ptrs = i_data_ptrs;
  m_compiled = false;
}

void einsum_ir::frontend::EinsumExpression::init( int64_t                 i_num_dims,
                                                  int64_t const         * i_dim_sizes,
                                                  int64_t                 i_num_conts,
                                                  int64_t const         * i_string_num_dims,
                                                  int64_t const         * i_string_dim_ids,
                                                  int64_t const         * i_path,
                                                  data_t                  i_dtype,
                                                  void          * const * i_data_ptrs ) {
  init( i_num_dims,
        i_dim_sizes,
        i_num_conts,
        i_string_num_dims,
        i_string_dim_ids,
        i_path,
        complex_t::REAL_ONLY,
        i_dtype,
        i_data_ptrs );
}

einsum_ir::err_t einsum_ir::frontend::EinsumExpression::compile() {
  // derive contraction path using unqiue tensor ids
  m_path_int.resize( m_num_conts*2 );
  unique_tensor_ids( m_num_conts,
                     m_path_ext,
                     m_path_int.data() );

  // assemble dim id to sizes map
  for( int64_t l_di = 0; l_di < m_num_dims; l_di++ ) {
    m_map_dim_sizes.insert( {l_di, m_dim_sizes[l_di]} );
  }

  // number of input tensors
  int64_t l_num_tensors_in = m_num_conts + 1;
  // total number of tensors
  int64_t l_num_tensors    = l_num_tensors_in + 1;

  // derive sizes of string associated with input tensors and total
  int64_t l_string_size_in = 0;
  for( int64_t l_te = 0; l_te < l_num_tensors_in; l_te++ ) {
    l_string_size_in += m_string_num_dims_ext[l_te];
  }
  int64_t l_string_size = l_string_size_in + m_string_num_dims_ext[l_num_tensors-1];

  // initialize internal data structures with info of input tensors
  m_string_dim_ids_int = std::vector< int64_t >( m_string_dim_ids_ext,
                                                 m_string_dim_ids_ext + l_string_size_in );

  m_string_num_dims_int = std::vector< int64_t >( m_string_num_dims_ext,
                                                  m_string_num_dims_ext + l_num_tensors_in );

  // derive offsets w.r.t. input tensors
  std::vector< int64_t > l_string_offsets( l_num_tensors );
  l_string_offsets[0] = 0;
  for( int64_t l_te = 1; l_te < l_num_tensors; l_te++ ) {
    l_string_offsets[l_te] = l_string_offsets[l_te-1] + m_string_num_dims_int[l_te-1];
  }

  // derive histogram
  std::vector< int64_t > l_hist( m_num_dims );
  histogram( m_num_dims,
             l_string_size,
             m_string_dim_ids_ext,
             l_hist.data() );

  // assemble info for internal nodes
  for( int64_t l_co = 0; l_co < m_num_conts-1; l_co++ ) {
    int64_t l_id_left  = m_path_int[l_co*2 + 0];
    int64_t l_id_right = m_path_int[l_co*2 + 1];

    int64_t l_num_dims_left  = m_string_num_dims_int[l_id_left];
    int64_t l_num_dims_right = m_string_num_dims_int[l_id_right];

    int64_t const * l_dim_ids_left  = m_string_dim_ids_int.data() + l_string_offsets[l_id_left];
    int64_t const * l_dim_ids_right = m_string_dim_ids_int.data() + l_string_offsets[l_id_right];

    std::vector< int64_t > l_substring_out;
    substring_out( l_num_dims_left,
                   l_num_dims_right,
                   l_dim_ids_left,
                   l_dim_ids_right,
                   l_hist.data(),
                   l_substring_out );

    // append substring data
    m_string_dim_ids_int.insert( m_string_dim_ids_int.end(),
                                 l_substring_out.begin(),
                                 l_substring_out.end() );
    m_string_num_dims_int.push_back( l_substring_out.size() );
    l_string_offsets.push_back( m_string_dim_ids_int.size() );
  }

  // append data for batch-outer contraction if required
  int64_t const * l_dim_ids_ext_root = m_string_dim_ids_ext + l_string_offsets[l_num_tensors-1];

  if( m_ctype_ext == complex_t::BATCH_INNER ) {
    m_string_num_dims_int.push_back( m_string_num_dims_ext[l_num_tensors-1] );
    m_string_dim_ids_int.push_back( l_dim_ids_ext_root[ m_string_num_dims_int.back()-1 ] ); // complex batch inner -> outermost dimension
    for( int64_t l_di = 0; l_di < m_string_num_dims_int.back()-1; l_di++ ) {
      m_string_dim_ids_int.push_back( l_dim_ids_ext_root[l_di] );
    }
    m_string_num_dims_int.push_back( m_string_num_dims_int.back() );
    l_string_offsets.push_back( m_string_dim_ids_int.size() );
  }

  // add root data
  m_string_num_dims_int.push_back( m_string_num_dims_ext[l_num_tensors-1] );
  m_string_dim_ids_int.insert( m_string_dim_ids_int.end(),
                               l_dim_ids_ext_root,
                               l_dim_ids_ext_root + m_string_num_dims_int.back() );
  l_string_offsets.push_back( m_string_dim_ids_int.size() );

  /*
   * add nodes
   */
  int64_t l_num_nodes = l_num_tensors_in + m_num_conts;
  // batch-outer to batch-inner conversion
  if( m_ctype_ext == complex_t::BATCH_INNER ) {
    l_num_nodes++;
  }
  m_nodes.resize( l_num_nodes );

  // add input nodes
  for( int64_t l_te = 0; l_te < l_num_tensors_in; l_te++ ) {
    int64_t l_num_dims = m_string_num_dims_int[l_te];

    m_nodes[l_te].init( l_num_dims,
                        m_string_dim_ids_int.data() + l_string_offsets[l_te],
                        &m_map_dim_sizes,
                        nullptr,
                        m_dtype,
                        m_data_ptrs[l_te],
                        &m_memory );
  }

  // derive kernel types
  kernel_t l_ktype_first_touch = (m_ctype_ext == complex_t::REAL_ONLY) ? einsum_ir::ZERO : einsum_ir::CPX_ZERO;
  kernel_t l_ktype_main        = (m_ctype_ext == complex_t::REAL_ONLY) ? einsum_ir::MADD : einsum_ir::CPX_MADD;

  // add internal nodes
  for( int64_t l_co = 0; l_co < m_num_conts-1; l_co++ ) {
    int64_t l_id_left  = m_path_int[l_co*2 + 0];
    int64_t l_id_right = m_path_int[l_co*2 + 1];
    int64_t l_id_out = l_num_tensors_in + l_co;

    int64_t l_num_dims = m_string_num_dims_int[l_id_out];
    int64_t * l_dim_ids_out = m_string_dim_ids_int.data() + l_string_offsets[l_id_out];
  
    m_nodes[l_num_tensors_in+l_co].init( l_num_dims,
                                         l_dim_ids_out,
                                         &m_map_dim_sizes,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         m_dtype,
                                         nullptr,
                                         nullptr,
                                         l_ktype_first_touch,
                                         l_ktype_main,
                                         kernel_t::UNDEFINED_KTYPE,
                                         &m_nodes[l_id_left],
                                         &m_nodes[l_id_right],
                                         &m_memory );
  }

  // add root contraction
  int64_t   l_root_id_left  = m_path_int[ (m_num_conts-1)*2 + 0];
  int64_t   l_root_id_right = m_path_int[ (m_num_conts-1)*2 + 1];

  int64_t   l_root_num_dims = m_string_num_dims_int[l_num_tensors_in + m_num_conts - 1];
  int64_t * l_root_dim_ids_out = m_string_dim_ids_int.data() + l_string_offsets[l_num_tensors_in + m_num_conts - 1];

  m_nodes[l_num_tensors_in + m_num_conts - 1].init( l_root_num_dims,
                                                    l_root_dim_ids_out,
                                                    &m_map_dim_sizes,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    m_dtype,
                                                    nullptr,
                                                    (m_ctype_ext != complex_t::BATCH_INNER) ? m_data_ptrs[l_num_tensors-1] : nullptr,
                                                    l_ktype_first_touch,
                                                    l_ktype_main,
                                                    kernel_t::UNDEFINED_KTYPE,
                                                    &m_nodes[l_root_id_left],
                                                    &m_nodes[l_root_id_right],
                                                    &m_memory );

  // add batch-outer to batch-inner conversion
  if( m_ctype_ext == complex_t::BATCH_INNER ) {
    int64_t   l_cpx_conv_child    = l_num_tensors_in + m_num_conts - 1;
    int64_t   l_cpx_conv_num_dims = m_string_num_dims_int[l_num_tensors_in + m_num_conts];
    int64_t * l_cpx_conv_dim_ids  = m_string_dim_ids_int.data() + l_string_offsets[l_num_tensors_in + m_num_conts];

    m_nodes.back().init( l_cpx_conv_num_dims,
                         l_cpx_conv_dim_ids,
                         &m_map_dim_sizes,
                         nullptr,
                         m_dtype,
                         m_data_ptrs[l_num_tensors-1],
                         &m_nodes[l_cpx_conv_child],
                         &m_memory );
  }

  err_t l_err = m_nodes.back().compile();

  /*
   * init intra-op parallelism
   */
#ifdef _OPENMP
  // four times overload
  int64_t l_num_tasks = omp_get_max_threads() * 4;

  for( std::size_t l_no = 0; l_no < m_nodes.size(); l_no++ ) {
    // magic number: 64^3
    if(    m_nodes[l_no].m_num_ops_node == 0
        || m_nodes[l_no].m_num_ops_node >= 262144 ) {
      m_nodes[l_no].threading_intra_op( l_num_tasks );
    }
  }
#endif

  m_compiled = true;

  return l_err;
}

einsum_ir::err_t einsum_ir::frontend::EinsumExpression::store_and_lock_data( int64_t i_tensor_id ) {
  if( m_compiled == false ) {
    return err_t::CALLED_BEFORE_COMPILATION;
  }
  else if( !(i_tensor_id < m_num_conts+1) ) {
    return err_t::INVALID_ID;
  }

  err_t l_err = m_nodes[i_tensor_id].store_and_lock_data();

  return l_err;
}

einsum_ir::err_t einsum_ir::frontend::EinsumExpression::unlock_data( int64_t i_tensor_id ) {
  if( m_compiled == false ) {
    return err_t::CALLED_BEFORE_COMPILATION;
  }
  else if( !(i_tensor_id < m_num_conts+1) ) {
    return err_t::INVALID_ID;
  }

  err_t l_err = m_nodes[i_tensor_id].unlock_data();

  return l_err;
}

void einsum_ir::frontend::EinsumExpression::eval() {
  m_nodes.back().eval();
}

int64_t einsum_ir::frontend::EinsumExpression::num_ops() {
  if( m_nodes.size() > 0 ) {
    return m_nodes.back().num_ops( true );
  }
  else {
    return 0;
  }
}

std::string einsum_ir::frontend::EinsumExpression::to_string_render() const {
  if( m_compiled == false ) {
    return "Error: Expression not compiled.";
  }

  std::vector< backend::EinsumNode const * > l_nodes;
  std::vector< int64_t > l_pos;
  std::vector< int64_t > l_offset_lvl;

  l_nodes.push_back( &m_nodes.back() );
  l_pos.push_back( 0 );
  l_offset_lvl.push_back( 0 );
  l_offset_lvl.push_back( 1 );

  int64_t l_lvl = 0;
  while( true ) {
    // extract nodes and positions of next level
    for( int64_t l_no = l_offset_lvl[l_lvl]; l_no < l_offset_lvl[l_lvl+1]; l_no++ ) {
      backend::EinsumNode const * l_node = l_nodes[l_no];

      // add children
      for( std::size_t l_ch = 0; l_ch < l_node->m_children.size(); l_ch++ ) {
        l_nodes.push_back( l_node->m_children[l_ch] );
        l_pos.push_back( 2*l_pos[l_no] + l_ch );
      }
    }

    // update level offset
    if( (int64_t) l_nodes.size() > l_offset_lvl.back() ) {
      l_offset_lvl.push_back( l_nodes.size() );
    }
    else {
      break;
    }
    l_lvl++;
  }
  int64_t l_num_lvls = l_lvl + 1;

  // create a string representation of each node
  std::vector< std::string > l_nodes_str( l_nodes.size() );
  for( std::size_t l_no = 0; l_no < l_nodes.size(); l_no++ ) {
    std::string l_dims = "";
    for( int64_t l_di = 0; l_di < l_nodes[l_no]->m_num_dims; l_di++ ) {
      l_dims += std::to_string( l_nodes[l_no]->m_dim_ids_int[l_di] );
      if( l_di < l_nodes[l_no]->m_num_dims-1 ) {
        l_dims += " ";
      }
    }
    l_nodes_str[l_no] = l_dims;
  }

  // determine max size of string representations
  int64_t l_max_size = 0;
  for( std::size_t l_no = 0; l_no < l_nodes.size(); l_no++ ) {
    if( (int64_t) l_nodes_str[l_no].size() > l_max_size ) {
      l_max_size = l_nodes_str[l_no].size();
    }
  }

  // pad string representations
  for( std::size_t l_no = 0; l_no < l_nodes.size(); l_no++ ) {
    while( (int64_t) l_nodes_str[l_no].size() < l_max_size ) {
      l_nodes_str[l_no] += " ";
    }
  }

  // scale positions to obtain correct spacing between nodes
  // [...]
  // lvl 3: 2^3 = 8
  // lvl 2: 2^2 = 4
  // lvl 0: 2^1 = 2
  l_lvl = 0;
  while( l_lvl < l_num_lvls ) {
    int64_t l_scale = std::pow( 2, l_num_lvls-l_lvl );
    for( int64_t l_no = l_offset_lvl[l_lvl]; l_no < l_offset_lvl[l_lvl+1]; l_no++ ) {
      l_pos[l_no] *= l_scale;
    }
    l_lvl++;
  }

  // adjust positions to account for initial white space
  // [...]
  // lvl 2: 2^2 - 1 = 3
  // lvl 1: 2^1 - 1 = 1
  // lvl 0: 2^0 - 1 = 0
  l_lvl = 0;
  while( l_lvl < l_num_lvls ) {
    int64_t l_num_ws = std::pow( 2, l_num_lvls-l_lvl-1 ) - 1;
    for( int64_t l_no = l_offset_lvl[l_lvl]; l_no < l_offset_lvl[l_lvl+1]; l_no++ ) {
      l_pos[l_no] += l_num_ws;
    }
    l_lvl++;
  }

  // create whitespaces matching the max dims sizes
  std::string l_ws = std::string( l_max_size, ' ' );

  // assemble output string
  std::stringstream l_result;

  l_lvl = 0;
  while( l_lvl < l_num_lvls ) {
    int64_t l_po = 0;

    for( int64_t l_no = l_offset_lvl[l_lvl]; l_no < l_offset_lvl[l_lvl+1]; l_no++ ) {
      while( l_po < l_pos[l_no] ) {
        l_result << l_ws;
        l_po++;
      }

      l_result << l_nodes_str[l_no];
      l_po++;
    }
    if( l_lvl < l_num_lvls-1 ) {
      l_result << std::endl;
    }
    l_lvl++;
  }

  return l_result.str();
}

std::string to_string_exchange_format() {
  return "";
}

std::string einsum_ir::frontend::EinsumExpression::to_string_exchange_format( backend::EinsumNode const * i_node ) const {
  if( m_compiled == false ) {
    return "Error: Expression not compiled.";
  }

  std::string l_str = "";

  backend::EinsumNode const * l_node = i_node;
  if( l_node == nullptr ) {
    l_node = &m_nodes.back();
  }

  bool l_leaf = true;

  if(    l_node->m_children.size() == 0
      && l_node->m_dim_ids_ext != nullptr ) {

    for( int64_t l_di = 0; l_di < l_node->m_num_dims; l_di++ ) {
      if( l_node->m_dim_ids_int[l_di] != l_node->m_dim_ids_ext[l_di] ) {
        l_leaf = false;
        break;
      }
    }

    if( l_leaf == false ) {
      l_str += "[";
      for( int64_t l_di = 0; l_di < l_node->m_num_dims; l_di++ ) {
        l_str += std::to_string( l_node->m_dim_ids_ext[l_di] );
        if( l_di < l_node->m_num_dims-1 ) {
          l_str += ",";
        }
      }
      l_str += "]->";
    }
  }

  if( l_node->m_children.size() == 2 ) {
    l_leaf = false;
    l_str += "[";
    l_str += to_string_exchange_format( l_node->m_children[0] );
    l_str += "],[";
    l_str += to_string_exchange_format( l_node->m_children[1] );
    l_str += "]->";
  }

  if( l_leaf == false ) {
    l_str += "[";
  }
  for( int64_t l_di = 0; l_di < l_node->m_num_dims; l_di++ ) {
    l_str += std::to_string( l_node->m_dim_ids_int[l_di] );
    if( l_di < l_node->m_num_dims-1 ) {
      l_str += ",";
    }
  }
  if( l_leaf == false ) {
    l_str += "]";
  }

  return l_str;
}
