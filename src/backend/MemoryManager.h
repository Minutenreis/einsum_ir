#ifndef EINSUM_IR_BACKEND_MEMORY_MANAGER
#define EINSUM_IR_BACKEND_MEMORY_MANAGER

#include <vector>
#include <list>
#include "../constants.h"

namespace einsum_ir {
  namespace backend {
    class MemoryManager;
  }
}

class einsum_ir::backend::MemoryManager{
  public:
    //! pointer to the start of all allocated memory
    char * m_memory_ptr = nullptr;

    //! pointer to the start of aligend memory
    char * m_aligned_memory_ptr = nullptr;

    //! vector with thread specific allocated memory
    std::vector<char *> m_thread_memory;

    //! vector with thread specific aligned memory
    std::vector<char *> m_aligned_thread_memory;

    //! the required memory for all data
    int64_t m_req_mem = 0;

    //! required memory per thread
    int64_t m_req_thread_mem = 0;

    //! last id given to any tensor
    int64_t m_last_id = 0;

    // id of the current layer
    int64_t m_layer_id = 0;

    // alignment of memory to cache lines in bytes 
    int64_t m_alignment_line = 64;

    // alignment of memory to pages in bytes 
    int64_t m_alignment_page = 4096;

    //offset of the tensor for pointer calculation
    std::vector<int64_t> m_tensor_offset;

    //propertys of allocated memory
    std::list<int64_t> m_allocated_id_left;
    std::list<int64_t> m_allocated_id_right;
    std::list<int64_t> m_allocated_offset_left;
    std::list<int64_t> m_allocated_offset_right;
    

    /**
     * Destructor.
     **/
    ~MemoryManager();

    /**
     * reserves memory for a calculation. Only used in theoretical compilation of the memory manager. 
     *
     * @param i_size size of reserved memory.
     * 
     * @return id of the memory reservation.
     **/
    int64_t reserve_memory(int64_t i_size);

    /**
     * removes a memory reservation.
     *
     * @param i_id id of the memory reservation.
     **/
    void remove_reservation(int64_t i_size);

    /**
     * Allocates the required memory.
     **/
    void alloc_all_memory();

    /**
     * returns a pointer to requested memory
     *
     * @param i_id id of the memory request.
     * 
     * @return pointer to requested memory
     **/
    void * get_mem_ptr(int64_t i_id);

    /**
     * reserves thread specific memory for intermediate data in contractions. 
     *
     * @param i_size size of reserved memory.
     **/
    void reserve_thread_memory(int64_t i_size);

    /**
     * returns a pointer to thread specific memory
     *
     * @return pointer to requested memory
     **/
    void * get_thread_memory();
};

#endif
