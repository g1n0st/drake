#ifndef BELLLOCH_SCAN_H
#define BELLLOCH_SCAN_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

__global__
void gpu_add_block_sums(unsigned int* const d_out,
    const unsigned int* const d_in,
    unsigned int* const d_block_sums,
    const size_t numElems);


__global__
void gpu_prescan(unsigned int* const d_out,
    const unsigned int* const d_in,
    unsigned int* const d_block_sums,
    const unsigned int len,
    const unsigned int shmem_sz,
    const unsigned int max_elems_per_block);
 
void sum_scan_blelloch(unsigned int* const d_out,
    const unsigned int* const d_in,
    const size_t numElems,
    unsigned int* const d_tmp);

#endif