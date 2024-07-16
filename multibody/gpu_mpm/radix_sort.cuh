#ifndef RADIX_SORT_H
#define RADIX_SORT_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "multibody/gpu_mpm/settings.h"
#include "multibody/gpu_mpm/blelloch_scan.cuh"

#define MAX_BLOCK_SZ 128

template<class keyT, class valueT>
__global__ void gpu_radix_sort_local(keyT* d_key_out_sorted,
                                     valueT* d_value_out_sorted,
    unsigned int* d_prefix_sums,
    unsigned int* d_block_sums,
    unsigned int input_shift_width,
    keyT* d_key_in,
    valueT* d_value_in,
    unsigned int d_key_in_len,
    unsigned int max_elems_per_block) {
    // need shared memory array for:
    // - block's share of the input data (local sort will be put here too)
    // - mask outputs
    // - scanned mask outputs
    // - merged scaned mask outputs ("local prefix sum")
    // - local sums of scanned mask outputs
    // - scanned local sums of scanned mask outputs

    // for all radix combinations:
    //  build mask output for current radix combination
    //  scan mask ouput
    //  store needed value from current prefix sum array to merged prefix sum array
    //  store total sum of mask output (obtained from scan) to global block sum array
    // calculate local sorted address from local prefix sum and scanned mask output's total sums
    // shuffle input block according to calculated local sorted addresses
    // shuffle local prefix sums according to calculated local sorted addresses
    // copy locally sorted array back to global memory
    // copy local prefix sum array back to global memory

    extern __shared__ unsigned int shmem[];
    // s_mask_out[] will be scanned in place
    unsigned int s_mask_out_len = max_elems_per_block + 1;
    unsigned int* s_mask_out = static_cast<unsigned int*>(shmem);
    unsigned int* s_merged_scan_mask_out = &s_mask_out[s_mask_out_len];
    unsigned int* s_mask_out_sums = &s_merged_scan_mask_out[max_elems_per_block];
    unsigned int* s_scan_mask_out_sums = &s_mask_out_sums[4];
    
    __shared__ keyT s_data[MAX_BLOCK_SZ];
    __shared__ valueT s_v_data[MAX_BLOCK_SZ];

    unsigned int thid = threadIdx.x;

    // Copy block's portion of global input data to shared memory
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;
    if (cpy_idx < d_key_in_len) {
        s_data[thid] = d_key_in[cpy_idx];
        s_v_data[thid] = d_value_in[cpy_idx];
    }
    else {
        s_data[thid] = 0;
    }

    __syncthreads();

    // To extract the correct 2 bits, we first shift the number
    //  to the right until the correct 2 bits are in the 2 LSBs,
    //  then mask on the number with 11 (3) to remove the bits
    //  on the left
    keyT t_data = s_data[thid];
    valueT t_v_data = s_v_data[thid];
    unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;

    for (unsigned int i = 0; i < 4; ++i) {
        // Zero out s_mask_out
        s_mask_out[thid] = 0;
        if (thid == 0)
            s_mask_out[s_mask_out_len - 1] = 0;

        __syncthreads();

        // build bit mask output
        bool val_equals_i = false;
        if (cpy_idx < d_key_in_len) {
            val_equals_i = t_2bit_extract == i;
            s_mask_out[thid] = val_equals_i;
        }
        __syncthreads();

        // Scan mask outputs (Hillis-Steele)
        int partner = 0;
        unsigned int sum = 0;
        unsigned int max_steps = (unsigned int) log2f(max_elems_per_block);
        for (unsigned int d = 0; d < max_steps; d++) {
            partner = thid - (1 << d);
            if (partner >= 0) {
                sum = s_mask_out[thid] + s_mask_out[partner];
            }
            else {
                sum = s_mask_out[thid];
            }
            __syncthreads();
            s_mask_out[thid] = sum;
            __syncthreads();
        }

        // Shift elements to produce the same effect as exclusive scan
        unsigned int cpy_val = 0;
        cpy_val = s_mask_out[thid];
        __syncthreads();
        s_mask_out[thid + 1] = cpy_val;
        __syncthreads();

        if (thid == 0) {
            // Zero out first element to produce the same effect as exclusive scan
            s_mask_out[0] = 0;
            unsigned int total_sum = s_mask_out[s_mask_out_len - 1];
            s_mask_out_sums[i] = total_sum;
            d_block_sums[i * gridDim.x + blockIdx.x] = total_sum;
        }
        __syncthreads();

        if (val_equals_i && (cpy_idx < d_key_in_len)) {
            s_merged_scan_mask_out[thid] = s_mask_out[thid];
        }

        __syncthreads();
    }

    // Scan mask output sums
    // Just do a naive scan since the array is really small
    if (thid == 0) {
        unsigned int run_sum = 0;
        for (unsigned int i = 0; i < 4; ++i) {
            s_scan_mask_out_sums[i] = run_sum;
            run_sum += s_mask_out_sums[i];
        }
    }

    __syncthreads();

    if (cpy_idx < d_key_in_len) {
        // Calculate the new indices of the input elements for sorting
        unsigned int t_prefix_sum = s_merged_scan_mask_out[thid];
        unsigned int new_pos = t_prefix_sum + s_scan_mask_out_sums[t_2bit_extract];
        
        __syncthreads();

        // Shuffle the block's input elements to actually sort them
        // Do this step for greater global memory transfer coalescing
        //  in next step
        s_data[new_pos] = t_data;
        s_v_data[new_pos] = t_v_data;
        s_merged_scan_mask_out[new_pos] = t_prefix_sum;
        
        __syncthreads();

        // Copy block - wise prefix sum results to global memory
        // Copy block-wise sort results to global 
        d_prefix_sums[cpy_idx] = s_merged_scan_mask_out[thid];

        d_key_out_sorted[cpy_idx] = s_data[thid];
        d_value_out_sorted[cpy_idx] = s_v_data[thid];
    }
}

template<class keyT, class valueT>
__global__ void gpu_glbl_shuffle(keyT* d_key_out,
    keyT* d_key_in,
    valueT* d_value_out,
    valueT* d_value_in,
    unsigned int* d_scan_block_sums,
    unsigned int* d_prefix_sums,
    unsigned int input_shift_width,
    unsigned int d_key_in_len,
    unsigned int max_elems_per_block) {
    // get d = digit
    // get n = blockIdx
    // get m = local prefix sum array value
    // calculate global position = P_d[n] + m
    // copy input element to final position in d_key_out

    unsigned int thid = threadIdx.x;
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;

    if (cpy_idx < d_key_in_len) {
        keyT t_data = d_key_in[cpy_idx];
        valueT t_v_data = d_value_in[cpy_idx];
        unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;
        unsigned int t_prefix_sum = d_prefix_sums[cpy_idx];
        unsigned int data_glbl_pos = d_scan_block_sums[t_2bit_extract * gridDim.x + blockIdx.x]
            + t_prefix_sum;
        __syncthreads();
        d_key_out[data_glbl_pos] = t_data;
        d_value_out[data_glbl_pos] = t_v_data;
    }
}

// An attempt at the gpu radix sort variant described in this paper:
// https://vgc.poly.edu/~csilva/papers/cgf.pdf
template<class keyT, class valueT, int bitNum = sizeof(keyT) * 8>
void radix_sort(
    keyT* const d_key_out,
    keyT* const d_key_in,
    valueT* const d_value_out,
    valueT* const d_value_in,
    unsigned int* tmp_buffer,
    size_t &tmp_buffer_size,
    unsigned int d_key_in_len) {
    unsigned int block_sz = MAX_BLOCK_SZ;
    unsigned int max_elems_per_block = block_sz;
    unsigned int grid_sz = d_key_in_len / max_elems_per_block;
    // Take advantage of the fact that integer division drops the decimals
    if (d_key_in_len % max_elems_per_block != 0)
        grid_sz += 1;

    unsigned int d_prefix_sums_len = d_key_in_len;
    unsigned int d_block_sums_len = 4 * grid_sz; // 4-way split
    size_t _tmp_buffer_size = (d_prefix_sums_len + d_block_sums_len * 2) + (d_prefix_sums_len * 4);

    unsigned int* d_prefix_sums = static_cast<unsigned int*>(tmp_buffer);
    unsigned int* d_block_sums = &d_prefix_sums[d_prefix_sums_len];
    unsigned int* d_scan_block_sums = &d_block_sums[d_block_sums_len];
    unsigned int* d_scan_tmp_buffer = &d_scan_block_sums[d_block_sums_len];

    if (tmp_buffer_size == 0) {
        tmp_buffer_size = _tmp_buffer_size;
        return;
    } else {
        assert(tmp_buffer_size >= _tmp_buffer_size);
        CUDA_SAFE_CALL(cudaMemset(d_prefix_sums, 0, sizeof(unsigned int) * d_prefix_sums_len));
        CUDA_SAFE_CALL(cudaMemset(d_block_sums, 0, sizeof(unsigned int) * d_block_sums_len));
        CUDA_SAFE_CALL(cudaMemset(d_scan_block_sums, 0, sizeof(unsigned int) * d_block_sums_len));
    }

    // shared memory consists of 3 arrays the size of the block-wise input
    //  and 2 arrays the size of n in the current n-way split (4)
    unsigned int s_mask_out_len = max_elems_per_block + 1;
    unsigned int s_merged_scan_mask_out_len = max_elems_per_block;
    unsigned int s_mask_out_sums_len = 4; // 4-way split
    unsigned int s_scan_mask_out_sums_len = 4;
    unsigned int shmem_sz = (s_mask_out_len
                            + s_merged_scan_mask_out_len
                            + s_mask_out_sums_len
                            + s_scan_mask_out_sums_len)
                            * sizeof(unsigned int);

    // for every 2 bits from LSB to MSB:
    //  block-wise radix sort (write blocks back to global memory)
    for (unsigned int shift_width = 0; shift_width <= bitNum - 2; shift_width += 2) {
        gpu_radix_sort_local<<<grid_sz, block_sz, shmem_sz>>>(d_key_out, 
                                                              d_value_out,
                                                                d_prefix_sums, 
                                                                d_block_sums, 
                                                                shift_width, 
                                                                d_key_in, 
                                                                d_value_in,
                                                                d_key_in_len, 
                                                                max_elems_per_block);

        //unsigned int* h_test = new unsigned int[d_key_in_len];
        //CUDA_SAFE_CALL(cudaMemcpy(h_test, d_key_in, sizeof(unsigned int) * d_key_in_len, cudaMemcpyDeviceToHost));
        //for (unsigned int i = 0; i < d_key_in_len; ++i)
        //    std::cout << h_test[i] << " ";
        //std::cout << std::endl;
        //delete[] h_test;

        // scan global block sum array
        sum_scan_blelloch(d_scan_block_sums, d_block_sums, d_block_sums_len, d_scan_tmp_buffer);

        // scatter/shuffle block-wise sorted array to final positions
        gpu_glbl_shuffle<<<grid_sz, block_sz>>>(d_key_in, 
                                                    d_key_out, 
                                                    d_value_in,
                                                    d_value_out,
                                                    d_scan_block_sums, 
                                                    d_prefix_sums, 
                                                    shift_width, 
                                                    d_key_in_len, 
                                                    max_elems_per_block);
    }
    CUDA_SAFE_CALL(cudaMemcpy(d_key_out, d_key_in, sizeof(keyT) * d_key_in_len, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_value_out, d_value_in, sizeof(valueT) * d_key_in_len, cudaMemcpyDeviceToDevice));
}

#endif