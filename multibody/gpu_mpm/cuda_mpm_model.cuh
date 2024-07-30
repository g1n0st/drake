#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <assert.h>

#include "multibody/gpu_mpm/settings.h"

namespace drake {
namespace multibody {
namespace gmpm {

template <typename T>
struct GpuMpmState {

public:
    GpuMpmState() = default;

    const size_t& n_verts() const { return n_verts_; }
    const size_t& n_faces() const { return n_faces_; }
    const size_t& n_particles() const { return n_particles_; }
    const uint32_t& current_particle_buffer_id() const { return current_particle_buffer_id_; }

    T* current_positions() { return particle_buffer_[current_particle_buffer_id_].d_positions; }
    T* current_velocities() { return particle_buffer_[current_particle_buffer_id_].d_velocities; }
    T* current_volumes() { return particle_buffer_[current_particle_buffer_id_].d_volumes; }
    T* current_affine_matrices() { return particle_buffer_[current_particle_buffer_id_].d_affine_matrices; }

    T* forces() { return d_forces_; }
    T* taus() { return d_taus_; }
    T* deformation_gradients() { return d_deformation_gradients_; }
    T* Dm_inverses() { return d_Dm_inverses_; }
    int* indices() { return d_indices_; }

    T* next_positions() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_positions; }
    T* next_velocities() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_velocities; }
    T* next_volumes() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_volumes; }
    T* next_affine_matrices() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_affine_matrices; }
    
    uint32_t* current_sort_keys() { return particle_buffer_[current_particle_buffer_id_].d_sort_keys; }
    uint32_t* current_sort_ids() { return particle_buffer_[current_particle_buffer_id_].d_sort_ids; }
    uint32_t* next_sort_keys() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_sort_keys; }
    uint32_t* next_sort_ids() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_sort_ids; }

    T* grid_masses() { return grid_buffer_.d_g_masses; }
    T* grid_momentum() { return grid_buffer_.d_g_momentum; }
    uint32_t* grid_touched_flags() { return grid_buffer_.d_g_touched_flags; }
    uint32_t* grid_touched_ids() { return grid_buffer_.d_g_touched_ids; }
    uint32_t* grid_touched_cnt() { return grid_buffer_.d_g_touched_cnt; }
    uint32_t grid_touched_cnt_host() {
        uint32_t h_g_touched_cnt = 0u;
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        CUDA_SAFE_CALL(cudaMemcpy(&h_g_touched_cnt, this->grid_touched_cnt(), sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return h_g_touched_cnt;
    }

    unsigned int* sort_buffer() { return sort_buffer_; };
    size_t& sort_buffer_size() { return sort_buffer_size_; }

    // NOTE (changyu): initialize GPU MPM state, all gpu memory allocation should be done here to avoid re-allocation.
    void InitializeQRCloth(const std::vector<Vec3<T>> &pos, 
                           const std::vector<Vec3<T>> &vel,
                           const std::vector<int> &indices);

    // NOTE (changyu): free GPU MPM state, all gpu memory free should be done here.
    void Destroy();

    void SwitchCurrentState() { current_particle_buffer_id_ ^= 1; }

private:

    // Particles state device ptrs
    size_t n_verts_;
    size_t n_faces_;
    size_t n_particles_;

    // scratch data
    T* d_forces_ = nullptr;      // size: n_faces + n_verts, NO sort
    T* d_taus_ = nullptr;        // size: n_faces + n_verts, NO sort

    // element-based data
    int* d_indices_ = nullptr; // size: n_faces NO sort
    T* d_deformation_gradients_ = nullptr; // size: n_faces NO sort
    T* d_Dm_inverses_ = nullptr; // size: n_faces NO sort
    
    struct ParticleBuffer {
        T* d_positions = nullptr;   // size: n_faces + n_verts
        T* d_velocities = nullptr;  // size: n_faces + n_verts
        T* d_volumes = nullptr;     // size: n_faces + n_verts
        T* d_affine_matrices = nullptr; // size: n_faces + n_verts

        uint32_t* d_sort_keys = nullptr;
        uint32_t* d_sort_ids = nullptr;
    };
    
    uint32_t current_particle_buffer_id_ = 0;
    std::array<ParticleBuffer, 2> particle_buffer_;

    size_t sort_buffer_size_ = 0;
    unsigned int* sort_buffer_ = nullptr;

    // Particles state host ptrs
    // TODO(changyu): Host memory should be managed by Drake context instead of here.
    std::vector<Vec3<T>> h_positions_;
    std::vector<Vec3<T>> h_velocities_;
    std::vector<int> h_indices_;

    // Grid state device ptrs

    struct GridBuffer {
        T* d_g_masses = nullptr;
        T* d_g_momentum = nullptr;
        uint32_t* d_g_touched_flags = nullptr;
        uint32_t* d_g_touched_ids   = nullptr;
        uint32_t* d_g_touched_cnt   = nullptr;
    };

    GridBuffer grid_buffer_;
};

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake