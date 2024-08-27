#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <tuple>
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
    const T* current_positions() const { return particle_buffer_[current_particle_buffer_id_].d_positions; }
    T* current_velocities() { return particle_buffer_[current_particle_buffer_id_].d_velocities; }
    const T* current_velocities() const { return particle_buffer_[current_particle_buffer_id_].d_velocities; }
    T* current_volumes() { return particle_buffer_[current_particle_buffer_id_].d_volumes; }
    const T* current_volumes() const { return particle_buffer_[current_particle_buffer_id_].d_volumes; }
    T* current_affine_matrices() { return particle_buffer_[current_particle_buffer_id_].d_affine_matrices; }
    const T* current_affine_matrices() const { return particle_buffer_[current_particle_buffer_id_].d_affine_matrices; }

    int* current_pids() { return particle_buffer_[current_particle_buffer_id_].d_pids; }
    const int* current_pids() const { return particle_buffer_[current_particle_buffer_id_].d_pids; }
    uint32_t* current_sort_keys() { return particle_buffer_[current_particle_buffer_id_].d_sort_keys; }
    const uint32_t* current_sort_keys() const { return particle_buffer_[current_particle_buffer_id_].d_sort_keys; }
    uint32_t* current_sort_ids() { return particle_buffer_[current_particle_buffer_id_].d_sort_ids; }
    const uint32_t* current_sort_ids() const { return particle_buffer_[current_particle_buffer_id_].d_sort_ids; }

    T* next_positions() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_positions; }
    T* next_velocities() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_velocities; }
    T* next_volumes() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_volumes; }
    T* next_affine_matrices() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_affine_matrices; }
    int* next_pids() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_pids; }
    uint32_t* next_sort_keys() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_sort_keys; }
    uint32_t* next_sort_ids() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_sort_ids; }

    T* backup_positions() { return particle_buffer_[backup_buffer_id].d_positions; }
    T* backup_velocities() { return particle_buffer_[backup_buffer_id].d_velocities; }
    T* backup_volumes() { return particle_buffer_[backup_buffer_id].d_volumes; }
    T* backup_affine_matrices() { return particle_buffer_[backup_buffer_id].d_affine_matrices; }
    int* backup_pids() { return particle_buffer_[backup_buffer_id].d_pids; }
    uint32_t* backup_sort_keys() { return particle_buffer_[backup_buffer_id].d_sort_keys; }
    uint32_t* backup_sort_ids() { return particle_buffer_[backup_buffer_id].d_sort_ids; }

    // NOTE (changyu): next state data is only meaningful at BuildReturnMapping stage,
    // we can reuse these buffers to splat particle contact dv to the grid.
    T* contact_positions() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_positions; }
    T* contact_velocities() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_velocities; }
    T* contact_volumes() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_volumes; }
    T* contact_affine_matrices() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_affine_matrices; }
    uint32_t* contact_sort_keys() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_sort_keys; }
    uint32_t* contact_sort_ids() { return particle_buffer_[current_particle_buffer_id_ ^ 1].d_sort_ids; }

    T* forces() { return d_forces_; }
    const T* forces() const { return d_forces_; }
    T* taus() { return d_taus_; }
    const T* taus() const { return d_taus_; }
    T* deformation_gradients() { return d_deformation_gradients_; }
    const T* deformation_gradients() const { return d_deformation_gradients_; }
    T* backup_deformation_gradients() { return d_backup_deformation_gradients_; }
    T* Dm_inverses() { return d_Dm_inverses_; }
    const T* Dm_inverses() const { return d_Dm_inverses_; }
    int* indices() { return d_indices_; }
    const int* indices() const { return d_indices_; }
    int* index_mappings() { return d_index_mappings_; }
    const int* index_mappings() const { return d_index_mappings_; }

    T* grid_masses() { return grid_buffer_.d_g_masses; }
    const T* grid_masses() const { return grid_buffer_.d_g_masses; }
    T* grid_momentum() { return grid_buffer_.d_g_momentum; }
    const T* grid_momentum() const { return grid_buffer_.d_g_momentum; }
    uint32_t* grid_touched_flags() { return grid_buffer_.d_g_touched_flags; }
    const uint32_t* grid_touched_flags() const { return grid_buffer_.d_g_touched_flags; }
    uint32_t* grid_touched_ids() { return grid_buffer_.d_g_touched_ids; }
    const uint32_t* grid_touched_ids() const { return grid_buffer_.d_g_touched_ids; }
    uint32_t* grid_touched_cnt() { return grid_buffer_.d_g_touched_cnt; }
    const uint32_t* grid_touched_cnt() const { return grid_buffer_.d_g_touched_cnt; }
    uint32_t grid_touched_cnt_host() const {
        uint32_t h_g_touched_cnt = 0u;
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        CUDA_SAFE_CALL(cudaMemcpy(&h_g_touched_cnt, this->grid_touched_cnt(), sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return h_g_touched_cnt;
    }

    unsigned int* sort_buffer() { return sort_buffer_; };
    size_t& sort_buffer_size() { return sort_buffer_size_; }

    std::vector<Vec3<T>>& positions_host() { return h_positions_; }
    const std::vector<Vec3<T>>& positions_host() const { return h_positions_; }
    std::vector<Vec3<T>>& velocities_host() { return h_velocities_; }
    const std::vector<Vec3<T>>& velocities_host() const { return h_velocities_; }
    std::vector<T>& volumes_host() { return h_volumes_; }
    const std::vector<T>& volumes_host() const { return h_volumes_; }
    std::vector<int>& contact_ids_host() { return h_contact_ids_; }
    const std::vector<int>& contact_ids_host() const { return h_contact_ids_; }
    std::vector<Vec3<T>>& post_contact_dv_host() { return h_post_contact_dv_; }
    const std::vector<Vec3<T>>& post_contact_dv_host() const { return h_post_contact_dv_; }

    void AddQRCloth(const std::vector<Vec3<T>> &pos, 
                           const std::vector<Vec3<T>> &vel,
                           const std::vector<int> &indices);

    // NOTE (changyu): finalize system configuration and initialize GPU MPM state, 
    // all gpu memory allocation should be done here to avoid re-allocation.    
    void Finalize();

    // NOTE (changyu): free GPU MPM state, all gpu memory free should be done here.
    void Destroy();

    void SwitchCurrentState() { current_particle_buffer_id_ ^= 1; }
    void BackUpState();
    void RestoreStateFromBackup();

    // NOTE (changyu): sync all visualization data to CPU side.
    using DumpT = std::tuple<std::vector<Vec3<T>>, std::vector<int>>;
    DumpT DumpCpuState() const;

private:

    // Particles state device ptrs
    size_t n_verts_ = 0;
    size_t n_faces_ = 0;
    size_t n_particles_ = 0;

    // scratch data
    T* d_forces_ = nullptr;       // size: n_faces + n_verts, NO sort
    T* d_taus_ = nullptr;         // size: n_faces + n_verts, NO sort
    // NOTE (changyu): when particle data get sorted, 
    // the index mapping from element/vertex index to particle index will be changed,
    // need to be updated in `compute_sorted_state_kernel`.
    int* d_index_mappings_ = nullptr; // size: n_faces + n_verts

    // element-based data
    int* d_indices_ = nullptr; // size: n_faces NO sort
    T* d_Dm_inverses_ = nullptr; // size: n_faces NO sort

    T* d_deformation_gradients_ = nullptr; // size: n_faces NO sort
    // NOTE (changyu): backup state used for substepping.
    T* d_backup_deformation_gradients_ = nullptr; // size: n_faces NO sort
    
    struct ParticleBuffer {
        T* d_positions = nullptr;   // size: n_faces + n_verts
        T* d_velocities = nullptr;  // size: n_faces + n_verts
        T* d_volumes = nullptr;     // size: n_faces + n_verts
        T* d_affine_matrices = nullptr; // size: n_faces + n_verts

        // used to work with index_mapping to get the original -> reordered mapping.
        int* d_pids = nullptr; // size: n_faces + n_verts
        uint32_t* d_sort_keys = nullptr;
        uint32_t* d_sort_ids = nullptr;
    };
    
    uint32_t current_particle_buffer_id_ = 0;
    // NOTE (changyu): 
    //    particle_buffer_[0/1] is used for switch between current state and next state,
    //    particle buffer [2] is used for the backup state.
    //    Note that while one of buffer[0/1] will not be used after sorting in a timestep,
    //    we cannot reuse it as the backup state,
    //    since it's already reserved for the collision state.
    std::array<ParticleBuffer, 3> particle_buffer_;

    size_t sort_buffer_size_ = 0;
    static constexpr size_t backup_buffer_id = 2;
    unsigned int* sort_buffer_ = nullptr;

    // Particles state host ptrs
    std::vector<Vec3<T>> h_positions_;
    std::vector<Vec3<T>> h_velocities_;
    std::vector<T> h_volumes_;
    std::vector<int> h_indices_;
    // host ptrs for post contact solving
    std::vector<int> h_contact_ids_;
    std::vector<Vec3<T>> h_post_contact_dv_;

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