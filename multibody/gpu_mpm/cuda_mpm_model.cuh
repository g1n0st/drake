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

// NOTE (changyu): buffers used for aggregating substepping contact impulse to rigid bodies;
template <typename T>
struct ExternalSpatialForce {
    Vec3<T> p_BoBq_B;
    // a spatial force F from a torque 𝛕 (tau) and a force 𝐟.
    Vec3<T> F_Bq_W_tau;
    Vec3<T> F_Bq_W_f;
};

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

    T* forces() { return d_forces_; }
    const T* forces() const { return d_forces_; }
    T* taus() { return d_taus_; }
    const T* taus() const { return d_taus_; }
    T* deformation_gradients() { return d_deformation_gradients_; }
    const T* deformation_gradients() const { return d_deformation_gradients_; }
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

    T* grid_Hess() { return d_g_Hess_; }
    T* grid_Grad() { return d_g_Grad_; }
    T* grid_Dir() { return d_g_Dir_; }
    T* grid_alpha() { return d_g_alpha_; }
    T* grid_v_star() { return d_g_v_star_; }
    T* grid_E0() { return d_g_E0_; }
    T* grid_E1() { return d_g_E1_; }

    unsigned int* sort_buffer() { return sort_buffer_; };
    size_t& sort_buffer_size() { return sort_buffer_size_; }

    std::vector<Vec3<T>>& positions_host() { return h_positions_; }
    const std::vector<Vec3<T>>& positions_host() const { return h_positions_; }

    const uint32_t* contact_mpm_id() const { return d_contact_mpm_id_; }
    uint32_t* contact_mpm_id() { return d_contact_mpm_id_; }
    const T* contact_pos() const { return d_contact_pos_; }
    T* contact_pos() { return d_contact_pos_; }
    const T* contact_vel() const { return d_contact_vel_; }
    T* contact_vel() { return d_contact_vel_; }
    const T* contact_dist() const { return d_contact_dist_; }
    T* contact_dist() { return d_contact_dist_; }
    const T* contact_normal() const { return d_contact_normal_; }
    T* contact_normal() { return d_contact_normal_; }
    const T* contact_rigid_v() const { return d_contact_rigid_v_; }
    T* contact_rigid_v() { return d_contact_rigid_v_; }
    uint32_t* contact_sort_keys() { return d_contact_sort_keys_; }
    uint32_t* contact_sort_ids() { return d_contact_sort_ids_; }
    size_t num_contacts() const { return num_contacts_; }

    const std::vector<ExternalSpatialForce<T>>& external_forces_host() const { return h_external_forces_; }
    std::vector<ExternalSpatialForce<T>>& external_forces_host() { return h_external_forces_; }

    void AddQRCloth(const std::vector<Vec3<T>> &pos, 
                           const std::vector<Vec3<T>> &vel,
                           const std::vector<int> &indices);

    // NOTE (changyu): finalize system configuration and initialize GPU MPM state, 
    // all gpu memory allocation should be done here to avoid re-allocation.    
    void Finalize();

    // NOTE (changyu): free GPU MPM state, all gpu memory free should be done here.
    void Destroy();

    void SwitchCurrentState() { assert(false); current_particle_buffer_id_ ^= 1; }

    // NOTE (changyu): sync all visualization data to CPU side.
    using DumpT = std::tuple<std::vector<Vec3<T>>, std::vector<int>>;
    DumpT DumpCpuState() const;

    // NOTE (changyu): allocate contact pairs buffer based on given number of contacts
    void ReallocateContacts(size_t num_contacts);

    int total_contact_iteration_count = 0;

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
    //    particle_buffer_[0/1] is used for switch between current state and next state.
    std::array<ParticleBuffer, 2> particle_buffer_;

    // contact pairs device ptr
    size_t contact_buffer_size = 0;
    size_t num_contacts_ = 0;
    uint32_t* d_contact_mpm_id_ = nullptr;
    uint32_t* d_contact_sort_keys_ = nullptr;
    uint32_t* d_contact_sort_ids_ = nullptr;
    T* d_contact_pos_ = nullptr;
    T* d_contact_vel_ = nullptr;
    T* d_contact_dist_ = nullptr;
    T* d_contact_normal_ = nullptr;
    T* d_contact_rigid_v_ = nullptr;
    

    size_t sort_buffer_size_ = 0;
    unsigned int* sort_buffer_ = nullptr;

    // Particles state host ptrs
    std::vector<Vec3<T>> h_positions_;
    std::vector<Vec3<T>> h_velocities_;
    std::vector<int> h_indices_;

    std::vector<ExternalSpatialForce<T>> h_external_forces_;

    // Grid state device ptrs

    struct GridBuffer {
        T* d_g_masses = nullptr;
        T* d_g_momentum = nullptr;
        uint32_t* d_g_touched_flags = nullptr;
        uint32_t* d_g_touched_ids   = nullptr;
        uint32_t* d_g_touched_cnt   = nullptr;
    };

    GridBuffer grid_buffer_;

    // Grid device ptrs for solving coordinate descent
    T* d_g_Hess_ = nullptr;
    T* d_g_Grad_ = nullptr;
    T* d_g_Dir_  = nullptr;
    T* d_g_alpha_ = nullptr;
    T* d_g_v_star_ = nullptr;
    T* d_g_E0_ = nullptr;
    T* d_g_E1_ = nullptr;
};

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake