#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <cuda_runtime.h>

#include "multibody/gpu_mpm/cuda_mpm_model.cuh"
#include "multibody/gpu_mpm/cuda_mpm_kernels.cuh"
#include "multibody/gpu_mpm/radix_sort.cuh"

namespace drake {
namespace multibody {
namespace gmpm {

template<typename T>
void GpuMpmState<T>::AddQRCloth(const std::vector<Vec3<T>> &pos,
                                const std::vector<Vec3<T>> &vel,
                                const std::vector<int> &indices) {
    const auto &verts = pos.size();
    const auto &faces = indices.size() / 3;
    assert(faces * 3  == indices.size());

    h_positions_.insert(h_positions_.end(), pos.begin(), pos.end());
    h_velocities_.insert(h_velocities_.end(), vel.begin(), vel.end());

    for (auto &v : indices) {
        h_indices_.push_back(v + n_verts_);
    }

    n_particles_ += verts + faces;
    n_verts_ += verts;
    n_faces_ += faces;
}

template<typename T>
void GpuMpmState<T>::Finalize() {
    h_positions_.resize(n_particles_);
    h_velocities_.resize(n_particles_);

    // NOTE (changyu): at the initial state, position/velocity is organized as [n_faces | n_verts].
    std::move_backward(h_positions_.begin(), h_positions_.begin() + h_positions_.size() - n_faces_, h_positions_.end());
    std::move_backward(h_velocities_.begin(), h_velocities_.begin() + h_velocities_.size() - n_faces_, h_velocities_.end());
    for (auto &v : h_indices_) {
        v += n_faces_;
    }

    // device particle buffer allocation for reorder data
    for (uint32_t i = 0; i < 2; ++i) {
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_positions, sizeof(Vec3<T>) * n_particles_));
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_velocities, sizeof(Vec3<T>) * n_particles_));
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_volumes, sizeof(T) * n_particles_));
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_affine_matrices, sizeof(Mat3<T>) * n_particles_));

        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_pids, sizeof(int) * n_particles_));
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_sort_keys, sizeof(uint32_t) * n_particles_));
        CUDA_SAFE_CALL(cudaMalloc(&particle_buffer_[i].d_sort_ids, sizeof(uint32_t) * n_particles_));
        CUDA_SAFE_CALL(cudaMemset(particle_buffer_[i].d_sort_keys, 0, sizeof(uint32_t) * n_particles_));
        CUDA_SAFE_CALL(cudaMemset(particle_buffer_[i].d_sort_ids, 0, sizeof(uint32_t) * n_particles_));

        if (i == current_particle_buffer_id_) {
            std::vector<int> id_sequence(n_particles_);
            std::iota(id_sequence.begin(), id_sequence.end(), 0);
            CUDA_SAFE_CALL(cudaMemcpy(particle_buffer_[i].d_pids, 
                                      id_sequence.data(), 
                                      sizeof(int) * n_particles_, 
                                      cudaMemcpyHostToDevice));

            CUDA_SAFE_CALL(cudaMemcpy(particle_buffer_[i].d_positions, 
                                      h_positions_.data(), 
                                      sizeof(Vec3<T>) * n_particles_, 
                                      cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(particle_buffer_[i].d_velocities, 
                                      h_velocities_.data(), 
                                      sizeof(Vec3<T>) * n_particles_, 
                                      cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemset(particle_buffer_[i].d_volumes, 0, sizeof(T) * n_particles_));
            CUDA_SAFE_CALL(cudaMemset(particle_buffer_[i].d_affine_matrices, 0, sizeof(Mat3<T>) * n_particles_));
        }
    }
    
    // scratch data
    CUDA_SAFE_CALL(cudaMalloc(&d_forces_, sizeof(Vec3<T>) * n_particles_));
    CUDA_SAFE_CALL(cudaMalloc(&d_taus_, sizeof(Mat3<T>) * n_particles_));
    CUDA_SAFE_CALL(cudaMalloc(&d_index_mappings_, sizeof(int) * n_particles_));
    std::vector<int> initial_index_mappings(n_particles_);
    std::iota(initial_index_mappings.begin(), initial_index_mappings.end(), 0);
    CUDA_SAFE_CALL(cudaMemcpy(d_index_mappings_, initial_index_mappings.data(), sizeof(int) * n_particles_, cudaMemcpyHostToDevice));

    // element-based data
    CUDA_SAFE_CALL(cudaMalloc(&d_deformation_gradients_, sizeof(Mat3<T>) * n_faces_));
    CUDA_SAFE_CALL(cudaMalloc(&d_Dm_inverses_, sizeof(Mat2<T>) * n_faces_));
    CUDA_SAFE_CALL(cudaMalloc(&d_indices_, sizeof(int) * n_faces_ * 3));
    CUDA_SAFE_CALL(cudaMemcpy(d_indices_, h_indices_.data(), sizeof(int) * n_faces_ * 3, cudaMemcpyHostToDevice));


    // device grid buffer allocation
    // NOTE(changyu): considering the problem size, we pre-allocate the dense grid once and skip the untouched parts when traversal.
    CUDA_SAFE_CALL(cudaMalloc(&grid_buffer_.d_g_masses, config::G_DOMAIN_VOLUME * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc(&grid_buffer_.d_g_momentum, config::G_DOMAIN_VOLUME * sizeof(Vec3<T>)));
    CUDA_SAFE_CALL(cudaMalloc(&grid_buffer_.d_g_touched_flags, config::G_GRID_VOLUME * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&grid_buffer_.d_g_touched_ids, config::G_GRID_VOLUME * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&grid_buffer_.d_g_touched_cnt, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(grid_buffer_.d_g_touched_cnt, 0, sizeof(uint32_t)));

    CUDA_SAFE_CALL(cudaMalloc(&d_g_Hess_, config::G_DOMAIN_VOLUME * sizeof(Mat3<T>)));
    CUDA_SAFE_CALL(cudaMalloc(&d_g_Grad_, config::G_DOMAIN_VOLUME * sizeof(Vec3<T>)));
    CUDA_SAFE_CALL(cudaMalloc(&d_g_Dir_, config::G_DOMAIN_VOLUME * sizeof(Vec3<T>)));
    CUDA_SAFE_CALL(cudaMalloc(&d_g_alpha_, config::G_DOMAIN_VOLUME * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc(&d_g_v_star_, config::G_DOMAIN_VOLUME * sizeof(Vec3<T>)));

    radix_sort(this->next_sort_keys(), this->current_sort_keys(), this->next_sort_ids(), this->current_sort_ids(), sort_buffer_, sort_buffer_size_, static_cast<unsigned int>(n_particles_));
    CUDA_SAFE_CALL(cudaMalloc(&sort_buffer_, sizeof(unsigned int) * sort_buffer_size_));

    CUDA_SAFE_CALL((
        initialize_fem_state_kernel<<<
        (this->n_faces() + config::DEFAULT_CUDA_BLOCK_SIZE - 1) / config::DEFAULT_CUDA_BLOCK_SIZE, config::DEFAULT_CUDA_BLOCK_SIZE>>>
        (this->n_faces(), this->indices(), this->current_positions(), this->current_velocities(), this->current_volumes(),
         this->deformation_gradients(), this->Dm_inverses())
        ));
}

template<typename T>
void GpuMpmState<T>::Destroy() {
    for (uint32_t i = 0; i < 2; ++i) {
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_positions));
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_velocities));
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_volumes));
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_affine_matrices));

        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_pids));
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_sort_keys));
        CUDA_SAFE_CALL(cudaFree(particle_buffer_[i].d_sort_ids));

        // make sure to throw error when illegal access happens
        particle_buffer_[i].d_positions = nullptr;
        particle_buffer_[i].d_velocities = nullptr;
        particle_buffer_[i].d_volumes = nullptr;
        particle_buffer_[i].d_affine_matrices = nullptr;
        particle_buffer_[i].d_pids = nullptr;
        particle_buffer_[i].d_sort_keys = nullptr;
        particle_buffer_[i].d_sort_ids = nullptr;
    }

    CUDA_SAFE_CALL(cudaFree(d_forces_));
    CUDA_SAFE_CALL(cudaFree(d_taus_));
    CUDA_SAFE_CALL(cudaFree(d_index_mappings_));
    CUDA_SAFE_CALL(cudaFree(d_deformation_gradients_));
    CUDA_SAFE_CALL(cudaFree(d_Dm_inverses_));
    CUDA_SAFE_CALL(cudaFree(d_indices_));
    d_forces_ = nullptr;
    d_taus_ = nullptr;
    d_index_mappings_ = nullptr;
    d_deformation_gradients_ = nullptr;
    d_Dm_inverses_ = nullptr;
    d_indices_ = nullptr;

    CUDA_SAFE_CALL(cudaFree(grid_buffer_.d_g_masses));
    CUDA_SAFE_CALL(cudaFree(grid_buffer_.d_g_momentum));
    CUDA_SAFE_CALL(cudaFree(grid_buffer_.d_g_touched_flags));
    CUDA_SAFE_CALL(cudaFree(grid_buffer_.d_g_touched_ids));
    CUDA_SAFE_CALL(cudaFree(grid_buffer_.d_g_touched_cnt));
    grid_buffer_.d_g_masses = nullptr;
    grid_buffer_.d_g_momentum = nullptr;
    grid_buffer_.d_g_touched_flags = nullptr;
    grid_buffer_.d_g_touched_ids = nullptr;
    grid_buffer_.d_g_touched_cnt = nullptr;

    CUDA_SAFE_CALL(cudaFree(d_g_Hess_));
    CUDA_SAFE_CALL(cudaFree(d_g_Grad_));
    CUDA_SAFE_CALL(cudaFree(d_g_Dir_));
    CUDA_SAFE_CALL(cudaFree(d_g_alpha_));
    CUDA_SAFE_CALL(cudaFree(d_g_v_star_));
    d_g_Hess_ = nullptr;
    d_g_Grad_ = nullptr;
    d_g_Dir_ = nullptr;
    d_g_alpha_ = nullptr;
    d_g_v_star_ = nullptr;

    CUDA_SAFE_CALL(cudaFree(sort_buffer_));
    sort_buffer_ = nullptr;
    sort_buffer_size_ = 0;

    if (d_contact_mpm_id_) {
        CUDA_SAFE_CALL(cudaFree(d_contact_mpm_id_));
        d_contact_mpm_id_ = nullptr;
    }
    if (d_contact_pos_) {
        CUDA_SAFE_CALL(cudaFree(d_contact_pos_));
        d_contact_pos_ = nullptr;
    }
    if (d_contact_vel_) {
        CUDA_SAFE_CALL(cudaFree(d_contact_vel_));
        d_contact_vel_ = nullptr;
    }
    if (d_contact_dist_) {
        CUDA_SAFE_CALL(cudaFree(d_contact_dist_));
        d_contact_dist_ = nullptr;
    }
    if (d_contact_normal_) {
        CUDA_SAFE_CALL(cudaFree(d_contact_normal_));
        d_contact_normal_ = nullptr;
    }
    if (d_contact_rigid_v_) {
        CUDA_SAFE_CALL(cudaFree(d_contact_rigid_v_));
        d_contact_rigid_v_ = nullptr;
    }
    if (d_contact_sort_keys_) {
        CUDA_SAFE_CALL(cudaFree(d_contact_sort_keys_));
        d_contact_sort_keys_ = nullptr;
    }
    if (d_contact_sort_ids_) {
        CUDA_SAFE_CALL(cudaFree(d_contact_sort_ids_));
        d_contact_sort_keys_ = nullptr;
    }
}

template<typename T>
GpuMpmState<T>::DumpT GpuMpmState<T>::DumpCpuState() const {
    std::vector<Vec3<T>> export_pos;
    std::vector<Vec3<T>> export_original_pos;
    std::vector<int> export_pid;
    std::vector<int> export_indices;
    export_pos.resize(n_particles());
    export_original_pos.resize(n_particles());
    export_pid.resize(n_particles());
    export_indices.resize(n_faces() * 3);
    CUDA_SAFE_CALL(cudaMemcpy(export_pos.data(), current_positions(), sizeof(Vec3<T>) * n_particles(), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(export_pid.data(), current_pids(), sizeof(int) * n_particles(), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(export_indices.data(), indices(), sizeof(int) * n_faces() * 3, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n_particles(); ++i) {
      export_original_pos[export_pid[i]] = export_pos[i];
    }
    export_pos = std::vector<Vec3<T>>(export_original_pos.begin() + n_faces(), export_original_pos.end());
    for (size_t i = 0; i < n_faces() * 3; ++i) {
      export_indices[i] -= n_faces();
    }
    return std::make_tuple(export_pos, export_indices);
}

template<typename T>
void GpuMpmState<T>::ReallocateContacts(size_t num_contacts) {
    this->num_contacts_ = num_contacts;
    if (num_contacts > contact_buffer_size) {
        contact_buffer_size = num_contacts;
        if (d_contact_mpm_id_) {
            CUDA_SAFE_CALL(cudaFree(d_contact_mpm_id_));
        }
        if (d_contact_pos_) {
            CUDA_SAFE_CALL(cudaFree(d_contact_pos_));
        }
        if (d_contact_vel_) {
            CUDA_SAFE_CALL(cudaFree(d_contact_vel_));
        }
        if (d_contact_dist_) {
            CUDA_SAFE_CALL(cudaFree(d_contact_dist_));
        }
        if (d_contact_normal_) {
            CUDA_SAFE_CALL(cudaFree(d_contact_normal_));
        }
        if (d_contact_rigid_v_) {
            CUDA_SAFE_CALL(cudaFree(d_contact_rigid_v_));
        }
        if (d_contact_sort_keys_) {
            CUDA_SAFE_CALL(cudaFree(d_contact_sort_keys_));
        }
        if (d_contact_sort_ids_) {
            CUDA_SAFE_CALL(cudaFree(d_contact_sort_ids_));
        }
        cudaMalloc(&d_contact_mpm_id_, sizeof(uint32_t) * contact_buffer_size);
        cudaMalloc(&d_contact_pos_, sizeof(T) * 3 * contact_buffer_size);
        cudaMalloc(&d_contact_vel_, sizeof(T) * 3 * contact_buffer_size);
        cudaMalloc(&d_contact_dist_, sizeof(T) * contact_buffer_size);
        cudaMalloc(&d_contact_normal_, sizeof(T) * 3 * contact_buffer_size);
        cudaMalloc(&d_contact_rigid_v_, sizeof(T) * 3 * contact_buffer_size);
        cudaMalloc(&d_contact_sort_keys_, sizeof(uint32_t) * contact_buffer_size);
        cudaMalloc(&d_contact_sort_ids_, sizeof(uint32_t) * contact_buffer_size);
    }
}

template class GpuMpmState<config::GpuT>;

}  // namespace gmpm
}  // namespace multibody
}  // namespace drake