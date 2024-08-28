#include "sparse_grid.h"

#include <utility>

#include "ips2ra/ips2ra.hpp"

#include "drake/multibody/plant/contact_properties.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

using drake::geometry::ProximityProperties;
using drake::geometry::SignedDistanceToPoint;
using Eigen::Vector3d;

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/* Solves the contact problem for a single particle against a rigid body
 assuming the rigid body has infinite mass and inertia.

 Let phi be the penetration distance (positive when penetration occurs) and vn
 be the relative velocity of the particle with respect to the rigid body in the
normal direction (vn>0 when separting). Then we have phi_dot = -vn.

In the normal direction, the contact force is modeled as a linear elastic system
with Hunt-Crossley dissipation.

  f = k * phi_+ * (1 + d * phi_dot)_+

  where phi_+ = max(0, phi)

The momentum balance in the normal direction becomes

m(vn_next - vn) = k * dt * (phi0 - dt * vn_next)_+ * (1 - d * vn_next)_+

where we used the fact that phi = phi0 - dt * vn_next. This is a quadratic
equation in vn_next, and we solve it to get the next velocity vn_next.

The quadratic equation is ax^2 + bx + c = 0, where

a = k * d * dt^2
b = -m - (k * dt * (dt + d * phi0))
c = k * dt * phi0 + m * vn

After solving for vn_next, we check if the friction force lies in the friction
cone, if not, we project the velocity back into the friction cone. */
template <typename T>
class ContactForceSolver {
 public:
  ContactForceSolver(T dt, T k, T d) : dt_(dt), k_(k), d_(d) {}
  // TODO(xuchenhan-tri): Take in the entire velocity vector and return the
  // next velocity (vector) after treating friction.
  T Solve(T m, T v0, T phi0) {
    T v_hat = std::min(phi0 / dt_, 1 / d_);
    if (v0 > v_hat) return v0;
    T a = k_ * d_ * dt_ * dt_;
    T b = -m - (k_ * dt_ * (dt_ + d_ * phi0));
    T c = k_ * dt_ * phi0 + m * v0;
    T discriminant = b * b - 4.0 * a * c;
    T v_next = (-b - std::sqrt(discriminant)) / (2.0 * a);
    return v_next;
  }

 private:
  T dt_;
  T k_;
  T d_;
};

template <typename T>
SparseGrid<T>::SparseGrid(T dx, Parallelism parallelism)
    : dx_(dx),
      allocator_(std::make_unique<Allocator>(kMaxGridSize, kMaxGridSize,
                                             kMaxGridSize)),
      blocks_(std::make_unique<PageMap>(*allocator_)),
      padded_blocks_(std::make_unique<PageMap>(*allocator_)),
      doubly_padded_blocks_(std::make_unique<PageMap>(*allocator_)),
      parallelism_(parallelism) {
  DRAKE_DEMAND(dx > 0);

  /* Compute the block offset strides. */
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      for (int k = -1; k <= 1; ++k) {
        block_offset_strides_[i + 1][j + 1][k + 1] =
            Mask::Linear_Offset(i * kNumNodesInBlockX, j * kNumNodesInBlockY,
                                k * kNumNodesInBlockZ);
      }
    }
  }
  /* Compute the cell offset strides. */
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      for (int k = -1; k <= 1; ++k) {
        cell_offset_strides_[i + 1][j + 1][k + 1] =
            Mask::Linear_Offset(i, j, k);
      }
    }
  }

  /* Maintain the invariance that the last entry in sentinel_particles_ is the
   number of particles. */
  sentinel_particles_.push_back(0);
}

template <typename T>
void SparseGrid<T>::Allocate(const std::vector<Vector3<T>>& q_WPs) {
  SortParticleIndices(q_WPs);
  blocks_->Clear();
  padded_blocks_->Clear();

  /* Touch all blocks that contain particles. */
  for (int i = 0; i < ssize(sentinel_particles_) - 1; ++i) {
    blocks_->Set_Page(base_node_offsets_[sentinel_particles_[i]]);
  }
  blocks_->Update_Block_Offsets();
  auto [block_offsets, num_blocks] = blocks_->Get_Blocks();
  /* Touch all neighboring blocks of each block in `blocks_` to ensure all grid
   nodes that might be affected by a particle has memory allocated. */
  for (int b = 0; b < static_cast<int>(num_blocks); ++b) {
    const uint64_t current_offset = block_offsets[b];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const uint64_t neighbor_block_offset =
              Mask::Packed_Add(current_offset, block_offset_strides_[i][j][k]);
          padded_blocks_->Set_Page(neighbor_block_offset);
        }
      }
    }
  }
  padded_blocks_->Update_Block_Offsets();
  std::tie(block_offsets, num_blocks) = padded_blocks_->Get_Blocks();
  const uint64_t page_size = 1 << kLog2Page;
  const uint64_t data_size = 1 << kDataBits;
  /* Note that Array is a wrapper around pointer to data memory and can be
   cheaply copied. */
  Array data = allocator_->Get_Array();
  /* Zero out the data in each block. */
  for (int b = 0; b < static_cast<int>(num_blocks); ++b) {
    const uint64_t offset = block_offsets[b];
    for (uint64_t i = 0; i < page_size; i += data_size) {
      data(offset + i).reset_mass_and_velocity();
    }
  }
}

template <typename T>
void SparseGrid<T>::AllocateForCollision(const std::vector<Vector3<T>>& q_WPs) {
  Allocate(q_WPs);
  auto [block_offsets, num_blocks] = padded_blocks_->Get_Blocks();
  /* Touch all neighboring blocks of each block in `padded_blocks_`. */
  for (int b = 0; b < static_cast<int>(num_blocks); ++b) {
    const uint64_t current_offset = block_offsets[b];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          const uint64_t neighbor_block_offset =
              Mask::Packed_Add(current_offset, block_offset_strides_[i][j][k]);
          doubly_padded_blocks_->Set_Page(neighbor_block_offset);
        }
      }
    }
  }
  doubly_padded_blocks_->Update_Block_Offsets();
  std::tie(block_offsets, num_blocks) = doubly_padded_blocks_->Get_Blocks();
  Array data = allocator_->Get_Array();
  const uint64_t page_size = 1 << kLog2Page;
  const uint64_t data_size = 1 << kDataBits;
  /* Zero out the data in each block. */
  for (int b = 0; b < static_cast<int>(num_blocks); ++b) {
    const uint64_t offset = block_offsets[b];
    for (uint64_t i = 0; i < page_size; i += data_size) {
      data(offset + i).set_zero();
    }
  }
}

template <typename T>
Pad<Vector3<T>> SparseGrid<T>::GetPadNodes(const Vector3<T>& q_WP) const {
  Pad<Vector3<T>> result;
  const Vector3<int> base_node = ComputeBaseNode<T>(q_WP / dx_);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        const Vector3<int> offset(i - 1, j - 1, k - 1);
        result[i][j][k] = dx_ * (base_node + offset).cast<T>();
      }
    }
  }
  return result;
}

template <typename T>
Pad<GridData<T>> SparseGrid<T>::GetPadData(uint64_t center_node_offset) const {
  Pad<GridData<T>> result;
  ConstArray data = allocator_->Get_Const_Array();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        const uint64_t offset =
            Mask::Packed_Add(center_node_offset, cell_offset_strides_[i][j][k]);
        result[i][j][k] = data(offset);
      }
    }
  }
  return result;
}

template <typename T>
void SparseGrid<T>::SetPadData(uint64_t center_node_offset,
                               const Pad<GridData<T>>& pad_data) {
  Array grid_data = allocator_->Get_Array();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        const uint64_t offset =
            Mask::Packed_Add(center_node_offset, cell_offset_strides_[i][j][k]);
        grid_data(offset) = pad_data[i][j][k];
      }
    }
  }
}

template <typename T>
void SparseGrid<T>::RasterizeRigidData(
    const geometry::QueryObject<double>& query_object,
    const std::vector<multibody::SpatialVelocity<double>>& spatial_velocities,
    const std::vector<math::RigidTransform<double>>& poses,
    const std::unordered_map<geometry::GeometryId, multibody::BodyIndex>&
        geometry_id_to_body_index,
    std::vector<multibody::ExternallyAppliedSpatialForce<double>>*
        rigid_forces) {
  DRAKE_DEMAND(rigid_forces != nullptr);
  DRAKE_DEMAND(spatial_velocities.size() == poses.size());
  rigid_forces->resize(spatial_velocities.size());
  for (int i = 0; i < ssize(*rigid_forces); ++i) {
    auto& force = (*rigid_forces)[i];
    force.body_index = BodyIndex(i);
    /* We use `p_BoBq_B` to temporarily store p_WB. We will replace it with the
     actual value of p_BoBq_B later on. */
    force.p_BoBq_B = poses[i].translation();
    force.F_Bq_W = SpatialForce<double>(Vector3d::Zero(), Vector3d::Zero());
  }
  const uint64_t data_size = 1 << kDataBits;
  auto [block_offsets, num_blocks] = doubly_padded_blocks_->Get_Blocks();
  Array grid_data = allocator_->Get_Array();
  for (int b = 0; b < static_cast<int>(num_blocks); ++b) {
    const uint64_t block_offset = block_offsets[b];
    uint64_t node_offset = block_offset;
    /* The coordinate of the origin of this block. */
    const Vector3<int> block_origin = OffsetToCoordinate(block_offset);
    for (int i = 0; i < kNumNodesInBlockX; ++i) {
      for (int j = 0; j < kNumNodesInBlockY; ++j) {
        for (int k = 0; k < kNumNodesInBlockZ; ++k) {
          GridData<T>& node_data = grid_data(node_offset);
          node_offset += data_size;
          /* World frame position of the node. */
          const Vector3<T> p_WN =
              (block_origin + Vector3<int>(i, j, k)).cast<T>() * dx_;
          const std::vector<SignedDistanceToPoint<double>>& signed_distances =
              query_object.ComputeSignedDistanceToPoint(
                  p_WN.template cast<double>());
          double min_distance = 0.0;
          int geometry_index = -1;
          for (int g = 0; g < ssize(signed_distances); ++g) {
            const SignedDistanceToPoint<double>& signed_distance =
                signed_distances[g];
            if (signed_distance.distance < min_distance) {
              min_distance = signed_distance.distance;
              geometry_index = g;
            }
          }
          if (min_distance < 0.0) {
            const SignedDistanceToPoint<double>& min_val =
                signed_distances[geometry_index];
            const geometry::GeometryId geometry_id = min_val.id_G;
            const CoulombFriction<double>& coulomb_friction =
                multibody::internal::GetCoulombFriction(
                    geometry_id, query_object.inspector());
            node_data.mu = coulomb_friction.dynamic_friction();
            node_data.phi = -min_distance;
            const int body_index = geometry_id_to_body_index.at(geometry_id);
            node_data.index = body_index;
            node_data.nhat_W = min_val.grad_W.normalized().cast<T>();
            /* World frame position of the origin of the rigid body. */
            const Vector3<double>& p_WR = poses[body_index].translation();
            const Vector3<double> p_RN = p_WN.template cast<double>() - p_WR;
            /* World frame velocity of a point affixed to the rigid body that
             coincide with the the grid node. */
            const Vector3<double> v_WN =
                spatial_velocities[body_index].Shift(p_RN).translational();
            node_data.rigid_v = v_WN.cast<T>();
          }
        }
      }
    }
  }
}

template <typename T>
void SparseGrid<T>::ExplicitVelocityUpdate(
    const Vector3<T>& dv,
    std::vector<multibody::ExternallyAppliedSpatialForce<double>>*
        rigid_forces) {
  const uint64_t data_size = 1 << kDataBits;
  auto [block_offsets, num_blocks] = padded_blocks_->Get_Blocks();
  Array grid_data = allocator_->Get_Array();
  for (int b = 0; b < static_cast<int>(num_blocks); ++b) {
    const T kStiffness = 1e6;
    const T kDamping = 10.0;
    ContactForceSolver<T> solver(1e-3, kStiffness, kDamping);
    const uint64_t block_offset = block_offsets[b];
    uint64_t node_offset = block_offset;
    /* The coordinate of the origin of this block. */
    const Vector3<int> block_origin = OffsetToCoordinate(block_offset);
    for (int i = 0; i < kNumNodesInBlockX; ++i) {
      for (int j = 0; j < kNumNodesInBlockY; ++j) {
        for (int k = 0; k < kNumNodesInBlockZ; ++k) {
          GridData<T>& node_data = grid_data(node_offset);
          const T& m = node_data.m;
          node_offset += data_size;
          if (m > 0.0) {
            node_data.v /= m;
            node_data.v += dv;
            if (node_data.index >= 0) {
              // TODO(xuchenhan-tri): Consider the friction coefficient of
              // particles. Currently, we are using the friction coefficient of
              // the rigid body as the combined friction coefficient.
              const Vector3<T>& nhat_W = node_data.nhat_W;
              const Vector3<T> v_rel = node_data.v - node_data.rigid_v;
              const T& mu = node_data.mu;
              const T vn = v_rel.dot(nhat_W);
              const T& phi0 = node_data.phi;
              const Vector3<T> vt = v_rel - vn * nhat_W;
              const T vn_next = solver.Solve(m, vn, phi0);
              if (vn != vn_next) {
                const Vector3<T> old_v = node_data.v;
                T dvn = vn_next - vn;
                node_data.v += dvn * nhat_W;
                if (dvn * mu < vt.norm()) {
                  node_data.v -= dvn * mu * vt.normalized();
                } else {
                  node_data.v -= vt;
                }
                /* We negate the sign of the grid node's momentum change to get
                 the impulse applied to the rigid body at the grid node. */
                const Vector3d l_WN_W =
                    (m * (old_v - node_data.v)).template cast<double>();
                const Vector3d p_WN =
                    (block_origin + Vector3<int>(i, j, k)).cast<double>() * dx_;
                const Vector3d& p_WB =
                    rigid_forces->at(node_data.index).p_BoBq_B;
                const Vector3d p_BN_W = p_WN - p_WB;
                /* The angular impulse applied to the rigid body at the grid
                 node. */
                const Vector3d h_WNBo_W = p_BN_W.cross(l_WN_W);
                /* Use `F_Bq_W` to store the spatial impulse applied to the body
                 at its origin, expressed in the world frame. */
                rigid_forces->at(node_data.index).F_Bq_W +=
                    SpatialForce<double>(h_WNBo_W, l_WN_W);
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
Vector3<int> SparseGrid<T>::OffsetToCoordinate(uint64_t offset) const {
  const std::array<int, 3> reference_space_coordinate =
      Mask::LinearToCoord(offset);
  const std::array<int, 3> reference_space_origin =
      Mask::LinearToCoord(origin_offset_);
  return Vector3<int>(
      reference_space_coordinate[0] - reference_space_origin[0],
      reference_space_coordinate[1] - reference_space_origin[1],
      reference_space_coordinate[2] - reference_space_origin[2]);
}

template <typename T>
void SparseGrid<T>::SetGridData(
    const std::function<GridData<T>(const Vector3<int>&)>& callback) {
  auto [block_offsets, num_blocks] = padded_blocks_->Get_Blocks();
  const uint64_t page_size = 1 << kLog2Page;
  const uint64_t data_size = 1 << kDataBits;
  Array data = allocator_->Get_Array();
  for (int b = 0; b < static_cast<int>(num_blocks); ++b) {
    const uint64_t offset = block_offsets[b];
    for (uint64_t i = 0; i < page_size; i += data_size) {
      const Vector3<int> coordinate = OffsetToCoordinate(offset + i);
      data(offset + i) = callback(coordinate);
    }
  }
}

template <typename T>
std::vector<std::pair<Vector3<int>, GridData<T>>> SparseGrid<T>::GetGridData()
    const {
  const uint64_t page_size = 1 << kLog2Page;
  const uint64_t data_size = 1 << kDataBits;
  std::vector<std::pair<Vector3<int>, GridData<T>>> result;
  auto [block_offsets, num_blocks] = padded_blocks_->Get_Blocks();
  ConstArray grid_data = allocator_->Get_Const_Array();
  for (int b = 0; b < static_cast<int>(num_blocks); ++b) {
    const uint64_t block_offset = block_offsets[b];
    for (uint64_t i = 0; i < page_size; i += data_size) {
      const uint64_t node_offset = block_offset + i;
      const GridData<T>& node_data = grid_data(node_offset);
      if (node_data.m > 0.0) {
        const Vector3<int> node_coordinate = OffsetToCoordinate(node_offset);
        result.emplace_back(std::make_pair(node_coordinate, node_data));
      }
    }
  }
  return result;
}

template <typename T>
MassAndMomentum<T> SparseGrid<T>::ComputeTotalMassAndMomentum() const {
  MassAndMomentum<T> result;
  const std::vector<std::pair<Vector3<int>, GridData<T>>> grid_data =
      GetGridData();
  for (int i = 0; i < ssize(grid_data); ++i) {
    const T& mi = grid_data[i].second.m;
    const Vector3<T>& vi = grid_data[i].second.v;
    const Vector3<T>& xi = grid_data[i].first.template cast<T>() * dx_;
    result.mass += grid_data[i].second.m;
    result.linear_momentum += mi * vi;
    result.angular_momentum += mi * xi.cross(vi);
  }
  return result;
}

template <typename T>
void SparseGrid<T>::SortParticleIndices(const std::vector<Vector3<T>>& q_WPs) {
  const int num_particles = q_WPs.size();
  data_indices_.resize(num_particles);
  base_node_offsets_.resize(num_particles);
  particle_sorters_.resize(num_particles);

  /* We sort particles first based on their base node offsets, and if those
   are the same, we sort by their data indices. To do that, we notice that the
   base node offset of the particle looks like

       page bits | block bits | data bits

   with all the data bits being equal to zero. Also, the left most bits of the
   page bits are zero because at most 2^(3*kLog2MaxGridSize) number of grid
   nodes and that takes up 3*kLog2MaxGridSize bits. The page bits and block
   bits have 64 - data bits in total, so the left most 64 - data bits - 3 *
   kLog2MaxGridSize bits are zero. So we left shift the base node offset by
   that amount and now we get the lowest 64 - 3 * kLog2MaxGridSize bits (which
   we name `kIndexBits`) to be zero. With kLog2MaxGridSize == 10, we have 44
   bits to work with, more than enough to store the particle indices. We then
   sort the resulting 64 bit unsigned integers which is enough to achieve the
   sorting objective. */
  constexpr int kIndexBits = 64 - 3 * kLog2MaxGridSize;
  constexpr int kZeroPageBits = 64 - kDataBits - 3 * kLog2MaxGridSize;
  [[maybe_unused]] const int num_threads = parallelism_.num_threads();

#if defined(_OPENMP)
#pragma omp parallel for num_threads(num_threads)
#endif
  for (int p = 0; p < num_particles; ++p) {
    const Vector3<int> base_node = ComputeBaseNode<T>(q_WPs[p] / dx_);
    base_node_offsets_[p] =
        CoordinateToOffset(base_node[0], base_node[1], base_node[2]);
    data_indices_[p] = p;
    /* Confirm the data bits of the base node offset are all zero. */
    DRAKE_ASSERT((base_node_offsets_[p] & ((uint64_t(1) << kDataBits) - 1)) ==
                 0);
    /* Confirm the left most bits in the page bits are unused. */
    DRAKE_ASSERT((base_node_offsets_[p] &
                  ~((uint64_t(1) << (64 - kZeroPageBits)) - 1)) == 0);
    particle_sorters_[p] =
        (base_node_offsets_[p] << kZeroPageBits) + data_indices_[p];
  }

#if defined(_OPENMP)
  ips2ra::parallel::sort(particle_sorters_.begin(), particle_sorters_.end(),
                         ips2ra::Config<>::identity{}, num_threads);
#else
  ips2ra::sort(particle_sorters_.begin(), particle_sorters_.end());
#endif

  /* Peel off the data indices and the base node offsets from
   particle_sorters_. Meanwhile, reorder the data indices and the base node
   offsets based on the sorting results. */
#if defined(_OPENMP)
#pragma omp parallel for num_threads(num_threads)
#endif
  for (int p = 0; p < ssize(particle_sorters_); ++p) {
    data_indices_[p] = particle_sorters_[p] & ((uint64_t(1) << kIndexBits) - 1);
    base_node_offsets_[p] = (particle_sorters_[p] >> kIndexBits) << kDataBits;
  }

  /* Record the sentinel particles and the coloring of the blocks. */
  sentinel_particles_.clear();
  for (int b = 0; b < 8; ++b) {
    colored_blocks_[b].clear();
  }
  uint64_t previous_page{};
  int block = 0;
  for (int p = 0; p < num_particles; ++p) {
    /* The bits in the offset is ordered as follows:

      page bits | block bits | data bits

     block bits and data bits add up to kLog2Page bits.
     We right shift to get the page bits. */
    const uint64_t page = base_node_offsets_[p] >> kLog2Page;
    if (p == 0 || previous_page != page) {
      previous_page = page;
      sentinel_particles_.push_back(p);
      const int color = get_color(page);
      colored_blocks_[color].push_back(block++);
    }
  }
  sentinel_particles_.push_back(num_particles);
}

template <typename T>
void SparseGrid<T>::SortParticles(std::vector<Vector3<double>>* q_WPs) const {
  DRAKE_DEMAND(q_WPs != nullptr);
  const int num_particles = q_WPs->size();
  std::vector<uint64_t> particle_sorters(num_particles);
  constexpr int kIndexBits = 64 - 3 * kLog2MaxGridSize;
  constexpr int kZeroPageBits = 64 - kDataBits - 3 * kLog2MaxGridSize;
  for (int p = 0; p < num_particles; ++p) {
    const Vector3<int> base_node = ComputeBaseNode<double>(q_WPs->at(p) / dx_);
    uint64_t base_node_offsets =
        CoordinateToOffset(base_node[0], base_node[1], base_node[2]);
    uint64_t data_indices = p;
    particle_sorters[p] = (base_node_offsets << kZeroPageBits) + data_indices;
  }
  ips2ra::sort(particle_sorters.begin(), particle_sorters.end());
  std::vector<Vector3<double>> result(num_particles);
  for (int p = 0; p < ssize(particle_sorters); ++p) {
    int data_indices = particle_sorters[p] & ((uint64_t(1) << kIndexBits) - 1);
    result[p] = q_WPs->at(data_indices);
  }
  *q_WPs = result;
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

template class drake::multibody::mpm::internal::SparseGrid<float>;
template class drake::multibody::mpm::internal::SparseGrid<double>;
