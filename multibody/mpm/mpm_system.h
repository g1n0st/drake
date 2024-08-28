#pragma once

#include "mpm_driver.h"

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/perception/point_cloud.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
class MpmSystem : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MpmSystem);

  MpmSystem(const MultibodyPlant<double>& plant, T dx, int num_substeps,
            Parallelism parallelism);

  const systems::OutputPort<double>& rigid_forces_output_port() const {
    return this->get_output_port(rigid_forces_output_port_);
  }

  const systems::OutputPort<double>& particles_output_port() const {
    return this->get_output_port(particles_output_port_);
  }

  const systems::InputPort<double>& query_object_input_port() const {
    return this->get_input_port(query_object_input_port_);
  }

  const systems::InputPort<double>& spatial_velocities_input_port() const {
    return this->get_input_port(spatial_velocities_input_port_);
  }

  const systems::InputPort<double>& poses_input_port() const {
    return this->get_input_port(poses_input_port_);
  }

  void Finalize();

  /* Sample particles inside the given geometry.
   @param[in] geometry_instance The geometry instance to sample particles
   inside. Only the shape and pose of the geometry is used; all the geometry
   properties are discarded.
   @param[in] particles_per_cell The targeted number of particle to be
   sampled in each grid cell (of size dx * dx * dx).
   @param[in] config  The physical properties of the material. */
  void SampleParticles(
      std::unique_ptr<geometry::GeometryInstance> geometry_instance,
      int particles_per_cell, const fem::DeformableBodyConfig<T>& config);

 private:
  systems::EventStatus UpdateMpmState(const systems::Context<double>& context,
                                      systems::State<double>* state) const;

  void CalcRigidForces(
      const systems::Context<double>& context,
      std::vector<ExternallyAppliedSpatialForce<double>>* output) const;

  void CalcParticles(const systems::Context<double>& context,
                     perception::PointCloud* output) const;

  internal::MpmDriver<T> blueprint_;
  systems::AbstractStateIndex mpm_state_index_;
  systems::InputPortIndex query_object_input_port_;
  systems::InputPortIndex spatial_velocities_input_port_;
  systems::InputPortIndex poses_input_port_;
  systems::OutputPortIndex rigid_forces_output_port_;
  systems::OutputPortIndex particles_output_port_;
  std::unordered_map<geometry::GeometryId, multibody::BodyIndex>
      geometry_id_to_body_index_;
  bool is_finalized_{false};
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
