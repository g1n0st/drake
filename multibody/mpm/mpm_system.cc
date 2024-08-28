#include "mpm_system.h"

#include "particles_to_bgeo.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
MpmSystem<T>::MpmSystem(const MultibodyPlant<double>& plant, T dx,
                        int num_substeps, Parallelism parallelism)
    : blueprint_(plant.time_step(), dx, num_substeps, parallelism) {
  DRAKE_DEMAND(plant.time_step() > 0.0);
  DRAKE_DEMAND(plant.is_finalized());
  this->DeclarePeriodicUnrestrictedUpdateEvent(plant.time_step(), 0.0,
                                               &MpmSystem::UpdateMpmState);
  query_object_input_port_ =
      this->DeclareAbstractInputPort("geometry_query",
                                     Value<geometry::QueryObject<double>>{})
          .get_index();

  spatial_velocities_input_port_ =
      this->DeclareAbstractInputPort(
              "spatial_velocities",
              Value<std::vector<multibody::SpatialVelocity<double>>>{})
          .get_index();

  poses_input_port_ =
      this->DeclareAbstractInputPort(
              "poses", Value<std::vector<math::RigidTransform<double>>>{})
          .get_index();

  rigid_forces_output_port_ =
      this->DeclareAbstractOutputPort("rigid_forces",
                                      &MpmSystem::CalcRigidForces)
          .get_index();

  particles_output_port_ =
      this->DeclareAbstractOutputPort("particles", &MpmSystem::CalcParticles)
          .get_index();

  geometry_id_to_body_index_ = plant.geometry_id_to_body_index();
}

template <typename T>
void MpmSystem<T>::Finalize() {
  mpm_state_index_ =
      this->DeclareAbstractState(Value<internal::MpmDriver<T>>(blueprint_));
  is_finalized_ = true;
}

template <typename T>
void MpmSystem<T>::SampleParticles(
    std::unique_ptr<geometry::GeometryInstance> geometry_instance,
    int particles_per_cell, const fem::DeformableBodyConfig<T>& config) {
  DRAKE_THROW_UNLESS(!is_finalized_);
  blueprint_.SampleParticles(std::move(geometry_instance), particles_per_cell,
                             config);
}

template <typename T>
systems::EventStatus MpmSystem<T>::UpdateMpmState(
    const systems::Context<double>& context,
    systems::State<double>* state) const {
  DRAKE_THROW_UNLESS(is_finalized_);
  auto& mpm_state =
      state->template get_mutable_abstract_state<internal::MpmDriver<T>>(
          mpm_state_index_);
  const auto& query_object =
      query_object_input_port().template Eval<geometry::QueryObject<double>>(
          context);
  const auto& spatial_velocities =
      spatial_velocities_input_port()
          .template Eval<std::vector<multibody::SpatialVelocity<double>>>(
              context);
  const auto& poses =
      poses_input_port()
          .template Eval<std::vector<math::RigidTransform<double>>>(context);
  mpm_state.AdvanceOneTimeStep(query_object, spatial_velocities, poses,
                               geometry_id_to_body_index_);

  static int i = 0;
  const std::string directory = "/home/xuchenhan/Desktop/mpm_data/";
  const std::string filename = fmt::format("particles_{:04d}.bgeo", i++);
  internal::WriteParticlesToBgeo<T>(directory + filename,
                                    mpm_state.particles());

  return systems::EventStatus::Succeeded();
}

template <typename T>
void MpmSystem<T>::CalcRigidForces(
    const systems::Context<double>& context,
    std::vector<ExternallyAppliedSpatialForce<double>>* output) const {
  DRAKE_THROW_UNLESS(is_finalized_);
  const auto& mpm_state =
      context.get_abstract_state<internal::MpmDriver<T>>(mpm_state_index_);
  *output = mpm_state.rigid_forces();
}

template <typename T>
void MpmSystem<T>::CalcParticles(const systems::Context<double>& context,
                                 perception::PointCloud* output) const {
  DRAKE_THROW_UNLESS(is_finalized_);
  const auto& mpm_state =
      context.get_abstract_state<internal::MpmDriver<T>>(mpm_state_index_);
  const std::vector<Vector3<T>>& x = mpm_state.particles().x;
  const int num_particles = ssize(x);
  output->resize(num_particles, true);
  Eigen::Ref<Matrix3X<float>> p_WPs = output->mutable_xyzs();
  for (int i = 0; i < num_particles; ++i) {
    p_WPs.col(i) = x[i].template cast<float>();
  }
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake

template class drake::multibody::mpm::MpmSystem<double>;
template class drake::multibody::mpm::MpmSystem<float>;