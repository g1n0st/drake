#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/sap/partial_permutation.h"
#include "drake/multibody/contact_solvers/sap/sap_fixed_constraint.h"
#include "drake/multibody/contact_solvers/schur_complement.h"
#include "drake/multibody/fem/discrete_time_integrator.h"
#include "drake/multibody/fem/fem_solver.h"
#include "drake/multibody/plant/deformable_contact_info.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/discrete_contact_data.h"
#include "drake/multibody/plant/discrete_contact_pair.h"
#include "drake/systems/framework/context.h"
#include "drake/geometry/geometry_state.h"

// NOTE (changyu): GPU MPM solver header files
#include "drake/multibody/contact_solvers/contact_solver_results.h"
#include "multibody/gpu_mpm/cuda_mpm_solver.cuh"

namespace drake {
namespace multibody {
namespace internal {

/* Helper class for DeformableDriver that acts both as a multiplexer and a
 demultiplexer -- it combines multiple Eigen vectors into a single stacked
 vector and it also splits an Eigen vector into multiple vectors.
 @tparam_default_scalar */
template <typename T>
class Multiplexer {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Multiplexer);

  /* Create an invalid Multiplexer. It cannot be used to (de)multiplex any
   vectors. */
  Multiplexer() = default;

  /* Constructs a Multiplexer that combines and splits vectors of the given
   sizes.
   @pre `sizes` is not empty and each entry is non-negative. */
  explicit Multiplexer(std::vector<int> sizes);

  /* The number of vectors to be multiplexed. */
  int num_vectors() const { return sizes_.size(); }

  /* Combines the given vectors into a single vector.
   @throws std::exception if the sizes of `inputs` aren't compatible with the
   sizes provided at construction. */
  VectorX<T> Multiplex(std::vector<VectorX<T>>&& inputs) const;

  /* Splits the given vector into multiple vectors and returns the one with
   the given `index`.
   @throws std::exception if the size of `input` is not the sum of sizes
   provided at construction.
   @throws std::exception if index is not in [0, num_vectors).
   @returns a vector block of the indexed vector. */
  Eigen::Ref<const VectorX<T>> Demultiplex(
      const Eigen::Ref<const VectorX<T>>& input, int index) const;

  /* Mutable version of `Demultiplex()` that takes a pointer to a stacked
   vector. */
  Eigen::Ref<VectorX<T>> Demultiplex(EigenPtr<VectorX<T>> input,
                                     int index) const;

 private:
  std::vector<int> sizes_;
  std::vector<int> offsets_;
  /* The sum over `sizes_`. */
  int num_entries_{0};
};

template <typename T>
class DiscreteUpdateManager;

/* DeformableDriver is responsible for computing dynamics information about
 all deformable bodies. It works in tandem with a DeformableModel and a
 DiscreteUpdateManager that are provided at construction time. The deformable
 model informs the driver of modeling choices of the deformable bodies
 such as its Finite Element Model. The discrete update manager consumes the
 results of the computation performed by the driver and also provides
 information about the result of the world that the deformable bodies are
 interacting with. In particular, the manager provides access to MultibodyPlant.

 For any vertex in a deformable body, we say that it is "participating in
 contact and constraints" (or "participating" in short) if it is incident to a
 tetrahedron containing one or more contact points or explicitly specified as
 under constraint. We say a degree of freedom (dof) is "participating" if it
 belongs to a participating vertex. DeformableDriver reports participating
 quantities in increasing order of deformable body indexes. That is, the
 participating quantities of body 0 come first, followed by participating
 quantities of body 1 and so on. Within a single body, the participating
 vertices/dofs are ordered according to their associated vertex indexes.

 @tparam_double_only */
template <typename T>
class DeformableDriver : public ScalarConvertibleComponent<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DeformableDriver);

  /* Constructs a deformable driver that solves for the dynamics of the given
   `deformable_model`. The newly constructed driver is used in the given
   `manager` to perform discrete updates. The given `deformable_model` and
   `manager` must outlive this driver.
   @pre deformable_model != nullptr.
   @pre manager != nullptr. */
  DeformableDriver(const DeformableModel<T>* deformable_model,
                   const DiscreteUpdateManager<T>* manager);

  ~DeformableDriver() override;

  // NOTE (changyu): add for GPU MPM
  bool ExistsMpmBody() const { return deformable_model_->ExistsMpmModel(); }
  
  void CalcMpmContactPairs(
      const systems::Context<T>& context, gmpm::GpuMpmState<gmpm::config::GpuT> *mpm_state,
      gmpm::MpmParticleContactPairs<gmpm::config::GpuT>* result) const {
    using GpuT = gmpm::config::GpuT;
    DRAKE_ASSERT(result != nullptr);
    long long before_ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    result->clear();
    const geometry::QueryObject<T>& query_object =
        manager_->plant().get_geometry_query_input_port().template Eval<geometry::QueryObject<T>>(context);
    
    // NOTE (changyu): make sure the pose is up-to-date at the time performing the distance query.
    query_object.FullPoseUpdate();
    const MultibodyTreeTopology& tree_topology = manager_->internal_tree().get_topology();

    // loop over each particle
#if defined(_OPENMP)
#pragma omp parallel for num_threads(16)
#endif
    for (size_t p = 0; p < mpm_state->n_particles(); ++p) {
      // compute the distance of this particle with each geometry in file
      // NOTE (changyu): when access attributes in GpuMpmState,
      // always remember it is type GpuT and should be casted to type T explicitly.
      std::vector<geometry::SignedDistanceToPoint<T>> p_to_geometries =
          query_object.geometry_state().ComputeSignedDistanceToPoint(
            mpm_state->positions_host()[p].template cast<T>(), T(0));
      // identify those that are in contact, i.e. signed_distance < 0
      for (const auto& p2geometry : p_to_geometries) {
        if (p2geometry.distance < 0) {
          // if particle is inside rigid body, i.e. in contact
          // note: normal direction
          // NOTE (changyu): we treat each collision pair as an individual collision particle,
          // i.e., if one mpm particle has multiple collision pairs, it will be treated as
          // multiple collision particles and get independent impulse dv for each constraint.
          const Eigen::VectorBlock<const VectorX<T>>& v = manager_->plant().GetVelocities(context);
          const BodyIndex index_rigid =
            manager_->geometry_id_to_body_index().at(p2geometry.id_G);
          const TreeIndex tree_index_rigid =
              tree_topology.body_to_tree_index(index_rigid);
          Vector3<T> rigid_v = Vector3<T>::Zero();
          if (tree_index_rigid.is_valid()) {
            Matrix3X<T> Jv_v_WBc_W(3, manager_->plant().num_velocities());
            const Body<T>& rigid_body = manager_->plant().get_body(index_rigid);
            const Frame<T>& frame_W = manager_->plant().world_frame();
            manager_->internal_tree().CalcJacobianTranslationalVelocity(
                context, JacobianWrtVariable::kV, rigid_body.body_frame(), frame_W,
                mpm_state->positions_host()[p].template cast<T>(), frame_W, frame_W,
                &Jv_v_WBc_W);
            Matrix3X<T> J_rigid =
                Jv_v_WBc_W.middleCols(
                    tree_topology.tree_velocities_start_in_v(tree_index_rigid),
                    tree_topology.num_tree_velocities(tree_index_rigid)).template cast<T>();
            rigid_v = J_rigid * 
                        v.segment(tree_topology.tree_velocities_start_in_v(tree_index_rigid),
                                tree_topology.num_tree_velocities(tree_index_rigid))
                                .template cast<T>();
          }

          #if defined(_OPENMP)
          #pragma omp critical
          #endif
          {
            result->push_back(
                uint32_t(p), p2geometry.id_G.get_value(), GpuT(p2geometry.distance),
                -p2geometry.grad_W.normalized().template cast<GpuT>(),
                mpm_state->positions_host()[p],
                rigid_v.template cast<GpuT>());
          }
        }
      }
    }
    long long after_ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    printf("\033[32mcollision detection time=%lldms N(contacts)=%lu\033[0m\n", (after_ts - before_ts), result->size());
  }

  void InitalizeExternalContactForces(const systems::Context<T>& context, 
                       gmpm::GpuMpmState<gmpm::config::GpuT> *mpm_state) const {
    mpm_state->external_forces_host().resize(manager_->plant().num_bodies());
    for (size_t i = 0; i < mpm_state->external_forces_host().size(); ++i) {
      mpm_state->external_forces_host()[i].p_BoBq_B = 
        manager_->plant().EvalBodyPoseInWorld(
          context, manager_->plant().get_body(BodyIndex(i))).translation()
          .template cast<gmpm::config::GpuT>();
      mpm_state->external_forces_host()[i].F_Bq_W_tau = Vector3<gmpm::config::GpuT>::Zero();
      mpm_state->external_forces_host()[i].F_Bq_W_f = Vector3<gmpm::config::GpuT>::Zero();
    }
  }

  void FinalizeExternalContactForces(gmpm::GpuMpmState<gmpm::config::GpuT> *mpm_state, 
                       const gmpm::config::GpuT &dt) const {
    // Restore p_BoBq_B value and divide by dt to turn impulse into forces.
    for (size_t i = 0; i < mpm_state->external_forces_host().size(); ++i) {
      mpm_state->external_forces_host()[i].p_BoBq_B = Vector3<gmpm::config::GpuT>::Zero();
      mpm_state->external_forces_host()[i].F_Bq_W_tau /= dt;
      mpm_state->external_forces_host()[i].F_Bq_W_f /= dt;
    }
  }

  void CalcAbstractStates(const systems::Context<T>& context,
                          systems::State<T>* update) const {
    if (deformable_model_->ExistsMpmModel()) {
      using GpuT = gmpm::config::GpuT;
      gmpm::GpuMpmState<GpuT>& mutable_mpm_state = 
        update->template get_mutable_abstract_state<gmpm::GpuMpmState<GpuT>>(
            deformable_model_->gpu_mpm_state_index()
        );
      GpuT dt = static_cast<GpuT>(manager_->plant().time_step());

      // Dynamic Stage
      int current_frame = std::round(context.get_time() / dt);
      GpuT substep_dt = GpuT(deformable_model_->cpu_mpm_model().config.substep_dt);
      GpuT dt_left = dt;
      int substep = 0;
      long long before_ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      InitalizeExternalContactForces(context, &mutable_mpm_state);
      gmpm::MpmParticleContactPairs<GpuT> mpm_contact_pairs;

      while (dt_left > 0) {
        GpuT ddt = std::min(dt_left, substep_dt);
        dt_left -= ddt;
        mpm_solver_.RebuildMapping(&mutable_mpm_state, false);
        mpm_solver_.CalcFemStateAndForce(&mutable_mpm_state, ddt);
        mpm_solver_.ParticleToGrid(&mutable_mpm_state, ddt);
        // NOTE (changyu): update contact information at each substep for weak coupling scheme
        CalcMpmContactPairs(context, &mutable_mpm_state, &mpm_contact_pairs);
        mpm_solver_.CopyContactPairs(&mutable_mpm_state, mpm_contact_pairs);
        mpm_solver_.UpdateContact(&mutable_mpm_state, ddt, 
          deformable_model_->cpu_mpm_model().config.contact_friction_mu,
          deformable_model_->cpu_mpm_model().config.contact_stiffness, 
          deformable_model_->cpu_mpm_model().config.contact_damping);
        mpm_solver_.UpdateGrid(&mutable_mpm_state, deformable_model_->cpu_mpm_model().config.mpm_bc);
        mpm_solver_.GridToParticle(&mutable_mpm_state, ddt);
        mpm_solver_.SyncParticleStateToCpu(&mutable_mpm_state);
        substep += 1;
      }
      FinalizeExternalContactForces(&mutable_mpm_state, dt);

      // logging
      long long after_ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      // NOTE (changyu): time step info logging
      printf("\033[32mframe=%d time=%lldms N(substeps)=%d\033[0m\n", current_frame, (after_ts - before_ts), substep);
      if (deformable_model_->cpu_mpm_model().config.write_files) {
        mpm_solver_.Dump(mutable_mpm_state, "test" + std::to_string(current_frame) + ".obj");
      }
    }
  }

  int num_deformable_bodies() const { return deformable_model_->num_bodies(); }

  // TODO(xuchenhan-tri): Implement CloneToDouble() and allow cloning to double.
  bool is_cloneable_to_double() const final { return false; }
  bool is_cloneable_to_autodiff() const final { return false; }
  bool is_cloneable_to_symbolic() const final { return false; }

  /* Declares cache entries used by this DeformableDriver through the given
   manager.
   @pre `manager` is not nullptr and points to the same DiscreteUpdateManager
   provided at construction. */
  void DeclareCacheEntries(DiscreteUpdateManager<T>* manager);

  /* Evaluates the velocities of all participating dofs. See class documentation
   for how the velocities are ordered. */
  const VectorX<T>& EvalParticipatingVelocities(
      const systems::Context<T>& context) const;

  /* Evaluates the free motion velocities of all participating dofs. See class
   documentation for how the velocities are ordered. */
  const VectorX<T>& EvalParticipatingFreeMotionVelocities(
      const systems::Context<T>& context) const;

  /* Appends the linear dynamics matrices for participating dofs of each
   deformable body registered in this model to `A` in increasing order of
   deformable body indexes. The matrix corresponding to a body without any
   participating dof is empty.
   @pre A != nullptr. */
  void AppendLinearDynamicsMatrix(const systems::Context<T>& context,
                                  std::vector<MatrixX<T>>* A) const;

  /* Appends discrete contact pairs where at least one of the bodies in contact
   is deformable.
   @pre result != nullptr. */
  void AppendDiscreteContactPairs(
      const systems::Context<T>& context,
      DiscreteContactData<DiscreteContactPair<T>>* result) const;

  /* Appends the constraint kinematics information for each deformable rigid
   fixed constraint.
   @pre result != nullptr. */
  void AppendDeformableRigidFixedConstraintKinematics(
      const systems::Context<T>& context,
      std::vector<contact_solvers::internal::FixedConstraintKinematics<T>>*
          result) const;

  /* Computes the contact information for all deformable bodies for the given
   `context`.
   @pre contact_info != nullptr. */
  void CalcDeformableContactInfo(
      const systems::Context<T>& context,
      std::vector<DeformableContactInfo<T>>* contact_info) const;

  /* Evaluates FemState at the next time step for each deformable body and
   copies the them into the corresponding DiscreteValues.
   @pre next_states != nullptr. */
  void CalcDiscreteStates(const systems::Context<T>& context,
                          systems::DiscreteValues<T>* next_states) const;

  /* Evaluates the multiplexer for participating velocities for all bodies.
   @pre result != nullptr. */
  const Multiplexer<T>& EvalParticipatingVelocityMultiplexer(
      const systems::Context<T>& context) const;

  /* Evaluates the constraint participation information of the deformable body
   with the given `index`. See geometry::internal::ContactParticipation. */
  const geometry::internal::ContactParticipation& EvalConstraintParticipation(
      const systems::Context<T>& context, DeformableBodyIndex index) const;

 private:
  friend class DeformableDriverTest;
  friend class DeformableDriverContactTest;
  friend class DeformableDriverContactKinematicsTest;
  friend class DeformableIntegrationTest;

  /* Struct used to conglomerate the indexes of cache entries declared by
   the manager. */
  struct CacheIndexes {
    /* Per body cache entries indexed by DeformableBodyIndex. */
    std::vector<systems::CacheIndex> fem_states;
    std::vector<systems::CacheIndex> fem_solvers;
    std::vector<systems::CacheIndex> next_fem_states;
    systems::CacheIndex deformable_contact;
    std::vector<systems::CacheIndex> constraint_participations;
    std::vector<systems::CacheIndex> dof_permutations;
    std::unordered_map<geometry::GeometryId, systems::CacheIndex>
        vertex_permutations;
    systems::CacheIndex participating_velocity_mux;
    systems::CacheIndex participating_velocities;
    systems::CacheIndex participating_free_motion_velocities;
  };

  /* Struct to hold intermediate data from one of the two geometries in contact
   when computing DiscreteContactPair. */
  struct ContactData {
    /* The world frame position of the relative-to point for reporting the
     contact results. See DiscreteContactPair::p_ApC_W and
     DiscreteContactPair::p_BqC_W. `p_WG` is coincident with P and Q (and as
     they are all measured and expressed in the world frame, they will all
     have the same values). */
    Vector3<T> p_WG;
    /* Contact Jacobians for the kinematic tree corresponding to the object
     participating in the contact. `jacobian[i]` stores the contact Jacobian for
     the i-th contact point. This is empty if the geometry is rigid and welded
     to World. */
    std::vector<typename DiscreteContactPair<T>::JacobianTreeBlock> jacobian;
    /* Velocity (in the world frame) of the point Gc affixed to the geometry
     that is coincident with the contact point C. `v_WGc[i]` stores the
     world-frame velocity of the i-th contact point. This is empty if the
     geometry is rigid and welded to World. */
    std::vector<Vector3<T>> v_WGc;
    /* Name of the geometry in contact. */
    std::string name;
  };

  /* Computes the contact data for a deformable geometry G participating in
   contact.
   @param[in] context          Context of the MultibodyPlant owning this driver.
   @param[in] contact_surface  The contact surface between two geometries with
                               one of the geometries being geometry G.
   @param[in] is_A             True if geometry G is labeled as geometry A in
                               the given `contact_surface`. See class
                               documentation for
                               geometry::internal::DeformableContactSurface for
                               details. */
  ContactData ComputeContactDataForDeformable(
      const systems::Context<T>& context,
      const geometry::internal::DeformableContactSurface<T>& contact_surface,
      bool is_A) const;

  /* Computes the contact data for a rigid geometry G participating in contact.
   @param[in] context          Context of the MultibodyPlant owning this driver.
   @param[in] contact_surface  The contact surface between two geometries with
                               one of the geometries being geometry G.
   @note Unlike ComputeContactDataForDeformable where we need to determine
   whether geometry G is labeled as geometry A or B in DeformableContactSurface,
   by convention, a rigid geometry is always labeled as geometry B in
   DeformableContactSurface if it participates in deformable contact. */
  ContactData ComputeContactDataForRigid(
      const systems::Context<T>& context,
      const geometry::internal::DeformableContactSurface<T>& contact_surface)
      const;

  /* Copies the state of the deformable body with `id` in the given `context`
   to the `fem_state`.
   @pre fem_state != nullptr and has size compatible with the state of the
        deformable body with the given `index`.
   @pre `index` is valid and less than the number of deformable bodies. */
  void CalcFemState(const systems::Context<T>& context,
                    DeformableBodyIndex index,
                    fem::FemState<T>* fem_state) const;

  /* Eval version of CalcFemState(). */
  const fem::FemState<T>& EvalFemState(const systems::Context<T>& context,
                                       DeformableBodyIndex index) const;

  /* Given the state of the deformable body with `index` in the given `context`,
   computes its "free motion" state (the state the body would have at the next
   time step in the absence of contact or constraints) and the dependent
   Schur complement of the tangent matrix of the FEM model.
   @pre state_and_data != nullptr and is compatible with the FemModel associated
   with the deformable body with the given `index`. */
  void CalcFreeMotionFemSolver(const systems::Context<T>& context,
                               DeformableBodyIndex index,
                               fem::internal::FemSolver<T>* fem_solver) const;

  /* Eval version of CalcFreeMotionFemState(). */
  const fem::internal::FemSolver<T>& EvalFreeMotionFemSolver(
      const systems::Context<T>& context, DeformableBodyIndex index) const;

  const fem::FemState<T>& EvalFreeMotionFemState(
      const systems::Context<T>& context, DeformableBodyIndex index) const;

  const contact_solvers::internal::SchurComplement&
  EvalFreeMotionTangentMatrixSchurComplement(const systems::Context<T>& context,
                                             DeformableBodyIndex index) const;

  /* Given the state of the deformable body with `index` in the given `context`,
   computes the state of the deformable body at the next time step.
   @note The state of the deformable body will the same as the "free motion"
         state in the absence of contact or constraints. Otherwise, the discrete
         solver results for participating dofs are evaluated, and the Schur
         complement of the tangent matrix is used to update the
         non-participating dofs.
   @pre next_fem_state != nullptr and is compatible with the state of
        the deformable body with the given `index`. */
  void CalcNextFemState(const systems::Context<T>& context,
                        DeformableBodyIndex index,
                        fem::FemState<T>* next_fem_state) const;

  /* Eval version of CalcNextFemState(). */
  const fem::FemState<T>& EvalNextFemState(const systems::Context<T>& context,
                                           DeformableBodyIndex index) const;

  /* Computes the contact information for all registered deformable bodies
   @pre The geometry query input port of the MultibodyPlant that owns the
        manager associated with this DeformableDriver is connected.
   @pre result != nullptr. */
  void CalcDeformableContact(
      const systems::Context<T>& context,
      geometry::internal::DeformableContact<T>* result) const;

  /* Eval version of CalcDeformableContact(). */
  const geometry::internal::DeformableContact<T>& EvalDeformableContact(
      const systems::Context<T>& context) const;

  /* Calc version of EvalConstraintParticipation.
   @pre constraint_participation != nullptr. */
  void CalcConstraintParticipation(
      const systems::Context<T>& context, DeformableBodyIndex index,
      geometry::internal::ContactParticipation* constraint_participation) const;

  /* Computes the partial permutation that maps degrees of freedom of the
   deformable body with the given `index` to degrees of freedom that belong to
   vertices of the body that participate in contact.
   @pre result != nullptr. */
  void CalcDofPermutation(
      const systems::Context<T>& context, DeformableBodyIndex index,
      contact_solvers::internal::PartialPermutation* result) const;

  /* Eval version of CalcDofPermutation(). */
  const contact_solvers::internal::PartialPermutation& EvalDofPermutation(
      const systems::Context<T>& context, DeformableBodyIndex index) const;

  /* Computes the partial permutation that maps vertices of the
   deformable geometry with the given `id` to vertices that belong to
   vertices of the geometry that participate in contact.
   @pre result != nullptr. */
  void CalcVertexPermutation(
      const systems::Context<T>& context, geometry::GeometryId id,
      contact_solvers::internal::PartialPermutation* result) const;

  /* Eval version of CalcVertexPermutation(). */
  const contact_solvers::internal::PartialPermutation& EvalVertexPermutation(
      const systems::Context<T>& context, geometry::GeometryId id) const;

  /* Calc version of EvalParticipatingVelocityMultiplexer(). */
  void CalcParticipatingVelocityMultiplexer(const systems::Context<T>& context,
                                            Multiplexer<T>* result) const;

  /* Calc version of EvalParticipatingVelocities().
   @pre result != nullptr. */
  void CalcParticipatingVelocities(const systems::Context<T>& context,
                                   VectorX<T>* result) const;

  /* Calc version of EvalParticipatingFreeMotionVelocities().
   @pre result != nullptr. */
  void CalcParticipatingFreeMotionVelocities(const systems::Context<T>& context,
                                             VectorX<T>* result) const;

  CacheIndexes cache_indexes_;
  /* Modeling information about all deformable bodies. */
  const DeformableModel<T>* const deformable_model_;
  const DiscreteUpdateManager<T>* const manager_;
  /* The integrator used to advance deformable body free motion states in
   time. */
  std::unique_ptr<fem::internal::DiscreteTimeIntegrator<T>> integrator_;

  // NOTE (changyu): GPU MPM solver
  gmpm::GpuMpmSolver<gmpm::config::GpuT> mpm_solver_;
};

}  // namespace internal
}  // namespace multibody
}  // namespace drake
