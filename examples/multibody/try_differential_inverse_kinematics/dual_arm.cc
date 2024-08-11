#include <fstream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/meshcat_point_cloud_visualizer.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/meshcat_visualizer_params.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/manipulation/kuka_iiwa/iiwa_constants.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/inverse_kinematics/differential_inverse_kinematics_integrator.h"
#include "drake/multibody/mpm/constitutive_model/corotated_elastic_model.h"
#include "drake/multibody/mpm/constitutive_model/linear_corotated_model.h"
#include "drake/multibody/mpm/constitutive_model/linear_corotated_with_plasticity.h"
#include "drake/multibody/mpm/constitutive_model/stvk_hencky_with_von_mises_model.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/matrix_gain.h"
#include "drake/systems/primitives/multiplexer.h"

DEFINE_double(simulation_time, 3.5, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 10e-3,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 1e4, "Young's modulus of the deformable body [Pa].");
DEFINE_double(nu, 0.4, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 1e3, "Mass density of the deformable body [kg/m³].");
DEFINE_double(beta, 0.01,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_double(friction, 10.5, "mpm friction");
DEFINE_double(ppc, 2, "mpm ppc");
DEFINE_double(shift, 0.98, "shift");
DEFINE_double(damping, 1.0, "larger, more damping");

using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::ProximityProperties;
using drake::manipulation::kuka_iiwa::get_iiwa_max_joint_velocities;
using drake::math::RigidTransformd;
using drake::math::RotationMatrix;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::Body;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::DifferentialInverseKinematicsIntegrator;
using drake::multibody::DifferentialInverseKinematicsParameters;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::Parser;
using drake::systems::BasicVector;
using drake::systems::Context;
using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

namespace drake {
namespace examples {
namespace {

RigidTransformd FromXyzRpy(const Vector3<double>& rpy,
                           const Vector3<double>& p) {
  return RigidTransformd(math::RollPitchYaw<double>(rpy), p);
}

RigidTransformd FromXyzRpyDegree(const Vector3<double>& rpy_deg,
                                 const Vector3<double>& p) {
  return RigidTransformd(
      math::RollPitchYaw<double>(rpy_deg * 3.1415926 / 180.0), p);
}

class HandPoseController : public drake::systems::LeafSystem<double> {
 public:
  HandPoseController(const multibody::MultibodyPlant<double>& plant)
      : plant_(plant) {
    open_state_ = Eigen::VectorXd::Zero(4);
    open_state_(0) = -0.08;
    open_state_(1) = 0.08;
    closed_state_ = Eigen::VectorXd::Zero(4);
    closed_state_(0) = -0.032;
    closed_state_(1) = 0.032;
    this->DeclareVectorOutputPort(
        "WsgDesiredState", drake::systems::BasicVector<double>(size_),
        &HandPoseController::CalcDesiredState, {this->time_ticket()});
  }
  void CalcDesiredState(const Context<double>& context,
                        drake::systems::BasicVector<double>* output) const {
    if (context.get_time() < 0.5) {
      output->set_value(open_state_);
    } else if (context.get_time() < 2.4) {
      // gripper gripping from 0.5 to 0.8, then hold until 2.2
      double t = (context.get_time() - 0.5) / (0.3);
      Eigen::VectorXd q_and_v = std::max(1.0 - t, 0.0) * open_state_ +
                                std::min(t, 1.0) * closed_state_;
      output->set_value(q_and_v);
    } else if (context.get_time() < 2.7) {
      // gripper opening from 2.4 to 2.7
      double t = (context.get_time() - 2.4) / (0.3);
      Eigen::VectorXd q_and_v = std::max(1.0 - t, 0.0) * closed_state_ +
                                std::min(t, 1.0) * open_state_;
      output->set_value(q_and_v);
    } else {
      output->set_value(open_state_);
    }
  }

 private:
  const multibody::MultibodyPlant<double>& plant_;
  int size_ = 4;  // 4 fingers with 4 joints
  double close_start_ = 0.05;
  double close_end_ = 0.35;
  double open_start_ = 1.25;
  Eigen::VectorXd open_state_;
  Eigen::VectorXd closed_state_;
};

class IiwaController : public drake::systems::LeafSystem<double> {
 public:
  IiwaController(const multibody::MultibodyPlant<double>& plant, bool is_left,
                 const RigidTransformd& init_pose)
      : plant_(plant), is_left_(is_left) {
    robot_state_index_ =
        this->DeclareVectorInputPort("robot_state", 14).get_index();

    this->DeclareAbstractOutputPort("X_WG_desired", init_pose,
                                    &IiwaController::CalcOutput);
    VectorXd X_xyz_rpy = Eigen::VectorXd::Zero(6);
    // state = [ryp, translation]
    X_xyz_rpy.segment(0, 3) = init_pose.rotation().ToRollPitchYaw().vector();
    X_xyz_rpy.segment(3, 3) = init_pose.translation();

    this->DeclareDiscreteState(X_xyz_rpy);
    this->DeclarePeriodicDiscreteUpdateEvent(plant_.time_step(), 0,
                                             &IiwaController::Update);
  }

  const systems::InputPort<double>& robot_state_input_port() const {
    return this->get_input_port(robot_state_index_);
  }

  void CalcOutput(const Context<double>& context,
                  RigidTransformd* value) const {
    const VectorXd xd = context.get_discrete_state().value();
    // xd = [rpy, translation]
    *value = FromXyzRpy(xd.segment(0, 3), xd.segment(3, 3));
  }

  void Update(const Context<double>& context,
              systems::DiscreteValues<double>* next_states) const {
    const VectorX<double>& current_state_values =
        context.get_discrete_state().value();
    // fake update:
    VectorX<double> dX = current_state_values;
    dX.setZero();

    if ((context.get_time() >= 0.0) && (context.get_time() <= 0.2)) {
      dX(5) = 0.006;  // up
    } else if (context.get_time() <= 0.3) {
      dX.setZero();  // hold
    } else if (context.get_time() <= 0.5) {
      dX(5) = -0.006;  // down
    } else if (context.get_time() <= 0.8) {
      dX.setZero();  // hold, grip is gripping from 0.5 to 0.8
    } else if (context.get_time() <= 1.2) {
      dX(5) = 0.003;  // up for 0.4s
    } else if (context.get_time() <= 1.3) {
      dX.setZero();  // hold
    } else if (context.get_time() <= 1.9) {
      if (is_left_) {
        dX(4) = 0.0025;  // split
      } else {
        dX(4) = -0.0025;  // split
      }
    } else if (context.get_time() <= 2.0) {
      dX.setZero();  // hold
    } else if (context.get_time() <= 2.2) {
      dX(5) = -0.004;  // down for 0.2s
    } else if (context.get_time() <= 2.8) {
      dX.setZero();  // hold
    } else if (context.get_time() <= 3.2) {
      dX(5) = 0.006;  // up
    } else {
      dX.setZero();  // hold
    }

    // if ((context.get_time() >= lift_start_) &&
    //     (context.get_time() <= lift_ends_)) {
    //   dX(5) = 0.004;
    // }
    // if ((context.get_time() >= split_start_) &&
    //     (context.get_time() <= split_end_)) {
    //   if (is_left_) {
    //     dX(4) = 0.004;  // split
    //   } else {
    //     dX(4) = -0.004;  // split
    //   }
    // }

    auto new_value = current_state_values + dX;
    next_states->set_value(new_value);
  }

 private:
  const multibody::MultibodyPlant<double>& plant_;
  int robot_state_index_{};
  bool is_left_;
  double preparation_time = 0.5;  // 0.2 up, 0.1 hold, 0.2 down
  double lift_start_ = 0.3;
  double lift_ends_ = 0.45;
  double split_start_ = 0.5;
  double split_end_ = 0.9;
};

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  plant_config.discrete_contact_approximation = "lagged";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  bool use_mpm_ground = false;
  if (!use_mpm_ground) {
    /* Set up a ground. */
    ProximityProperties rigid_proximity_props;
    /* Set the friction coefficient close to that of rubber against rubber. */
    const CoulombFriction<double> surface_friction(0.0, 0.0);
    AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
    rigid_proximity_props.AddProperty(geometry::internal::kHydroGroup,
                                      geometry::internal::kRezHint, 0.01);
    Box ground{10, 10, 10};
    const RigidTransformd X_WG(Eigen::Vector3d{0, 0, -5});
    plant.RegisterCollisionGeometry(plant.world_body(), X_WG, ground,
                                    "ground_collision", rigid_proximity_props);
  }

  multibody::Parser ground_parser(&plant, "ground");
  std::string visual_ground_filename = FindResourceOrThrow(
      "drake/examples/manipulation_station/models/table_wide.sdf");
  auto table = ground_parser.AddModels(visual_ground_filename)[0];

  // plant.mutable_gravity_field().set_gravity_vector(Eigen::Vector3d::Zero());
  multibody::Parser left_parser(&plant, "left");
  multibody::Parser right_parser(&plant, "right");

  const std::string iiwa_filename = FindResourceOrThrow(
      "drake/manipulation/models/"
      "iiwa_description/iiwa7/iiwa7_no_collision.sdf");
  auto left_iiwa = left_parser.AddModels(iiwa_filename)[0];
  auto right_iiwa = right_parser.AddModels(iiwa_filename)[0];

  MultibodyPlant<double> left_iiwa_controller_plant =
      MultibodyPlant<double>(plant_config.time_step);
  Parser(&left_iiwa_controller_plant).AddModels(iiwa_filename);
  MultibodyPlant<double> right_iiwa_controller_plant =
      MultibodyPlant<double>(plant_config.time_step);
  Parser(&right_iiwa_controller_plant).AddModels(iiwa_filename);

  std::string hand_filename = FindResourceOrThrow(
      "drake/manipulation/models/wsg_50_description/sdf/"
      "schunk_wsg_50_simon.sdf");
  auto left_wsg = left_parser.AddModels(hand_filename)[0];
  auto right_wsg = right_parser.AddModels(hand_filename)[0];

  RigidTransformd left_iiwa_position(Eigen::Vector3d(0, 0.58, 0));
  RigidTransformd right_iiwa_position =
      FromXyzRpyDegree(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0));
  plant.WeldFrames(plant.world_frame(),
                   plant.GetBodyByName("table_body", table).body_frame(),
                   RigidTransformd(Eigen::Vector3d(0.25, 0.29, 0)));
  plant.WeldFrames(plant.world_frame(),
                   plant.GetBodyByName("iiwa_link_0", left_iiwa).body_frame(),
                   left_iiwa_position);
  plant.WeldFrames(plant.world_frame(),
                   plant.GetBodyByName("iiwa_link_0", right_iiwa).body_frame(),
                   right_iiwa_position);
  left_iiwa_controller_plant.WeldFrames(
      left_iiwa_controller_plant.world_frame(),
      left_iiwa_controller_plant.GetBodyByName("iiwa_link_0").body_frame(),
      left_iiwa_position);
  right_iiwa_controller_plant.WeldFrames(
      right_iiwa_controller_plant.world_frame(),
      right_iiwa_controller_plant.GetBodyByName("iiwa_link_0").body_frame(),
      right_iiwa_position);
  plant.WeldFrames(
      plant.GetBodyByName("iiwa_link_7", left_iiwa).body_frame(),
      plant.GetBodyByName("body", left_wsg).body_frame(),
      FromXyzRpyDegree(Eigen::Vector3d(90, 0, 0), Eigen::Vector3d(0, 0, 0.07)));
  plant.WeldFrames(
      plant.GetBodyByName("iiwa_link_7", right_iiwa).body_frame(),
      plant.GetBodyByName("body", right_wsg).body_frame(),
      FromXyzRpyDegree(Eigen::Vector3d(90, 0, 0), Eigen::Vector3d(0, 0, 0.07)));
  // mpm stuff
  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);
  const DeformableModel<double>* deformable_model =
      owned_deformable_model.get();  // for drake viz originally
  std::unique_ptr<drake::multibody::mpm::internal::AnalyticLevelSet>
      mpm_geometry_level_set =
          std::make_unique<drake::multibody::mpm::internal::BoxLevelSet>(
              Vector3<double>(0.048, 0.16, 0.042));
  std::unique_ptr<
      drake::multibody::mpm::constitutive_model::ElastoPlasticModel<double>>
      model = std::make_unique<drake::multibody::mpm::constitutive_model::
                                   LinearCorotatedWithPlasticity<double>>(
          10e4, 0.2, 6e3);

  // std::unique_ptr<
  //       drake::multibody::mpm::constitutive_model::ElastoPlasticModel<double>>
  //       model = std::make_unique<drake::multibody::mpm::constitutive_model::
  //                                    LinearCorotatedModel<double>>(
  //           1e5, 0.2);

  Vector3<double> translation = {0.555 + 0.0, 0.29, 0.042};
  std::unique_ptr<math::RigidTransform<double>> pose =
      std::make_unique<math::RigidTransform<double>>(translation);
  double h = 0.015 * 1.5;
  owned_deformable_model->RegisterMpmBody(std::move(mpm_geometry_level_set),
                                          std::move(model), std::move(pose),
                                          1000.0, h);
  if (use_mpm_ground) {
    owned_deformable_model->ApplyMpmGround();
  }

  owned_deformable_model->SetMpmMinParticlesPerCell(
      static_cast<int>(FLAGS_ppc));
  owned_deformable_model->SetMpmFriction(FLAGS_friction);
  owned_deformable_model->SetMpmDamping(FLAGS_damping);
  owned_deformable_model->SetMpmStiffness(1e6);
  plant.AddPhysicalModel(std::move(owned_deformable_model));

  double Kp = 1e6;
  double Kd = 2 * std::sqrt(Kp);

  drake::multibody::PdControllerGains gain(Kp, Kd);
  for (int i = 0; i < plant.num_actuators(); ++i) {
    plant.get_mutable_joint_actuator(drake::multibody::JointActuatorIndex(i))
        .set_controller_gains(gain);
  }

  plant.Finalize();
  left_iiwa_controller_plant.Finalize();
  right_iiwa_controller_plant.Finalize();

  Eigen::VectorXd right_iiwa_initial_joint_values(7);
  right_iiwa_initial_joint_values << 0.3, 0.9, 0, -1.5, 0, 0.75, 0.30;

  Eigen::VectorXd left_iiwa_initial_joint_values(7);
  left_iiwa_initial_joint_values << -0.3, 0.9, 0, -1.5, 0, 0.75, -0.30;

  // hand controller
  auto left_hand_pose_controller =
      builder.template AddSystem<HandPoseController>(
          plant);  // put gripper timing here

  auto left_actuated_states_selector =
      builder.template AddSystem<drake::systems::MatrixGain<double>>(4);

  auto right_hand_pose_controller =
      builder.template AddSystem<HandPoseController>(
          plant);  // put gripper timing here

  auto right_actuated_states_selector =
      builder.template AddSystem<drake::systems::MatrixGain<double>>(4);

  builder.Connect(left_hand_pose_controller->get_output_port(),
                  left_actuated_states_selector->get_input_port());
  builder.Connect(left_actuated_states_selector->get_output_port(),
                  plant.get_desired_state_input_port(left_wsg));

  builder.Connect(right_hand_pose_controller->get_output_port(),
                  right_actuated_states_selector->get_input_port());
  builder.Connect(right_actuated_states_selector->get_output_port(),
                  plant.get_desired_state_input_port(right_wsg));

  // iiwa controller
  // first find the initial pose for the current joint values
  std::unique_ptr<Context<double>> left_temp_context =
      plant.CreateDefaultContext();
  std::unique_ptr<Context<double>> right_temp_context =
      plant.CreateDefaultContext();
  plant.SetPositions(left_temp_context.get(), left_iiwa,
                     left_iiwa_initial_joint_values);
  plant.SetPositions(right_temp_context.get(), right_iiwa,
                     right_iiwa_initial_joint_values);
  RigidTransformd left_iiwa_controller_initial_pose =
      plant.GetBodyByName("iiwa_link_7", left_iiwa)
          .body_frame()
          .CalcPoseInWorld(*(left_temp_context.get()));
  RigidTransformd right_iiwa_controller_initial_pose =
      plant.GetBodyByName("iiwa_link_7", right_iiwa)
          .body_frame()
          .CalcPoseInWorld(*(right_temp_context.get()));

  auto left_iiwa_controller = builder.template AddSystem<IiwaController>(
      plant, true, left_iiwa_controller_initial_pose);
  auto right_iiwa_controller = builder.template AddSystem<IiwaController>(
      plant, false, right_iiwa_controller_initial_pose);
  int nq_iiwa = plant.num_positions(left_iiwa);
  int nv_iiwa = plant.num_velocities(left_iiwa);
  int nu_iiwa = plant.num_actuated_dofs(left_iiwa);
  unused(nu_iiwa);

  DifferentialInverseKinematicsParameters left_params(nq_iiwa, nv_iiwa);
  left_params.set_nominal_joint_position(left_iiwa_initial_joint_values);

  DifferentialInverseKinematicsParameters right_params(nq_iiwa, nv_iiwa);
  right_params.set_nominal_joint_position(left_iiwa_initial_joint_values);

  auto left_diff_ik =
      builder.template AddSystem<DifferentialInverseKinematicsIntegrator>(
          left_iiwa_controller_plant,
          left_iiwa_controller_plant.GetFrameByName("iiwa_link_7"),
          plant_config.time_step, left_params);
  auto right_diff_ik =
      builder.template AddSystem<DifferentialInverseKinematicsIntegrator>(
          right_iiwa_controller_plant,
          right_iiwa_controller_plant.GetFrameByName("iiwa_link_7"),
          plant_config.time_step, right_params);
  std::vector<int> input_sizes;
  input_sizes.push_back(nq_iiwa);
  input_sizes.push_back(nv_iiwa);
  auto left_mux =
      builder.template AddSystem<drake::systems::Multiplexer<double>>(
          input_sizes);
  auto right_mux =
      builder.template AddSystem<drake::systems::Multiplexer<double>>(
          input_sizes);

  auto zero_vs =
      builder.template AddSystem<drake::systems::ConstantVectorSource>(
          Eigen::VectorXd::Zero(nv_iiwa));
  builder.Connect(plant.get_state_output_port(left_iiwa),
                  left_iiwa_controller->robot_state_input_port());
  builder.Connect(plant.get_state_output_port(right_iiwa),
                  right_iiwa_controller->robot_state_input_port());

  builder.Connect(left_iiwa_controller->get_output_port(),
                  left_diff_ik->GetInputPort("X_WE_desired"));
  builder.Connect(right_iiwa_controller->get_output_port(),
                  right_diff_ik->GetInputPort("X_WE_desired"));
  builder.Connect(plant.get_state_output_port(left_iiwa),
                  left_diff_ik->GetInputPort("robot_state"));
  builder.Connect(plant.get_state_output_port(right_iiwa),
                  right_diff_ik->GetInputPort("robot_state"));

  builder.Connect(left_diff_ik->GetOutputPort("joint_positions"),
                  left_mux->get_input_port(0));
  builder.Connect(right_diff_ik->GetOutputPort("joint_positions"),
                  right_mux->get_input_port(0));
  builder.Connect(zero_vs->get_output_port(), left_mux->get_input_port(1));
  builder.Connect(zero_vs->get_output_port(), right_mux->get_input_port(1));

  builder.Connect(left_mux->get_output_port(),
                  plant.get_desired_state_input_port(left_iiwa));
  builder.Connect(right_mux->get_output_port(),
                  plant.get_desired_state_input_port(right_iiwa));
  
  geometry::DrakeVisualizerParams visualize_params;
  visualize_params.publish_period = 1.0 / 100;
  auto& visualizer = geometry::DrakeVisualizerd::AddToBuilder(
      &builder, scene_graph, nullptr, visualize_params);
  // connect mpm to output port
  builder.Connect(deformable_model->mpm_particle_positions_port(),
                  visualizer.mpm_data_input_port());

  // meshcat viz
  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  auto meshcat_params = drake::geometry::MeshcatVisualizerParams();
  meshcat_params.publish_period = FLAGS_time_step * 2;
  drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, scene_graph, meshcat, meshcat_params);
  auto meshcat_pc_visualizer =
      builder.AddSystem<drake::geometry::MeshcatPointCloudVisualizer>(
          meshcat, "cloud", meshcat_params.publish_period);
  meshcat_pc_visualizer->set_point_size(0.005);
  builder.Connect(deformable_model->mpm_point_cloud_port(),
                  meshcat_pc_visualizer->cloud_input_port());

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  auto& mutable_context = simulator.get_mutable_context();
  auto& plant_context = plant.GetMyMutableContextFromRoot(&mutable_context);

  plant.SetPositions(&plant_context, left_iiwa, left_iiwa_initial_joint_values);
  plant.SetPositions(&plant_context, right_iiwa,
                     right_iiwa_initial_joint_values);
  plant.SetPositions(&plant_context, left_wsg, Eigen::Vector2d(-0.08, 0.08));
  plant.SetPositions(&plant_context, right_wsg, Eigen::Vector2d(-0.08, 0.08));

  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);

  sleep(6);

  bool recording = true;

  if (recording) {
    meshcat->StartRecording();
    simulator.AdvanceTo(FLAGS_simulation_time);
    meshcat->StopRecording();
    meshcat->PublishRecording();

  } else {
    simulator.AdvanceTo(FLAGS_simulation_time);
  }

  std::ofstream htmlFile("output.html");
  htmlFile << meshcat->StaticHtml();
  htmlFile.close();

  return 0;
}

}  // namespace
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is a demo used to showcase deformable body simulations in Drake. "
      "A simple parallel gripper grasps a deformable torus on the ground, "
      "lifts it up, and then drops it back on the ground. "
      "Launch meldis before running this example. "
      "Refer to README for instructions on meldis as well as optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}
