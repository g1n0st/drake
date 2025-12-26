#include <math.h>

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
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/mpm/constitutive_model/corotated_elastic_model.h"
#include "drake/multibody/mpm/constitutive_model/linear_corotated_model.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/unit_inertia.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 5.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 1e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 5e5, "Young's modulus of the deformable body [Pa].");
DEFINE_double(rho, 100, "density.");
DEFINE_double(nu, 0.4, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 1e3,
              "Mass density of the deformable body [kg/m³]. We observe that "
              "density above 2400 kg/m³ makes the torus too heavy to be picked "
              "up by the suction gripper.");
DEFINE_double(beta, 0.01,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_double(hydro_modulus, 1e8, "Hydroelastic modulus [Pa].");
DEFINE_double(damping, 1e2, "H&C damping.");
DEFINE_double(ppc, 11, "mpm ppc");

using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::Capsule;
using drake::geometry::Ellipsoid;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::geometry::Sphere;
using drake::math::RigidTransformd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::Parser;
using drake::multibody::PrismaticJoint;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::BasicVector;
using drake::systems::Context;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

namespace drake {
namespace examples {
namespace {

class DummyZBoxController : public drake::systems::LeafSystem<double> {
 public:
  DummyZBoxController(const multibody::MultibodyPlant<double>& plant,
                      double initial_height, double box_width)
      : plant_(plant) {
    this->DeclareVectorOutputPort(
        "DummyZBoxDesiredState", drake::systems::BasicVector<double>(2),
        &DummyZBoxController::CalcDesiredState, {this->time_ticket()});
    initial_height_ = initial_height;
    box_width_ = box_width;
    target_z_displacement_ = target_z_displacement_ * box_width;
  }
  void CalcDesiredState(const Context<double>& context,
                        drake::systems::BasicVector<double>* output) const {
    unused(context);
    unused(output);
    Vector2<double> state_value;
    double target_z_pos = initial_height_;
    if (context.get_time() > lift_start_) {
      double fraction = (context.get_time() - lift_start_) / lift_duration_;
      target_z_pos =
          initial_height_ + std::min(fraction, 1.0) * target_z_displacement_;

      if ((context.get_time() > shake_start_) &&
          (context.get_time() < shake_start_ + delta_t_)) {
        fraction = (context.get_time() - shake_start_) / delta_t_;
        target_z_pos -= fraction * 0.18 * box_width_;
      }
      if ((context.get_time() >= shake_start_ + delta_t_) &&
          (context.get_time() < shake_start_ + 3 * delta_t_)) {
        fraction =
            1 - (context.get_time() - (shake_start_ + delta_t_)) / delta_t_;
        target_z_pos -= fraction * 0.18 * box_width_;
      }
      if ((context.get_time() >= shake_start_ + 3 * delta_t_) &&
          (context.get_time() < shake_start_ + 5 * delta_t_)) {
        fraction =
            1 - (context.get_time() - (shake_start_ + 3 * delta_t_)) / delta_t_;
        target_z_pos += fraction * 0.18 * box_width_;
      }
      if ((context.get_time() >= shake_start_ + 5 * delta_t_) &&
          (context.get_time() < shake_start_ + 7 * delta_t_)) {
        fraction =
            1 - (context.get_time() - (shake_start_ + 5 * delta_t_)) / delta_t_;
        target_z_pos -= fraction * 0.18 * box_width_;
      }
      if ((context.get_time() >= shake_start_ + 7 * delta_t_) &&
          (context.get_time() < shake_start_ + 9 * delta_t_)) {
        fraction =
            1 - (context.get_time() - (shake_start_ + 7 * delta_t_)) / delta_t_;
        target_z_pos += fraction * 0.18 * box_width_;
      }
      if ((context.get_time() >= shake_start_ + 9 * delta_t_) &&
          (context.get_time() < shake_start_ + 11 * delta_t_)) {
        fraction =
            1 - (context.get_time() - (shake_start_ + 9 * delta_t_)) / delta_t_;
        target_z_pos -= fraction * 0.18 * box_width_;
      }
      if ((context.get_time() >= shake_start_ + 11 * delta_t_) &&
          (context.get_time() < shake_start_ + 13 * delta_t_)) {
        fraction = 1 - (context.get_time() - (shake_start_ + 11 * delta_t_)) /
                           delta_t_;
        target_z_pos += fraction * 0.18 * box_width_;
      }
      if ((context.get_time() >= shake_start_ + 13 * delta_t_) &&
          (context.get_time() < shake_start_ + 15 * delta_t_)) {
        fraction = 1 - (context.get_time() - (shake_start_ + 13 * delta_t_)) /
                           delta_t_;
        target_z_pos -= fraction * 0.18 * box_width_;
      }
      if ((context.get_time() >= shake_start_ + 15 * delta_t_) &&
          (context.get_time() < shake_start_ + 16 * delta_t_)) {
        fraction = 1 - (context.get_time() - (shake_start_ + 15 * delta_t_)) /
                           delta_t_;
        target_z_pos += fraction * 0.18 * box_width_;
      }
    }
    state_value << target_z_pos, 0;
    output->set_value(state_value);
  }

 private:
  const multibody::MultibodyPlant<double>& plant_;
  double initial_height_ = 0.0;
  double lift_start_ = 0.4;
  double lift_duration_ = 1.6;
  double target_z_displacement_ = 2.0;
  double box_width_;
  double shake_start_ = 2.2;
  double delta_t_ = 0.08;
};

class XBoxController : public drake::systems::LeafSystem<double> {
 public:
  XBoxController(const multibody::MultibodyPlant<double>& plant, bool is_right,
                 double initial_pos, double box_width)
      : plant_(plant) {
    this->DeclareVectorOutputPort(
        "XBoxDesiredState", drake::systems::BasicVector<double>(2),
        &XBoxController::CalcDesiredState, {this->time_ticket()});
    is_right_ = is_right;
    initial_pos_ = initial_pos;
    box_width_ = box_width;
    target_movement_ = target_movement_ * box_width;
  }
  void CalcDesiredState(const Context<double>& context,
                        drake::systems::BasicVector<double>* output) const {
    unused(context);
    Vector2<double> state_value;
    double target_x_pos = initial_pos_;
    if (context.get_time() > move_start_) {
      double fraction = (context.get_time() - move_start_) / move_duration_;
      double dx = std::min(fraction, 1.0) * target_movement_;

      if (is_right_) {
        target_x_pos -= dx;
      } else {
        target_x_pos += dx;
      }
    }
    state_value << target_x_pos, 0;
    output->set_value(state_value);
  }

 private:
  const multibody::MultibodyPlant<double>& plant_;
  double initial_pos_;
  bool is_right_;
  double move_start_ = 0.0;
  double move_duration_ = 0.4;
  double target_movement_ = 1.0;
  double box_width_;
};

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;

  plant_config.discrete_contact_approximation = "lagged";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  ProximityProperties compliant_hydro_props;
  ProximityProperties rigid_hydro_props;

  const CoulombFriction<double> surface_friction(1.0, 1.0);
  AddContactMaterial(FLAGS_damping, {}, surface_friction,
                     &compliant_hydro_props);
  AddContactMaterial(FLAGS_damping, {}, surface_friction, &rigid_hydro_props);

  AddCompliantHydroelasticProperties(0.01, FLAGS_hydro_modulus,
                                     &compliant_hydro_props);
  AddRigidHydroelasticProperties(0.01, &rigid_hydro_props);
  /* Set up a ground. */
  Box ground{20, 20, 10};
  const RigidTransformd X_WG(Eigen::Vector3d{0, 0, -5});
  plant.RegisterCollisionGeometry(plant.world_body(), X_WG, ground,
                                  "ground_collision", rigid_hydro_props);
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse",
                                 Vector4d(0.9, 0.9, 0.9, 1.0));
  plant.RegisterVisualGeometry(plant.world_body(), X_WG, ground,
                               "ground_visual", std::move(illustration_props));

  double box_width = 0.4 / 4;
  double ratio = 150.0;

  // a dummy box for lifting in z-direction
  const drake::multibody::UnitInertia<double> unit_inertia(0, 0, 0);
  const SpatialInertia<double> zero_inertia =
      SpatialInertia<double>(0.0, Vector3<double>(0, 0, 0), unit_inertia);
  ModelInstanceIndex dummy_z_instance =
      plant.AddModelInstance("dummy_z_instance");
  const RigidBody<double>& dummy_z_body =
      plant.AddRigidBody("dummy_z_body", dummy_z_instance, zero_inertia);
  const auto& prismatic_joint_z = plant.AddJoint<PrismaticJoint>(
      "translate_z_joint", plant.world_body(), RigidTransformd(), dummy_z_body,
      std::nullopt, Vector3d::UnitZ());
  plant.GetMutableJointByName<PrismaticJoint>("translate_z_joint")
      .set_default_translation(box_width / 2.0);
  auto dummy_z_box_controller = builder.template AddSystem<DummyZBoxController>(
      plant, box_width / 2.0, box_width);
  const auto actuator_z_index =
      plant.AddJointActuator("z prismatic joint actuator", prismatic_joint_z)
          .index();
  plant.get_mutable_joint_actuator(actuator_z_index)
      .set_controller_gains({1e7, 1});

  // box controlled on the left
  ModelInstanceIndex left_box_model_instance =
      plant.AddModelInstance("left_box_instance");
  const SpatialInertia<double> left_box_spatial =
      SpatialInertia<double>::SolidBoxWithDensity(FLAGS_rho, box_width / 6.0,
                                                  box_width * 1.4, box_width * 1.0);
  const RigidBody<double>& left_box =
      plant.AddRigidBody("left_box", left_box_model_instance, left_box_spatial);
  const auto& left_prismatic_joint_x = plant.AddJoint<PrismaticJoint>(
      "left_translate_x_joint", dummy_z_body, RigidTransformd(), left_box,
      std::nullopt, Vector3d::UnitX());
  plant.GetMutableJointByName<PrismaticJoint>("left_translate_x_joint")
      .set_default_translation(-(1.5 + 0.5 / 6.0) * box_width);
  const auto left_actuator_x_index =
      plant.AddJointActuator("left x actuator", left_prismatic_joint_x).index();
  double stiffness = (2.0 + ratio) / 0.1 * 3;
  plant.get_mutable_joint_actuator(left_actuator_x_index)
      .set_controller_gains({stiffness, 1});
  auto left_box_controller = builder.template AddSystem<XBoxController>(
      plant, false, -(1.5 + 0.5 / 6.0) * box_width, box_width);

  // box controlled on the right
  ModelInstanceIndex right_box_model_instance =
      plant.AddModelInstance("right_box_instance");
  const SpatialInertia<double> right_box_spatial =
      SpatialInertia<double>::SolidBoxWithDensity(FLAGS_rho, box_width / 4.0,
                                                  box_width * 1.4, box_width * 1.0);
  const RigidBody<double>& right_box = plant.AddRigidBody(
      "right_box", right_box_model_instance, right_box_spatial);
  const auto& right_prismatic_joint_x = plant.AddJoint<PrismaticJoint>(
      "right_translate_x_joint", dummy_z_body, RigidTransformd(), right_box,
      std::nullopt, Vector3d::UnitX());
  plant.GetMutableJointByName<PrismaticJoint>("right_translate_x_joint")
      .set_default_translation((1.5 + 0.5 / 6.0) * box_width);
  const auto right_actuator_x_index =
      plant.AddJointActuator("right x actuator", right_prismatic_joint_x)
          .index();
  plant.get_mutable_joint_actuator(right_actuator_x_index)
      .set_controller_gains({stiffness, 1});
  auto right_box_controller = builder.template AddSystem<XBoxController>(
      plant, true, (1.5 + 0.5 / 6.0) * box_width, box_width);

  unused(left_prismatic_joint_x, right_prismatic_joint_x);

  ModelInstanceIndex free_body_model_instance =
      plant.AddModelInstance("free_body_instance");
  const SpatialInertia<double> free_body_box_spatial =
      SpatialInertia<double>::SolidBoxWithDensity(FLAGS_rho * ratio, box_width,
                                                  box_width, box_width);
  const RigidBody<double>& free_box = plant.AddRigidBody(
      "free_box", free_body_model_instance, free_body_box_spatial);

  const Vector4<double> light_blue(0.5, 0.8, 1.0, 1.0);
  const Vector4<double> red(1.0, 0.0, 0.0, 1.0);
  const Vector4<double> green(0.0, 1.0, 0.0, 1.0);
  const Vector4<double> blue(0.0, 0.0, 1.0, 1.0);
  const Vector4<double> dark_blue(0.0, 0.0, 0.8, 1.0);
  const Vector4<double> orange(1.0, 0.55, 0.0, 0.2);
  const Vector4<double> grey(0.5, 0.5, 0.5, 1.0);
  unused(light_blue, red, green, blue, dark_blue, orange);

  plant.RegisterVisualGeometry(left_box, RigidTransformd::Identity(),
                               Box(box_width / 6.0, box_width * 1.4, box_width * 1.0),
                               "LeftCubeV", grey);
  plant.RegisterCollisionGeometry(
      left_box, RigidTransformd::Identity(),
      Box(box_width / 6.0, box_width * 1.4, box_width), "LeftCube",
      compliant_hydro_props);
  plant.RegisterVisualGeometry(right_box, RigidTransformd::Identity(),
                               Box(box_width / 6.0, box_width * 1.4, box_width * 1.0),
                               "RightCubeV", grey);
  plant.RegisterCollisionGeometry(
      right_box, RigidTransformd::Identity(),
      Box(box_width / 6.0, box_width * 1.4, box_width), "RightCube",
      compliant_hydro_props);

  plant.RegisterVisualGeometry(free_box, RigidTransformd::Identity(),
                               Box(box_width, box_width, box_width),
                               "FreeCubeV", red);
  plant.RegisterCollisionGeometry(free_box, RigidTransformd::Identity(),
                                  Box(box_width, box_width, box_width),
                                  "FreeCube", compliant_hydro_props);

  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);

  // set a MPM body
  std::unique_ptr<drake::multibody::mpm::internal::AnalyticLevelSet>
      mpm_geometry_level_set1 =
          std::make_unique<drake::multibody::mpm::internal::BoxLevelSet>(
              Vector3<double>(box_width / 2.0, box_width / 2.0,
                              box_width / 2.0));

  std::unique_ptr<drake::multibody::mpm::internal::AnalyticLevelSet>
      mpm_geometry_level_set2 =
          std::make_unique<drake::multibody::mpm::internal::BoxLevelSet>(
              Vector3<double>(box_width / 2.0, box_width / 2.0,
                              box_width / 2.0));

  std::unique_ptr<
      drake::multibody::mpm::constitutive_model::ElastoPlasticModel<double>>
      model1 =
          std::make_unique<drake::multibody::mpm::constitutive_model::
                               LinearCorotatedModel<double>>(FLAGS_E, FLAGS_nu);

  std::unique_ptr<
      drake::multibody::mpm::constitutive_model::ElastoPlasticModel<double>>
      model2 =
          std::make_unique<drake::multibody::mpm::constitutive_model::
                               LinearCorotatedModel<double>>(FLAGS_E, FLAGS_nu);

  std::unique_ptr<math::RigidTransform<double>> pose1 =
      std::make_unique<math::RigidTransform<double>>(
          Vector3<double>(-1.0 * box_width, 0.0, box_width / 2.0));

  std::unique_ptr<math::RigidTransform<double>> pose2 =
      std::make_unique<math::RigidTransform<double>>(
          Vector3<double>(1.0 * box_width, 0.0, box_width / 2.0));

  double h = box_width / 4.0;

  owned_deformable_model->RegisterMpmBody(std::move(mpm_geometry_level_set1),
                                          std::move(model1), std::move(pose1),
                                          FLAGS_rho, h);

  owned_deformable_model->RegisterAdditionalMpmBody(
      std::move(mpm_geometry_level_set2), std::move(model2), std::move(pose2),
      FLAGS_rho, h);

  owned_deformable_model->SetMpmDamping(10.0);
  owned_deformable_model->SetMpmStiffness(1e6);
  owned_deformable_model->SetMpmFriction(0.8);
  owned_deformable_model->SetMpmMinParticlesPerCell(
      static_cast<int>(FLAGS_ppc));
  owned_deformable_model->SetIntegrator(drake::multibody::mpm::MpmIntegratorType::Substep);
  owned_deformable_model->SetSubstepCount(100);

  const DeformableModel<double>* deformable_model =
      owned_deformable_model.get();
  plant.AddPhysicalModel(std::move(owned_deformable_model));

  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  builder.Connect(dummy_z_box_controller->get_output_port(),
                  plant.get_desired_state_input_port(dummy_z_instance));

  builder.Connect(left_box_controller->get_output_port(),
                  plant.get_desired_state_input_port(left_box_model_instance));

  builder.Connect(right_box_controller->get_output_port(),
                  plant.get_desired_state_input_port(right_box_model_instance));

  unused(right_box_controller);

  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  auto meshcat_params = drake::geometry::MeshcatVisualizerParams();
  meshcat_params.publish_period = 1 / 128.0;
  drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, scene_graph, meshcat, meshcat_params);
  auto meshcat_pc_visualizer =
      builder.AddSystem<drake::geometry::MeshcatPointCloudVisualizer>(
          meshcat, "cloud", meshcat_params.publish_period);
  meshcat_pc_visualizer->set_point_size(0.01 / 4.0);
  builder.Connect(deformable_model->mpm_point_cloud_port(),
                  meshcat_pc_visualizer->cloud_input_port());

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  auto& mutable_context = simulator.get_mutable_context();
  auto& plant_context = plant.GetMyMutableContextFromRoot(&mutable_context);

  plant.SetFreeBodyPose(
      &plant_context, plant.GetBodyByName("free_box"),
      math::RigidTransformd{Vector3d(0.0, 0, box_width / 2.0)});

  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);

  sleep(5);
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
      "A parallel (or suction) gripper grasps a deformable torus on the "
      "ground, lifts it up, and then drops it back on the ground. "
      "Launch meldis before running this example. "
      "Refer to README for instructions on meldis as well as optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}