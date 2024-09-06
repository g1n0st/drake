#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/examples/multibody/deformable/parallel_gripper_controller.h"
#include "drake/examples/multibody/deformable/point_source_force_field.h"
#include "drake/examples/multibody/deformable/suction_cup_controller.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_bool(write_files, false, "Enable dumping MPM data to files.");
DEFINE_double(simulation_time, 10.0, "Desired duration of the simulation [s].");
DEFINE_int32(res, 50, "Cloth Resolution.");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 1e-3,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(substep, 1e-3,
              "Discrete time step for the substepping scheme [s]. Must be positive.");
DEFINE_string(contact_approximation, "sap",
              "Type of convex contact approximation. See "
              "multibody::DiscreteContactApproximation for details. Options "
              "are: 'sap', 'lagged', and 'similar'.");
DEFINE_double(stiffness, 1000000.0, "Contact Stiffness.");
DEFINE_double(friction, 1.0, "Contact Friction.");
DEFINE_double(damping, 1e-5,
    "Hunt and Crossley damping for the deformable body, only used when "
    "'contact_approximation' is set to 'lagged' or 'similar' [s/m].");

using drake::examples::deformable::ParallelGripperController;
using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::Sphere;
using drake::geometry::Capsule;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::math::RigidTransformd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::gmpm::MpmConfigParams;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
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

class MyGripperController : public systems::LeafSystem<double> {
 public:
  MyGripperController() {
    this->DeclareVectorOutputPort("desired state", BasicVector<double>(6),
                                   &MyGripperController::CalcDesiredState);
  }

 private:
  void CalcDesiredState(const systems::Context<double>& context,
                        systems::BasicVector<double>* output) const {
    unused(context);
    const double t = context.get_time();
    Vector3d desired_velocities = Vector3d::Zero();
    Vector3d desired_positions = Vector3d(0.5, 0.5, 0.1 + t * 0.05);
    output->get_mutable_value() << desired_positions, desired_velocities;
  }
};

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  plant_config.discrete_contact_approximation = FLAGS_contact_approximation;

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  /* Minimum required proximity properties for rigid bodies to interact with
   deformable bodies.
   1. A valid Coulomb friction coefficient, and
   2. A resolution hint. (Rigid bodies need to be tessellated so that collision
   queries can be performed against deformable geometries.) The value dictates
   how fine the mesh used to represent the rigid collision geometry is. */
  ProximityProperties rigid_proximity_props;
  ProximityProperties ground_proximity_props;
  /* Set the friction coefficient close to that of rubber against rubber. */
  const CoulombFriction<double> surface_friction(1.0, 1.0);
  AddCompliantHydroelasticProperties(1.0, 2e5, &rigid_proximity_props);
  AddRigidHydroelasticProperties(1.0, &ground_proximity_props);
  AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
  AddContactMaterial({}, {}, surface_friction, &ground_proximity_props);
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse", Vector4d(0.7, 0.5, 0.4, 0.8));

  DeformableModel<double>& deformable_model = plant.mutable_deformable_model();

  const int res = FLAGS_res;
  const double l = 0.005 * res;
  int length = res;
  int width = res;
  double dx = l / width;

  auto p = [&](int i, int j) {
    return i * width + j;
  };

  {
    std::vector<Eigen::Vector3d> inital_pos;
    std::vector<Eigen::Vector3d> inital_vel;
    std::vector<int> indices;
    for (int i = 0; i < length; ++i) {
      for (int j = 0; j < width; ++j) {
        inital_pos.emplace_back((0.5 - 0.5 * l) + i * dx, (0.5 - 0.5 * l) + j * dx, 0.2);
        inital_vel.emplace_back(0., 0., 0.);
      }
    }

    for (int i = 0; i < length; ++i) {
      for (int j = 0; j < width; ++j) {
        if (i < length - 1 && j < width - 1) {
          indices.push_back(p(i, j));
          indices.push_back(p(i+1, j));
          indices.push_back(p(i, j+1));

          indices.push_back(p(i+1, j+1));
          indices.push_back(p(i, j+1));
          indices.push_back(p(i+1, j));
        }
      }
    }

    deformable_model.RegisterMpmCloth(inital_pos, inital_vel, indices);
  }

  MpmConfigParams mpm_config;
  mpm_config.substep_dt = FLAGS_substep;
  mpm_config.write_files = FLAGS_write_files;
  mpm_config.contact_stiffness = FLAGS_stiffness;
  mpm_config.contact_damping = FLAGS_damping;
  mpm_config.contact_friction_mu = FLAGS_friction;
  deformable_model.SetMpmConfig(std::move(mpm_config));

  const double gripper_xy = 0.05;
  const double gripper_z = 0.025;
  const double gripper_density = 4000.0;
  Box gripper_shape(gripper_xy, gripper_xy, gripper_z);
  const auto &gripper_inertia = SpatialInertia<double>::SolidBoxWithDensity(gripper_density, gripper_xy, gripper_xy, gripper_z);

  ModelInstanceIndex g1_instance = plant.AddModelInstance("g1_instance");
  const RigidBody<double>& x_body = plant.AddRigidBody("g1_x", g1_instance, gripper_inertia);
  const auto& x_joint = plant.AddJoint<PrismaticJoint>("g1_x", plant.world_body(), 
        RigidTransformd::Identity(), x_body, std::nullopt, Vector3d::UnitX());
  const RigidBody<double>& y_body = plant.AddRigidBody("g1_y", g1_instance, gripper_inertia);
  const auto& y_joint = plant.AddJoint<PrismaticJoint>("g1_y", x_body, 
        RigidTransformd::Identity(), y_body, std::nullopt, Vector3d::UnitY());
  const RigidBody<double>& z_body = plant.AddRigidBody("g1_z", g1_instance, gripper_inertia);
  const auto& z_joint = plant.AddJoint<PrismaticJoint>("g1_z", y_body, 
        RigidTransformd::Identity(), z_body, std::nullopt, Vector3d::UnitZ());
  plant.RegisterCollisionGeometry(z_body, RigidTransformd::Identity(), gripper_shape,
                                    "g1_collision", rigid_proximity_props);
  plant.RegisterVisualGeometry(z_body, RigidTransformd::Identity(), gripper_shape,
                                    "g1_visual", illustration_props);
  const auto g1_x_actuator = plant.AddJointActuator("prismatic g1_x", x_joint).index();
  const auto g1_y_actuator = plant.AddJointActuator("prismatic g1_y", y_joint).index();
  const auto g1_z_actuator = plant.AddJointActuator("prismatic g1_z", z_joint).index();
  plant.GetMutableJointByName<PrismaticJoint>("g1_x").set_default_translation(0.5);
  plant.GetMutableJointByName<PrismaticJoint>("g1_y").set_default_translation(0.5);
  plant.GetMutableJointByName<PrismaticJoint>("g1_z").set_default_translation(0.1);
  plant.get_mutable_joint_actuator(g1_x_actuator).set_controller_gains({1e10, 1});
  plant.get_mutable_joint_actuator(g1_y_actuator).set_controller_gains({1e10, 1});
  plant.get_mutable_joint_actuator(g1_z_actuator).set_controller_gains({1e10, 1});



  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams visualize_params;
  visualize_params.show_mpm = true;
  auto& visualizer = geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph, nullptr, visualize_params);

  // NOTE (changyu): MPM shortcut port shuould be explicit connected for visualization.
  builder.Connect(plant.get_output_port(
    plant.deformable_model().mpm_output_port_index()), 
    visualizer.mpm_input_port());
  
  const auto& control = *builder.AddSystem<MyGripperController>();
  builder.Connect(control.get_output_port(),
                  plant.get_desired_state_input_port(g1_instance));

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context = diagram->CreateDefaultContext();

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);
  simulator.AdvanceTo(FLAGS_simulation_time);

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
