#include "drake/examples/multibody/deformable/mpm_cloth_shared.h"

DEFINE_bool(write_files, false, "Enable dumping MPM data to files.");
DEFINE_double(simulation_time, 25.0, "Desired duration of the simulation [s].");
DEFINE_int32(res, 50, "Cloth Resolution.");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 2e-3,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(substep, 5e-4,
              "Discrete time step for the substepping scheme [s]. Must be positive.");
DEFINE_string(contact_approximation, "sap",
              "Type of convex contact approximation. See "
              "multibody::DiscreteContactApproximation for details. Options "
              "are: 'sap', 'lagged', and 'similar'.");
DEFINE_double(stiffness, 1.5e3, "Contact Stiffness.");
DEFINE_double(friction, 1.0, "Contact Friction.");
DEFINE_double(damping, 1e-5,
    "Hunt and Crossley damping for the deformable body, only used when "
    "'contact_approximation' is set to 'lagged' or 'similar' [s/m].");

namespace drake {
namespace examples {
namespace {

class FoldingGripperController : public systems::LeafSystem<double> {
 public:
  FoldingGripperController() {
    this->DeclareVectorOutputPort("desired state", BasicVector<double>(24),
                                   &FoldingGripperController::CalcDesiredState);
  }
 
 static constexpr double gripper_xy = 0.04;
 static constexpr double gripper_z = 0.02;
 static constexpr double gripper_density = 10000.0;

 static constexpr double grasp_duration = 4.0;
 static constexpr double up_duration = 1.75;
 static constexpr double forward_duration = 2.0;
 static constexpr double up_v = 0.05;
 static constexpr double forward_v = 0.1;
 static constexpr double l_z = 0.14;
 static constexpr double h_z = 0.16;
 static constexpr double offset_x = 0.1;
 static constexpr double l_x = 0.5 - offset_x;
 static constexpr double h_x = 0.5 + offset_x;

 static constexpr double s2_l_y = 0.57;
 static constexpr double s2_h_y = 0.63;
 static constexpr double s2_x = 0.6;
 static constexpr double s2_grasp_duration = 0.7;
 static constexpr double s2_grasp_v = 0.2;
 static constexpr double s2_l_z = 0.14;
 static constexpr double s2_h_z = 0.16;
 static constexpr double s2_up_duration = 1.75;
 static constexpr double s2_up_v = 0.05;
 static constexpr double s2_forward_duration = 2.0;
 static constexpr double s2_forward_v = 0.1;

 static ModelInstanceIndex AddGripperInstance(MultibodyPlant<double>* plant, ProximityProperties rigid_proximity_props) {
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse", Vector4d(0.5, 0.5, 0.5, 0.8));

  Box gripper_shape(gripper_xy, gripper_xy, gripper_z);
  const auto &gripper_inertia = SpatialInertia<double>::SolidBoxWithDensity(gripper_density, gripper_xy, gripper_xy, gripper_z);

  ModelInstanceIndex gripper_instance = plant->AddModelInstance("gripper_instance");

  const auto &add_single_gripper = [&](std::string name, double x, double y, double z) {
    const RigidBody<double>& x_body = plant->AddRigidBody(name + "_x", gripper_instance, gripper_inertia);
    const auto& x_joint = plant->AddJoint<PrismaticJoint>(name + "_x", plant->world_body(), 
          RigidTransformd::Identity(), x_body, std::nullopt, Vector3d::UnitX());

    const RigidBody<double>& y_body = plant->AddRigidBody(name + "_y", gripper_instance, gripper_inertia);
    const auto& y_joint = plant->AddJoint<PrismaticJoint>(name + "_y", x_body, 
          RigidTransformd::Identity(), y_body, std::nullopt, Vector3d::UnitY());

    const RigidBody<double>& z_body = plant->AddRigidBody(name + "_z", gripper_instance, gripper_inertia);
    const auto& z_joint = plant->AddJoint<PrismaticJoint>(name + "_z", y_body, 
          RigidTransformd::Identity(), z_body, std::nullopt, Vector3d::UnitZ());

    plant->RegisterCollisionGeometry(z_body, RigidTransformd::Identity(), gripper_shape, name + "_collision", rigid_proximity_props);
    plant->RegisterVisualGeometry   (z_body, RigidTransformd::Identity(), gripper_shape, name + "_visual"   , illustration_props);

    const auto x_actuator = plant->AddJointActuator("prismatic" + name + "_x", x_joint).index();
    const auto y_actuator = plant->AddJointActuator("prismatic" + name + "_y", y_joint).index();
    const auto z_actuator = plant->AddJointActuator("prismatic" + name + "_z", z_joint).index();
    plant->GetMutableJointByName<PrismaticJoint>(name + "_x").set_default_translation(x);
    plant->GetMutableJointByName<PrismaticJoint>(name + "_y").set_default_translation(y);
    plant->GetMutableJointByName<PrismaticJoint>(name + "_z").set_default_translation(z);
    plant->get_mutable_joint_actuator(x_actuator).set_controller_gains({1e5, 1});
    plant->get_mutable_joint_actuator(y_actuator).set_controller_gains({1e5, 1});
    plant->get_mutable_joint_actuator(z_actuator).set_controller_gains({1e5, 1});
  };

  add_single_gripper("g1_up",  l_x, l_x, h_z + grasp_duration * up_v);
  add_single_gripper("g1_down", l_x, l_x, l_z - grasp_duration * up_v);
  add_single_gripper("g2_up",  h_x, l_x,  h_z + grasp_duration * up_v);
  add_single_gripper("g2_down", h_x, l_x, l_z - grasp_duration * up_v);

  return gripper_instance;
}

 private:
  void CalcDesiredState(const systems::Context<double>& context,
                        systems::BasicVector<double>* output) const {
    const double t = context.get_time();
    Vector3d g1_up_v;
    Vector3d g1_up_p;
    Vector3d g1_down_v;
    Vector3d g1_down_p;
    Vector3d g2_up_v;
    Vector3d g2_up_p;
    Vector3d g2_down_v;
    Vector3d g2_down_p;
    // stage1: folding
    if (t <= grasp_duration) {
      g1_up_v = Vector3d(0, 0, -up_v);
      g1_up_p = Vector3d(l_x, l_x, h_z + grasp_duration * up_v - t * up_v);
      g1_down_v = Vector3d(0, 0, up_v);
      g1_down_p = Vector3d(l_x, l_x, l_z - grasp_duration * up_v + t * up_v);
      g2_up_v = Vector3d(0, 0, -up_v);
      g2_up_p = Vector3d(h_x, l_x, h_z + grasp_duration * up_v - t * up_v);
      g2_down_v = Vector3d(0, 0, up_v);
      g2_down_p = Vector3d(h_x, l_x, l_z - grasp_duration * up_v + t * up_v);
    } else if (t < grasp_duration + up_duration) {
      double dt = t-grasp_duration;
      g1_up_v = Vector3d(0, 0, up_v);
      g1_up_p = Vector3d(l_x, l_x, h_z + dt * up_v);
      g1_down_v = Vector3d(0, 0, up_v);
      g1_down_p = Vector3d(l_x, l_x, l_z + dt * up_v);
      g2_up_v = Vector3d(0, 0, up_v);
      g2_up_p = Vector3d(h_x, l_x, h_z + dt * up_v);
      g2_down_v = Vector3d(0, 0, up_v);
      g2_down_p = Vector3d(h_x, l_x, l_z + dt * up_v);
    } else if (t < grasp_duration + up_duration + forward_duration) {
      double dt = t-grasp_duration-up_duration;
      g1_up_v = Vector3d(0, forward_v, 0);
      g1_up_p = Vector3d(l_x, l_x + dt * forward_v, h_z + up_duration * up_v);
      g1_down_v = Vector3d(0, forward_v, 0);
      g1_down_p = Vector3d(l_x, l_x + dt * forward_v, l_z + up_duration * up_v);
      g2_up_v = Vector3d(0, forward_v, 0);
      g2_up_p = Vector3d(h_x, l_x + dt * forward_v, h_z + up_duration * up_v);
      g2_down_v = Vector3d(0, forward_v, 0);
      g2_down_p = Vector3d(h_x, l_x + dt * forward_v, l_z + up_duration * up_v);
    } 
    // stage2: folding
    else if (t < grasp_duration + up_duration + forward_duration + s2_grasp_duration) {
      double dt = t - grasp_duration - up_duration - forward_duration;
      g1_up_v = Vector3d(0, 0, -s2_grasp_v);
      g1_up_p = Vector3d(s2_x, s2_l_y, s2_h_z + s2_grasp_duration * s2_grasp_v - dt * s2_grasp_v);
      g1_down_v = Vector3d(0, 0, s2_grasp_v);
      g1_down_p = Vector3d(s2_x, s2_l_y, s2_l_z - s2_grasp_duration * s2_grasp_v + dt * s2_grasp_v);
      g2_up_v = Vector3d(0, 0, -s2_grasp_v);
      g2_up_p = Vector3d(s2_x, s2_h_y, s2_h_z + s2_grasp_duration * s2_grasp_v - dt * s2_grasp_v);
      g2_down_v = Vector3d(0, 0, s2_grasp_v);
      g2_down_p = Vector3d(s2_x, s2_h_y, s2_l_z - s2_grasp_duration * s2_grasp_v + dt * s2_grasp_v);
    } else if (t < grasp_duration + up_duration + forward_duration + s2_grasp_duration + s2_up_duration) {
      double dt = t - grasp_duration - up_duration - forward_duration - s2_grasp_duration;
      g1_up_v = Vector3d(0, 0, s2_up_v);
      g1_up_p = Vector3d(s2_x, s2_l_y, s2_h_z + dt * s2_up_v);
      g1_down_v = Vector3d(0, 0, s2_up_v);
      g1_down_p = Vector3d(s2_x, s2_l_y, s2_l_z + dt * s2_up_v);
      g2_up_v = Vector3d(0, 0, s2_up_v);
      g2_up_p = Vector3d(s2_x , s2_h_y, s2_h_z + dt * s2_up_v);
      g2_down_v = Vector3d(0, 0, s2_up_v);
      g2_down_p = Vector3d(s2_x, s2_h_y, s2_l_z + dt * s2_up_v);
    } else if (t < grasp_duration + up_duration + forward_duration + s2_grasp_duration + s2_up_duration + s2_forward_duration) {
      double dt = t - grasp_duration - up_duration - forward_duration - s2_grasp_duration - s2_up_duration;
      g1_up_v = Vector3d(-s2_forward_v, 0, 0);
      g1_up_p = Vector3d(s2_x - dt * s2_forward_v, s2_l_y, s2_h_z + s2_up_duration * s2_up_v);
      g1_down_v = Vector3d(-s2_forward_v, 0, 0);
      g1_down_p = Vector3d(s2_x - dt * s2_forward_v, s2_l_y, s2_l_z + s2_up_duration * s2_up_v);
      g2_up_v = Vector3d(-s2_forward_v, 0, 0);
      g2_up_p = Vector3d(s2_x - dt * s2_forward_v, s2_h_y, s2_h_z + s2_up_duration * s2_up_v);
      g2_down_v = Vector3d(-s2_forward_v, 0, 0);
      g2_down_p = Vector3d(s2_x - dt * s2_forward_v, s2_h_y, s2_l_z + s2_up_duration * s2_up_v);
    } else {
      g1_up_v = Vector3d(0, 0, 0);
      g1_up_p = Vector3d(s2_x - s2_forward_duration * s2_forward_v, s2_l_y, s2_h_z + s2_up_duration * s2_up_v+1.0);
      g1_down_v = Vector3d(0, 0, 0);
      g1_down_p = Vector3d(s2_x - s2_forward_duration * s2_forward_v, s2_l_y, s2_l_z + s2_up_duration * s2_up_v+1.0);
      g2_up_v = Vector3d(0, 0, 0);
      g2_up_p = Vector3d(s2_x - s2_forward_duration * s2_forward_v, s2_h_y, s2_h_z + s2_up_duration * s2_up_v+1.0);
      g2_down_v = Vector3d(0, 0, 0);
      g2_down_p = Vector3d(s2_x - s2_forward_duration * s2_forward_v, s2_h_y, s2_l_z + s2_up_duration * s2_up_v+1.0);
    }

    output->get_mutable_value() << g1_up_p, g1_down_p, g2_up_p, g2_down_p, g1_up_v, g1_down_v, g2_up_v, g2_down_v;
  }
};

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  plant_config.discrete_contact_approximation = FLAGS_contact_approximation;

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  ProximityProperties rigid_proximity_props;
  ProximityProperties ground_proximity_props;
  const CoulombFriction<double> surface_friction(1.0, 1.0);
  AddCompliantHydroelasticProperties(1.0, 2e5, &rigid_proximity_props);
  AddRigidHydroelasticProperties(1.0, &ground_proximity_props);
  AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
  AddContactMaterial({}, {}, surface_friction, &ground_proximity_props);
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse", Vector4d(0.7, 0.5, 0.4, 0.8));

  /* Set up a ground. */
  Box ground{4, 4, 4};
  const RigidTransformd X_WG(Eigen::Vector3d{0, 0, -2 + 0.1});
  // plant.RegisterCollisionGeometry(plant.world_body(), X_WG, ground, "ground_collision", ground_proximity_props);
  plant.RegisterVisualGeometry(plant.world_body(), X_WG, ground, "ground_visual", std::move(illustration_props));

  DeformableModel<double>& deformable_model = plant.mutable_deformable_model();
  AddCloth(&deformable_model, FLAGS_res, 0.15);
  // AddClothFromFile(&deformable_model, "/home/changyu/Desktop/tshirt.obj");

  MpmConfigParams mpm_config;
  mpm_config.substep_dt = FLAGS_substep;
  mpm_config.write_files = FLAGS_write_files;
  mpm_config.contact_stiffness = FLAGS_stiffness;
  mpm_config.contact_damping = FLAGS_damping;
  mpm_config.contact_friction_mu = FLAGS_friction;
  mpm_config.contact_query_frequency = 8;
  deformable_model.SetMpmConfig(std::move(mpm_config));

  const auto& gripper_instance = FoldingGripperController::AddGripperInstance(&plant, rigid_proximity_props);

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
  
  builder.Connect(builder.AddSystem<FoldingGripperController>()->get_output_port(), plant.get_desired_state_input_port(gripper_instance));

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
  gflags::SetUsageMessage("This is a demo used to showcase cloth folding simulations in Drake.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}
