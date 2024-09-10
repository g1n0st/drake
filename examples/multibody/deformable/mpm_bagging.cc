#include "drake/examples/multibody/deformable/mpm_cloth_shared.h"

DEFINE_bool(write_files, false, "Enable dumping MPM data to files.");
DEFINE_double(simulation_time, 1.0, "Desired duration of the simulation [s].");
DEFINE_int32(res, 60, "Cloth Resolution.");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 1e-3,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(substep, 2e-4,
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

class BaggingGripperController : public systems::LeafSystem<double> {
 public:
  BaggingGripperController() {
    this->DeclareVectorOutputPort("desired state", BasicVector<double>(48),
                                   &BaggingGripperController::CalcDesiredState);
  }
 
 static constexpr double gripper_xy = 0.05;
 static constexpr double gripper_z = 0.02;
 static constexpr double gripper_density = 10000.0;

 static constexpr double l_x = 0.34;
 static constexpr double h_x = 0.66;
 static constexpr double l_z = 0.49+2e-3;
 static constexpr double h_z = 0.51-2e-3;

 static constexpr double free_duration = 2.0;
 static constexpr double bagging_duration = 1.25;
 static constexpr double bagging_v = 0.1;

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
    plant->get_mutable_joint_actuator(x_actuator).set_controller_gains({1e6, 1});
    plant->get_mutable_joint_actuator(y_actuator).set_controller_gains({1e6, 1});
    plant->get_mutable_joint_actuator(z_actuator).set_controller_gains({1e6, 1});
  };

  add_single_gripper("gll_up",  l_x, l_x, h_z);
  add_single_gripper("glh_up", l_x, h_x, h_z);
  add_single_gripper("ghl_up",  h_x, l_x,  h_z);
  add_single_gripper("ghh_up", h_x, h_x, h_z);
  add_single_gripper("gll_down",  l_x, l_x, l_z);
  add_single_gripper("glh_down", l_x, h_x, l_z);
  add_single_gripper("ghl_down",  h_x, l_x,  l_z);
  add_single_gripper("ghh_down", h_x, h_x, l_z);

  return gripper_instance;
}

 private:
  void CalcDesiredState(const systems::Context<double>& context,
                        systems::BasicVector<double>* output) const {
    const double t = context.get_time();

    Vector3d gll_up_p;
    Vector3d glh_up_p;
    Vector3d ghl_up_p;
    Vector3d ghh_up_p;
    Vector3d gll_down_p;
    Vector3d glh_down_p;
    Vector3d ghl_down_p;
    Vector3d ghh_down_p;

    Vector3d gll_up_v;
    Vector3d glh_up_v;
    Vector3d ghl_up_v;
    Vector3d ghh_up_v;
    Vector3d gll_down_v;
    Vector3d glh_down_v;
    Vector3d ghl_down_v;
    Vector3d ghh_down_v;
    if (t < free_duration) {
      gll_up_p = Vector3d(l_x, l_x, h_z);
      glh_up_p = Vector3d(l_x, h_x, h_z);
      ghl_up_p = Vector3d(h_x, l_x,  h_z);
      ghh_up_p = Vector3d(h_x, h_x, h_z);
      gll_down_p = Vector3d(l_x, l_x, l_z);
      glh_down_p = Vector3d(l_x, h_x, l_z);
      ghl_down_p = Vector3d(h_x, l_x,  l_z);
      ghh_down_p = Vector3d(h_x, h_x, l_z);

      gll_up_v = Vector3d(0, 0, 0);
      glh_up_v = Vector3d(0, 0, 0);
      ghl_up_v = Vector3d(0, 0, 0);
      ghh_up_v = Vector3d(0, 0, 0);
      gll_down_v = Vector3d(0, 0, 0);
      glh_down_v = Vector3d(0, 0, 0);
      ghl_down_v = Vector3d(0, 0, 0);
      ghh_down_v = Vector3d(0, 0, 0);
    } else if (t < free_duration + bagging_duration) {
      double dt = t - free_duration;
      gll_up_p = Vector3d(l_x + dt * bagging_v, l_x + dt * bagging_v, h_z);
      glh_up_p = Vector3d(l_x + dt * bagging_v, h_x - dt * bagging_v, h_z);
      ghl_up_p = Vector3d(h_x - dt * bagging_v, l_x + dt * bagging_v,  h_z);
      ghh_up_p = Vector3d(h_x - dt * bagging_v, h_x - dt * bagging_v, h_z);
      gll_down_p = Vector3d(l_x + dt * bagging_v, l_x + dt * bagging_v, l_z);
      glh_down_p = Vector3d(l_x + dt * bagging_v, h_x - dt * bagging_v, l_z);
      ghl_down_p = Vector3d(h_x - dt * bagging_v, l_x + dt * bagging_v,  l_z);
      ghh_down_p = Vector3d(h_x - dt * bagging_v, h_x - dt * bagging_v, l_z);

      gll_up_v = Vector3d(+ dt * bagging_v, + dt * bagging_v, 0);
      glh_up_v = Vector3d(+ dt * bagging_v, - dt * bagging_v, 0);
      ghl_up_v = Vector3d(- dt * bagging_v, + dt * bagging_v, 0);
      ghh_up_v = Vector3d(- dt * bagging_v, - dt * bagging_v, 0);
      gll_down_v = Vector3d(+ dt * bagging_v, + dt * bagging_v, 0);
      glh_down_v = Vector3d(+ dt * bagging_v, - dt * bagging_v, 0);
      ghl_down_v = Vector3d(- dt * bagging_v, + dt * bagging_v, 0);
      ghh_down_v = Vector3d(- dt * bagging_v, - dt * bagging_v, 0);
    } else {
      gll_up_p = Vector3d(l_x + bagging_duration * bagging_v, l_x + bagging_duration * bagging_v, h_z);
      glh_up_p = Vector3d(l_x + bagging_duration * bagging_v, h_x - bagging_duration * bagging_v, h_z);
      ghl_up_p = Vector3d(h_x - bagging_duration * bagging_v, l_x + bagging_duration * bagging_v,  h_z);
      ghh_up_p = Vector3d(h_x - bagging_duration * bagging_v, h_x - bagging_duration * bagging_v, h_z);
      gll_down_p = Vector3d(l_x + bagging_duration * bagging_v, l_x + bagging_duration * bagging_v, l_z);
      glh_down_p = Vector3d(l_x + bagging_duration * bagging_v, h_x - bagging_duration * bagging_v, l_z);
      ghl_down_p = Vector3d(h_x - bagging_duration * bagging_v, l_x + bagging_duration * bagging_v,  l_z);
      ghh_down_p = Vector3d(h_x - bagging_duration * bagging_v, h_x - bagging_duration * bagging_v, l_z);

      gll_up_v = Vector3d(0, 0, 0);
      glh_up_v = Vector3d(0, 0, 0);
      ghl_up_v = Vector3d(0, 0, 0);
      ghh_up_v = Vector3d(0, 0, 0);
      gll_down_v = Vector3d(0, 0, 0);
      glh_down_v = Vector3d(0, 0, 0);
      ghl_down_v = Vector3d(0, 0, 0);
      ghh_down_v = Vector3d(0, 0, 0);
    }

    output->get_mutable_value() << 
      gll_up_p, glh_up_p, ghl_up_p, ghh_up_p, gll_down_p, glh_down_p, ghl_down_p, ghh_down_p,
      gll_up_v, glh_up_v, ghl_up_v, ghh_up_v, gll_down_v, glh_down_v, ghl_down_v, ghh_down_v;
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

  const auto &AddRigidBox = [&](std::string name) {
    const double side_length = 0.05;
    Box box(side_length, side_length, side_length);
    const RigidBody<double>& rigid_box = plant.AddRigidBody(
        name, SpatialInertia<double>::SolidBoxWithDensity(300.0, side_length, side_length, side_length));
    plant.RegisterCollisionGeometry(rigid_box, RigidTransformd::Identity(), box,
                                    name + "_collision", rigid_proximity_props);
    plant.RegisterVisualGeometry(rigid_box, RigidTransformd::Identity(), box,
                                name + "_visual", illustration_props);
  };

  AddRigidBox("box1");
  AddRigidBox("box2");
  AddRigidBox("box3");
  AddRigidBox("box4");

  DeformableModel<double>& deformable_model = plant.mutable_deformable_model();
  AddCloth(&deformable_model, FLAGS_res, 0.5);
  // AddClothFromFile(&deformable_model, "/home/changyu/Desktop/tshirt.obj");

  MpmConfigParams mpm_config;
  mpm_config.substep_dt = FLAGS_substep;
  mpm_config.write_files = FLAGS_write_files;
  mpm_config.contact_stiffness = FLAGS_stiffness;
  mpm_config.contact_damping = FLAGS_damping;
  mpm_config.contact_friction_mu = FLAGS_friction;
  mpm_config.contact_query_frequency = 8;
  mpm_config.mpm_bc = -1;
  deformable_model.SetMpmConfig(std::move(mpm_config));

  const auto& gripper_instance = BaggingGripperController::AddGripperInstance(&plant, rigid_proximity_props);

  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  /* Add a visualizer that emits LCM messages for visualization. */
  /*geometry::DrakeVisualizerParams visualize_params;
  visualize_params.show_mpm = true;
  auto& visualizer = geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph, nullptr, visualize_params);

  // NOTE (changyu): MPM shortcut port shuould be explicit connected for visualization.
  builder.Connect(plant.get_output_port(
    plant.deformable_model().mpm_output_port_index()), 
    visualizer.mpm_input_port());*/

  builder.Connect(builder.AddSystem<BaggingGripperController>()->get_output_port(), plant.get_desired_state_input_port(gripper_instance));

  auto meshcat = std::make_shared<geometry::Meshcat>();
  auto meshcat_params = drake::geometry::MeshcatVisualizerParams();
  meshcat_params.show_mpm = true;
  auto& meshcat_visualizer = drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, scene_graph, meshcat, meshcat_params);
  visualization::ApplyVisualizationConfig(
      visualization::VisualizationConfig{
          .default_proximity_color = geometry::Rgba{1, 0, 0, 0.25},
          .enable_alpha_sliders = true,
      },
      &builder, nullptr, nullptr, nullptr, meshcat);
  
  builder.Connect(plant.get_output_port(
    plant.deformable_model().mpm_output_port_index()), 
    meshcat_visualizer.mpm_input_port());

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context = diagram->CreateDefaultContext();
  auto& plant_context = plant.GetMyMutableContextFromRoot(diagram_context.get());
  plant.SetFreeBodyPose(&plant_context, plant.GetBodyByName("box1"), RigidTransformd(Eigen::Vector3d{0.5, 0.5, 0.59}));
  plant.SetFreeBodyPose(&plant_context, plant.GetBodyByName("box2"), RigidTransformd(Eigen::Vector3d{0.4, 0.4, 0.54}));
  plant.SetFreeBodyPose(&plant_context, plant.GetBodyByName("box3"), RigidTransformd(Eigen::Vector3d{0.35, 0.55, 0.57}));
  plant.SetFreeBodyPose(&plant_context, plant.GetBodyByName("box4"), RigidTransformd(Eigen::Vector3d{0.55, 0.45, 0.53}));

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  meshcat->StartRecording();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);
  meshcat->StopRecording();
  meshcat->PublishRecording();

  std::ofstream htmlFile("mpm_bagging.html");
  htmlFile << meshcat->StaticHtml();
  htmlFile.close();

  return 0;
}

}  // namespace
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("This is a demo used to showcase bagging simulations in Drake.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}
