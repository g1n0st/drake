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
  AddCloth(&deformable_model, FLAGS_res, 0.75);
  // AddClothFromFile(&deformable_model, "/home/changyu/Desktop/tshirt.obj");

  MpmConfigParams mpm_config;
  mpm_config.substep_dt = FLAGS_substep;
  mpm_config.write_files = FLAGS_write_files;
  mpm_config.contact_stiffness = FLAGS_stiffness;
  mpm_config.contact_damping = FLAGS_damping;
  mpm_config.contact_friction_mu = FLAGS_friction;
  mpm_config.contact_query_frequency = 8;
  mpm_config.mpm_bc = 1;
  deformable_model.SetMpmConfig(std::move(mpm_config));

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
  gflags::SetUsageMessage("This is a demo used to showcase bagging simulations in Drake.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}
