#include "../sparse_grid.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

using drake::geometry::AddContactMaterial;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using Eigen::Vector3d;
using Eigen::Vector3f;
using Eigen::Vector3i;
using geometry::FrameId;
using geometry::GeometryInstance;
using geometry::ProximityProperties;
using geometry::QueryObject;
using geometry::SceneGraph;
using geometry::SourceId;
using geometry::Sphere;
using math::RigidTransformd;
using systems::Context;

GTEST_TEST(SparseGridTest, Allocate) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  const Vector3d q_WP = Vector3d(1.001, 0.001, 0.001);
  std::vector<Vector3d> q_WPs = {q_WP};

  grid.Allocate(q_WPs);

  EXPECT_EQ(grid.dx(), 0.01);

  const std::vector<int> expected_sentinel_particles = {0, 1};
  EXPECT_EQ(grid.sentinel_particles(), expected_sentinel_particles);

  const std::vector<int>& data_indices = grid.data_indices();
  ASSERT_EQ(data_indices.size(), 1);
  EXPECT_EQ(data_indices[0], 0);

  const std::vector<uint64_t>& base_node_offsets = grid.base_node_offsets();
  ASSERT_EQ(base_node_offsets.size(), 1);
  EXPECT_EQ(base_node_offsets[0], grid.CoordinateToOffset(100, 0, 0));

  /* Verify grid data is all zeroed out. */
  const std::vector<std::pair<Vector3i, GridData<double>>> grid_data =
      grid.GetGridData();
  for (const auto& [node, data] : grid_data) {
    EXPECT_EQ(data.m, 0.0);
    EXPECT_EQ(data.v, Vector3d::Zero());
  }

  EXPECT_EQ(grid.num_blocks(), 1);
}

GTEST_TEST(SparseGridTest, AllocateForCollision) {
  const float dx = 0.01;
  SparseGrid<float> grid(dx);
  const Vector3f q_WP = Vector3f(1.001, 0.001, 0.001);
  std::vector<Vector3f> q_WPs = {q_WP};
  grid.AllocateForCollision(q_WPs);
}

GTEST_TEST(SparseGridTest, BaseNodeOffsets) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  /* Particle 0, 1 has the same base node (0, 0, 0).
     Particle 2 has a different base node (1, 0, 0), but is still in the same
     block. Particle 3 with base node (4, 0, 0) is in a different block from
     particles 0, 1, and 2. */
  const Vector3d q_WP0 = Vector3d(0.001, 0.001, 0.001);
  const Vector3d q_WP1 = Vector3d(-0.004, 0.001, 0.001);
  const Vector3d q_WP2 = Vector3d(0.012, 0.004, 0.001);
  const Vector3d q_WP3 = Vector3d(0.04, 0.0, 0.0);
  std::vector<Vector3d> q_WPs = {q_WP0, q_WP1, q_WP2, q_WP3};

  grid.Allocate(q_WPs);

  const std::vector<uint64_t>& base_node_offsets = grid.base_node_offsets();
  ASSERT_EQ(base_node_offsets.size(), 4);
  EXPECT_EQ(base_node_offsets[0], grid.CoordinateToOffset(0, 0, 0));
  EXPECT_EQ(base_node_offsets[1], grid.CoordinateToOffset(0, 0, 0));
  EXPECT_EQ(base_node_offsets[2], grid.CoordinateToOffset(1, 0, 0));
  EXPECT_EQ(base_node_offsets[3], grid.CoordinateToOffset(4, 0, 0));

  EXPECT_EQ(grid.num_blocks(), 2);
}

GTEST_TEST(SparseGridTest, SentinelParticles) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  /* Particle 0, 1 has the same base node (0, 0, 0).
     Particle 2 has a different base node (1, 0, 0), but is still in the same
     block. Particle 3 is in a different block from particles 0, 1, and 2. */
  const Vector3d q_WP0 = Vector3d(0.001, 0.001, 0.001);
  const Vector3d q_WP1 = Vector3d(-0.004, 0.001, 0.001);
  const Vector3d q_WP2 = Vector3d(0.012, 0.004, 0.001);
  const Vector3d q_WP3 = Vector3d(0.04, 0.0, 0.0);
  std::vector<Vector3d> q_WPs = {q_WP0, q_WP1, q_WP2, q_WP3};

  grid.Allocate(q_WPs);

  EXPECT_EQ(grid.dx(), 0.01);

  /* Sentinel particles are particles 0 and 3, marking boundary of new blocks.
   The last entry is the number of particles. */
  const std::vector<int> expected_sentinel_particles = {0, 3, 4};
  EXPECT_EQ(grid.sentinel_particles(), expected_sentinel_particles);

  /* Particles are sorted first based on their base node offsets:

    base_node(0) == base_node(1) < base_node(2) < base_node(3),

    and then on original indices: 0 < 1.*/
  const std::vector<int> expected_data_indices = {0, 1, 2, 3};
  EXPECT_EQ(grid.data_indices(), expected_data_indices);
}

GTEST_TEST(SparseGridTest, DataIndices) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  /* Here we swap the positions of particle 0 and 3 from the previous test. */
  const Vector3d q_WP0 = Vector3d(0.04, 0.0, 0.0);
  const Vector3d q_WP1 = Vector3d(-0.004, 0.001, 0.001);
  const Vector3d q_WP2 = Vector3d(0.012, 0.004, 0.001);
  const Vector3d q_WP3 = Vector3d(0.001, 0.001, 0.001);
  std::vector<Vector3d> q_WPs = {q_WP0, q_WP1, q_WP2, q_WP3};

  grid.Allocate(q_WPs);

  EXPECT_EQ(grid.dx(), 0.01);

  /* The sentinel particles remain the same. */
  const std::vector<int> expected_sentinel_particles = {0, 3, 4};
  EXPECT_EQ(grid.sentinel_particles(), expected_sentinel_particles);

  /* But the data indices are different.
    base_node(1) == base_node(3) < base_node(2) < base_node(0). */
  const std::vector<int> expected_data_indices = {1, 3, 2, 0};
  EXPECT_EQ(grid.data_indices(), expected_data_indices);
}

GTEST_TEST(SparseGridTest, GetPadNodes) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  const Vector3d q_WP = Vector3d(0.001, 0.001, 0.001);
  /* Base node is (0,0,0), so we should get the 27 immediate neighbors of the
   (0, 0, 0) as the pad nodes. */
  const Pad<Vector3d> pad_nodes = grid.GetPadNodes(q_WP);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        const Vector3d node =
            Vector3d((i - 1) * dx, (j - 1) * dx, (k - 1) * dx);
        EXPECT_EQ(pad_nodes[i][j][k], node);
      }
    }
  }
}

GTEST_TEST(SparseGridTest, PadData) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  /* Base node is (2, 3, 0). */
  const Vector3d q_WP = Vector3d(0.021, 0.031, -0.001);
  std::vector<Vector3d> q_WPs = {q_WP};

  grid.Allocate(q_WPs);

  const std::vector<uint64_t>& base_node_offsets = grid.base_node_offsets();
  ASSERT_EQ(base_node_offsets.size(), 1);
  const uint64_t base_node_offset = base_node_offsets[0];

  Pad<GridData<double>> arbitrary_data;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        GridData<double> foo;
        foo.m = i + j + k;
        foo.v = Vector3d(i, j, k);
        arbitrary_data[i][j][k] = foo;
      }
    }
  }

  grid.SetPadData(base_node_offset, arbitrary_data);
  const Pad<GridData<double>> pad_data = grid.GetPadData(base_node_offset);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        EXPECT_EQ(pad_data, arbitrary_data);
      }
    }
  }

  /* Now get the pad centered at (1, 2, -1). It should overlap with the pad
   centered at (2, 3, 0). The non-overlapping portion should be zeroed out
   (during Allocate()).  */
  const Pad<GridData<double>> pad_data2 =
      grid.GetPadData(grid.CoordinateToOffset(1, 2, -1));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        if (i == 0 || j == 0 || k == 0) {
          EXPECT_EQ(pad_data2[i][j][k].m, 0.0);
          EXPECT_EQ(pad_data2[i][j][k].v, Vector3d::Zero());
        } else {
          EXPECT_EQ(pad_data2[i][j][k], arbitrary_data[i - 1][j - 1][k - 1]);
        }
      }
    }
  }
}

GTEST_TEST(SparseGridTest, CoordinateToOffset) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  /* Base node is (2, 3, 0). */
  const Vector3d q_WP = Vector3d(0.021, 0.031, -0.001);
  std::vector<Vector3d> q_WPs = {q_WP};

  grid.Allocate(q_WPs);

  const uint64_t base_node_offset = grid.base_node_offsets()[0];
  EXPECT_EQ(grid.CoordinateToOffset(2, 3, 0), base_node_offset);
  EXPECT_EQ(grid.OffsetToCoordinate(base_node_offset), Vector3i(2, 3, 0));
}

GTEST_TEST(SparseGridTest, ExplicitVelocityUpdate) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  const Vector3d q_WP = Vector3d(0.001, 0.001, 0.001);
  std::vector<Vector3d> q_WPs = {q_WP};

  grid.Allocate(q_WPs);

  auto set_arbitrary_grid_data = [](const Vector3i& node) {
    GridData<double> result;
    result.m = node[0] + node[1] + node[2];
    result.v = Vector3d(node[0], node[1], node[2]);
    return result;
  };

  grid.SetGridData(set_arbitrary_grid_data);

  /* Verify the grid data is set as expected. */
  const std::vector<std::pair<Vector3i, GridData<double>>> grid_data =
      grid.GetGridData();
  for (const auto& [node, data] : grid_data) {
    EXPECT_EQ(data.m, node[0] + node[1] + node[2]);
    EXPECT_EQ(data.v, Vector3d(node[0], node[1], node[2]));
  }

  /* Convert momentum to velocity and explicitly update the velocity field. */
  const Vector3d dv(1, 2, 3);
  std::vector<multibody::ExternallyAppliedSpatialForce<double>> unused;
  grid.ExplicitVelocityUpdate(dv, &unused);

  const std::vector<std::pair<Vector3i, GridData<double>>> grid_data2 =
      grid.GetGridData();
  for (const auto& [node, data] : grid_data2) {
    EXPECT_EQ(data.m, node[0] + node[1] + node[2]);
    EXPECT_EQ(data.v, Vector3d(node[0], node[1], node[2]) / data.m + dv);
  }
}

GTEST_TEST(SparseGridTest, ComputeTotalMassAndMomentum) {
  const double dx = 0.01;
  SparseGrid<double> grid(dx);
  const Vector3d q_WP = Vector3d(0.001, 0.001, 0.001);
  std::vector<Vector3d> q_WPs = {q_WP};

  grid.Allocate(q_WPs);

  const double mass = 1.2;
  const Vector3d velocity = Vector3d(1, 2, 3);
  /* World frame position of the node with non-zero mass. */
  const Vector3d q_WN = Vector3d(dx, dx, dx);
  /* Set grid data so that the grid node (1, 1, 1) has velocity (1, 2, 3) and
   all other grid nodes have zero velocity. */
  auto set_grid_data = [mass, velocity](const Vector3i& node) {
    GridData<double> result;
    if (node[0] == 1 && node[1] == 1 && node[2] == 1) {
      result.m = mass;
      result.v = velocity;
    } else {
      result.set_zero();
    }
    return result;
  };

  grid.SetGridData(set_grid_data);

  const MassAndMomentum<double> computed = grid.ComputeTotalMassAndMomentum();
  EXPECT_EQ(computed.mass, mass);
  EXPECT_TRUE(CompareMatrices(computed.linear_momentum, mass * velocity,
                              4.0 * std::numeric_limits<double>::epsilon()));
  EXPECT_TRUE(CompareMatrices(computed.angular_momentum,
                              mass * q_WN.cross(velocity),
                              4.0 * std::numeric_limits<double>::epsilon()));
}

/* We place 8 particles in the grid to activate one block for each color.
 For floats, each block is of size 4x4x4 grid nodes. We place particles at
 (5*i*dx, 5*j*dx, 5*k*dx) for i, j, k = 0 or 1 to activate the 8 blocks
 starting at the block containing the origin. */
GTEST_TEST(SparseGridTest, ColoredBlocks) {
  const float dx = 0.01;
  SparseGrid<float> grid(dx);
  std::vector<Vector3f> q_WPs;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        q_WPs.emplace_back(Vector3f(5 * i * dx, 5 * j * dx, 5 * k * dx));
      }
    }
  }

  grid.Allocate(q_WPs);

  const auto& sentinel_particles = grid.sentinel_particles();
  ASSERT_EQ(sentinel_particles.size(), 9);

  const std::array<std::vector<int>, 8> expected_blocks = {
      {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}}};

  EXPECT_EQ(grid.colored_blocks(), expected_blocks);
}

/* Tests SparseGrid::RasterizeRigidData and the logic in
 SparseGrid::ExplicitVelocityUpdate that handles boundary conditions on the
 grid nodes and the contact impulses applied to rigid bodies. */
GTEST_TEST(SparseGrid, BoundaryConditions) {
  systems::DiagramBuilder<double> builder;
  MultibodyPlantConfig plant_config;
  plant_config.time_step = 0.01;
  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  /* Add a free sphere a sphere with radius 0.6. */
  ProximityProperties rigid_proximity_props;
  /* Set the friction coefficient close to that of rubber against rubber. */
  const CoulombFriction<double> surface_friction(1.0, 1.0);
  AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
  const double radius = 0.6;
  const RigidBody<double>& body = plant.AddRigidBody(
      "sphere_body",
      SpatialInertia<double>::SolidSphereWithDensity(1000.0, radius));
  plant.RegisterCollisionGeometry(body, RigidTransformd::Identity(),
                                  Sphere(radius), "sphere_collision",
                                  rigid_proximity_props);
  plant.Finalize();
  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(diagram_context.get());
  /* Place the sphere at (1, 1, 1)*/
  plant.SetFreeBodyPose(&plant_context, body,
                        RigidTransformd(Vector3d(1, 1, 1)));
  /* Assign it some arbitrary spatial velocity (preferring numbers that produces
   simple analytic results). */
  const Vector3d v_WB = Vector3d(-1, 0, 0);
  const Vector3d w_WB = Vector3d(0, 0, 2);
  plant.SetFreeBodySpatialVelocity(&plant_context, body,
                                   SpatialVelocity<double>(w_WB, v_WB));
  Context<double>& scene_graph_context =
      scene_graph.GetMyMutableContextFromRoot(diagram_context.get());
  const QueryObject<double>& query_object =
      scene_graph.get_query_output_port().template Eval<QueryObject<double>>(
          scene_graph_context);
  const std::unordered_map<geometry::GeometryId, multibody::BodyIndex>&
      geometry_id_to_body_index = plant.geometry_id_to_body_index();
  const auto& V_WB_all =
      plant.get_body_spatial_velocities_output_port()
          .Eval<std::vector<SpatialVelocity<double>>>(plant_context);
  const auto& X_WB_all =
      plant.get_body_poses_output_port()
          .Eval<std::vector<math::RigidTransform<double>>>(plant_context);

  /* Set up a grid with grid nodes in [0, 2] x [0, 2] x [0, 2] all active. */
  const double dx = 0.5;
  SparseGrid<double> grid(dx);
  std::vector<Vector3d> q_WPs;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 5; ++k) {
        q_WPs.emplace_back(Vector3d(i * dx, j * dx, k * dx));
      }
    }
  }

  grid.AllocateForCollision(q_WPs);

  /* Set a constant velocity field (1, 0, 1). */
  auto set_constant_velocity_field = [](const Vector3i&) {
    GridData<double> result;
    result.m = 1.0;
    result.v = Vector3d(1.0, 0.0, 1.0);
    return result;
  };
  grid.SetGridData(set_constant_velocity_field);
  MassAndMomentum<double> initial_momentum = grid.ComputeTotalMassAndMomentum();

  /* Splat rigid data to grid. */
  std::vector<multibody::ExternallyAppliedSpatialForce<double>> rigid_forces;
  grid.RasterizeRigidData(query_object, V_WB_all, X_WB_all,
                          geometry_id_to_body_index, &rigid_forces);
  /* Two rigid bodies: the first is the world body and the second is the rigid
   sphere. */
  ASSERT_EQ(rigid_forces.size(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(rigid_forces[i].body_index, BodyIndex(i));
    /* We borrowed the field `p_BoBq_B to store p_WBo. */
    EXPECT_EQ(rigid_forces[i].p_BoBq_B, X_WB_all[i].translation());
    EXPECT_EQ(rigid_forces[i].F_Bq_W.rotational(), Vector3d::Zero());
    EXPECT_EQ(rigid_forces[i].F_Bq_W.translational(), Vector3d::Zero());
  }
  grid.ExplicitVelocityUpdate(Vector3d::Zero(), &rigid_forces);

  /* Verify the grid data is set as expected. */
  const std::vector<std::pair<Vector3i, GridData<double>>> grid_data =
      grid.GetGridData();

  for (const auto& [node, data] : grid_data) {
    /* The grid mass is unaffected by the rigid data. */
    EXPECT_EQ(data.m, 1.0);
    const Vector3d p_WN = node.cast<double>() * dx;
    /* There are seven nodes that fall inside the sphere: node (1, 1, 1) and its
     six immediate neighbors. The normal is ill-defined at (1, 1, 1) but is
     well-defined at the other six nodes. We check the rigid data is as expected
     at these locations.*/
    if (p_WN == Vector3d(1.0, 1.0, 1.0))
      continue;
    else if (p_WN == Vector3d(0.5, 1.0, 1.0)) {
      EXPECT_EQ(data.nhat_W, Vector3d(-1.0, 0.0, 0.0));
      const Vector3d p_BQ = Vector3d(-0.5, 0.0, 0.0);
      EXPECT_EQ(data.rigid_v, v_WB + w_WB.cross(p_BQ));
      EXPECT_EQ(data.rigid_v, Vector3d(-1.0, -1.0, 0.0));
      /* vn = 2.0, ||vt|| = sqrt(2), in the friction cone. */
      EXPECT_EQ(data.v, Vector3d(-1.0, -1.0, 0.0));
    } else if (p_WN == Vector3d(1.5, 1.0, 1.0)) {
      EXPECT_EQ(data.nhat_W, Vector3d(1.0, 0.0, 0.0));
      const Vector3d p_BQ = Vector3d(0.5, 0.0, 0.0);
      EXPECT_EQ(data.rigid_v, v_WB + w_WB.cross(p_BQ));
      EXPECT_EQ(data.rigid_v, Vector3d(-1.0, 1.0, 0.0));
      /* vn = -2.0, separating. */
      EXPECT_EQ(data.v, Vector3d(1.0, 0.0, 1.0));
    } else if (p_WN == Vector3d(1.0, 1.5, 1.0)) {
      EXPECT_EQ(data.nhat_W, Vector3d(0.0, 1.0, 0.0));
      const Vector3d p_BQ = Vector3d(0.0, 0.5, 0.0);
      EXPECT_EQ(data.rigid_v, v_WB + w_WB.cross(p_BQ));
      EXPECT_EQ(data.rigid_v, Vector3d(-2.0, 0.0, 0.0));
      /* vn = 0.0, ||vt|| = sqrt(5), out of the friction cone. */
      EXPECT_EQ(data.v, Vector3d(1.0, 0.0, 1.0));
    } else if (p_WN == Vector3d(1.0, 0.5, 1.0)) {
      EXPECT_EQ(data.nhat_W, Vector3d(0.0, -1.0, 0.0));
      const Vector3d p_BQ = Vector3d(0.0, -0.5, 0.0);
      EXPECT_EQ(data.rigid_v, v_WB + w_WB.cross(p_BQ));
      EXPECT_EQ(data.rigid_v, Vector3d(0.0, 0.0, 0.0));
      /* vn = 0.0, ||vt|| = sqrt(2), out of the friction cone. */
      EXPECT_EQ(data.v, Vector3d(1.0, 0.0, 1.0));
    } else if (p_WN == Vector3d(1.0, 1.0, 1.5)) {
      EXPECT_EQ(data.nhat_W, Vector3d(0.0, 0.0, 1.0));
      const Vector3d p_BQ = Vector3d(0.0, 0.0, 0.5);
      EXPECT_EQ(data.rigid_v, v_WB + w_WB.cross(p_BQ));
      EXPECT_EQ(data.rigid_v, Vector3d(-1.0, 0.0, 0.0));
      /* vn = 1.0, separating. */
      EXPECT_EQ(data.v, Vector3d(1.0, 0.0, 1.0));
    } else if (p_WN == Vector3d(1.0, 1.0, 0.5)) {
      EXPECT_EQ(data.nhat_W, Vector3d(0.0, 0.0, -1.0));
      const Vector3d p_BQ = Vector3d(0.0, 0.0, -0.5);
      EXPECT_EQ(data.rigid_v, v_WB + w_WB.cross(p_BQ));
      EXPECT_EQ(data.rigid_v, Vector3d(-1.0, 0.0, 0.0));
      /* vn = 1.0, ||vt|| = 2, outside of the friction cone. */
      EXPECT_EQ(data.v, Vector3d(0.0, 0.0, 0.0));
    } else {
      EXPECT_EQ(data.nhat_W, Vector3d::Zero());
      EXPECT_EQ(data.rigid_v, Vector3d::Zero());
      EXPECT_EQ(data.v, Vector3d(1.0, 0.0, 1.0));
    }
  }

  /* At this point, rigid_forces stores the spatial momentum of the rigid bodies
   (about the origins of each body) acquired from contact with the grid. We
   accumulate them in h_WO, the total spatial momentum transferred to the rigid
   bodies (about the world origin). */
  SpatialMomentum<double> h_WO = SpatialMomentum<double>::Zero();
  for (const auto& rigid_force : rigid_forces) {
    SpatialMomentum<double> h_WB;
    h_WB.translational() = rigid_force.F_Bq_W.translational();
    h_WB.rotational() = rigid_force.F_Bq_W.rotational();
    const Vector3d p_WB = rigid_force.p_BoBq_B;
    h_WO += h_WB.Shift(-p_WB);
  }

  /* Verify that the momentum is conserved. */
  const MassAndMomentum<double> post_contact_grid_momentum =
      grid.ComputeTotalMassAndMomentum();
  EXPECT_EQ(post_contact_grid_momentum.mass, initial_momentum.mass);
  EXPECT_TRUE(CompareMatrices(
      post_contact_grid_momentum.linear_momentum + h_WO.translational(),
      initial_momentum.linear_momentum,
      4.0 * std::numeric_limits<double>::epsilon()));
  EXPECT_TRUE(CompareMatrices(
      post_contact_grid_momentum.angular_momentum + h_WO.rotational(),
      initial_momentum.angular_momentum,
      4.0 * std::numeric_limits<double>::epsilon()));
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
