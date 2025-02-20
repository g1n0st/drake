#include "drake/examples/multibody/deformable/mpm_cloth_shared.h"
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
#include "drake/visualization/visualization_config.h"
#include "drake/visualization/visualization_config_functions.h"

DEFINE_bool(write_files, false, "Enable dumping MPM data to files.");
DEFINE_double(simulation_time, 14.0, "Desired duration of the simulation [s].");
DEFINE_int32(res, 30, "Cloth Resolution.");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 5e-3,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(substep, 5e-4,
              "Discrete time step for the substepping scheme [s]. Must be positive.");
DEFINE_double(stiffness, 1000.0, "Contact Stiffness.");
DEFINE_double(friction, 0.2, "Contact Friction.");
DEFINE_double(damping, 1.0,
    "Hunt and Crossley damping for the deformable body, only used when "
    "'contact_approximation' is set to 'lagged' or 'similar' [s/m].");

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
using drake::multibody::PackageMap;
using drake::systems::BasicVector;
using drake::systems::Context;
using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using drake::multibody::gmpm::MpmConfigParams;

namespace drake {
namespace examples {
namespace {

bool use_mpm_gripper = true;
bool add_free_box = true;

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
     this->DeclareVectorOutputPort(
         "AllegroDesiredState", drake::systems::BasicVector<double>(size_),
         &HandPoseController::CalcDesiredState, {this->time_ticket()});
   }
 
   void CalcDesiredState(const Context<double>& context,
                         drake::systems::BasicVector<double>* output) const {
     Eigen::VectorXd positions = GetHomePosition();
     double t = context.get_time();
     if (t < 0.5) {
     } else if (t < 1.0){
       // start gripping red box
       double dt = std::min(std::max((context.get_time() - 0.5) / 0.5, 0.0), 1.0);
       positions = (1.0 - dt) * GetHomePosition() + dt * GetGripPosition();
     } else if (t < 2.5) {
       positions = GetGripPosition();
     } else if (t < 4.0) {
       // loose hand to put red box down
       double dt = std::min(std::max((context.get_time() - 2.5) / 0.5, 0.0), 0.7);
       positions = (1.0 - dt) * GetGripPosition() + dt * GetHomePosition();
     } else if (t < 5.0){
      // start gripping blue box
      double dt = std::min(std::max((context.get_time() - 4.5) / 0.5, 0.3), 1.0);
      positions = (1.0 - dt) * GetHomePosition() + dt * GetGripPosition();
    } else if (t < 6.5) {
      positions = GetGripPosition();
    } else if (t < 7.0) {
      // loose hand to put blue box down
      double dt = std::min(std::max((context.get_time() - 6.5) / 0.5, 0.0), 0.5);
      positions = (1.0 - dt) * GetGripPosition() + dt * GetHomePosition();
    } else {
      positions = (1.0 - 0.5) * GetGripPosition() + 0.5 * GetHomePosition();
    }
     Eigen::VectorXd q_and_v(32);
     q_and_v << positions, GetHomeVelocity();
     output->set_value(q_and_v);
   } 
 
   Eigen::VectorXd GetHomePosition() const {
     Eigen::VectorXd pos(16);
     pos.setZero();
     pos(0) = 1.4;
     pos(1) = 0.25;
     return pos;
   }
 
   Eigen::VectorXd GetHomeVelocity() const { return Eigen::VectorXd::Zero(16); }
 
   Eigen::VectorXd GetGripPosition() const {
     Eigen::VectorXd vec(16);
     vec << 1.4, 0.25, 0.26, 1.22, -0.11, 0.54, 0.88, 0.93, 0.0, 0.54, 0.88,
         0.93, 0.12, 0.54, 0.88, 0.93;
     return (vec);
   }
 
   std::vector<std::string> GetPreferredJointOrdering() {
     std::vector<std::string> joint_name_mapping;
     // Thumb finger
     joint_name_mapping.push_back("joint_12");
     joint_name_mapping.push_back("joint_13");
     joint_name_mapping.push_back("joint_14");
     joint_name_mapping.push_back("joint_15");
     // Index finger
     joint_name_mapping.push_back("joint_0");
     joint_name_mapping.push_back("joint_1");
     joint_name_mapping.push_back("joint_2");
     joint_name_mapping.push_back("joint_3");
     // Middle finger
     joint_name_mapping.push_back("joint_4");
     joint_name_mapping.push_back("joint_5");
     joint_name_mapping.push_back("joint_6");
     joint_name_mapping.push_back("joint_7");
     // Ring finger
     joint_name_mapping.push_back("joint_8");
     joint_name_mapping.push_back("joint_9");
     joint_name_mapping.push_back("joint_10");
     joint_name_mapping.push_back("joint_11");
     return joint_name_mapping;
   }
 
  private:
   const multibody::MultibodyPlant<double>& plant_;
   int size_ = 16 * 2;  // 4 fingers with 4 joints
   double grip_time_ = 0.3;
   double prep_time_ = 0.4 + 1.0;
}; 

class IiwaController : public drake::systems::LeafSystem<double> {
  public:
   IiwaController(const multibody::MultibodyPlant<double>& plant,
                  const RigidTransformd& init_pose)
       : plant_(plant) {
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
     // Eigen::VectorXd current_robot_qv =
     //     this->get_input_port(robot_state_index_).Eval(context);
     const VectorXd xd = context.get_discrete_state().value();
     // xd = [rpy, translation]
     *value = FromXyzRpy(xd.segment(0, 3), xd.segment(3, 3));
   }
 
   void Update(const Context<double>& context,
               systems::DiscreteValues<double>* next_states) const {
     const VectorX<double>& current_state_values =
         context.get_discrete_state().value();
     unused(current_state_values);
 
     // fake update:
     VectorX<double> dX = current_state_values;
     dX.setZero();
     double t = context.get_time();
     double rate = plant_.time_step() / 0.01;
     // NOTE (changyu): this rate is used to control the height of gripper
     double uprt = 0.4 / 0.3;
     if (t < 0.5) {
       dX(5) -= 0.002 * rate; // move down
     } else if (t < 1.0) {
       // hold
     } else if (t < 1.5) {
      dX(5) += 0.00225 * rate * uprt; // move up
     } else if (t < 2.5) {
      dX(5) += 0.002 * rate * uprt; // move up
      dX(4) -= 0.0042 * rate; // move left
      dX(3) -= 0.0009 * rate; // move inward
     } else if (t < 3.0) {
      // hold
     } else if (t < 4.0) {
      if (t < 3.5) {
        dX(5) += 0.002 * rate * uprt; // move up 
      } else {
        dX(5) -= 0.002 * rate * uprt; // move up
      }
      dX(5) -= 0.0019 * rate * uprt; // move down 
      dX(4) += 0.0042 * rate; // move right
      dX(4) += 0.002 * rate; // move extra 20cm to get right up of the blue box
      if (t > 3.7) {
        dX(3) += 0.0009 * rate / 1.5 * 5; // move outward
      }
     } else if (t < 4.5) {
      dX(5) -= 0.00225 * rate * uprt; // move down
     } else if (t < 5.0) {
      // hold
     } else if (t < 5.5) {
      dX(5) += 0.003 * rate * uprt; // move up
     } else if (t < 6.5) {
        dX(5) += 0.0018 * rate * uprt; // move up 
        dX(4) -= 0.0048 * rate; // move left
        dX(4) -= 0.0018 * rate; // move extra 18cm
        dX(3) -= 0.0005 * rate; // move inward
     } else if (t < 7.0) {
        // hold
     } else if (t < 9.0) {
      // if (t < 8.0) {
      //   dX(5) += 0.002 * rate / 4.0;
      // } else {
      //   dX(5) -= 0.002 * rate / 4.0;
      // }
      dX(4) += 0.0084 * rate / 4.0; // move right
      dX(4) += 0.004 * rate / 4.0; // move extra 20cm to get right up of the blue box
      dX(3) += 0.0015 * rate / 4.0; // move outward
     } else {
      dX.setZero();
     }
     auto new_value = current_state_values + dX;
     next_states->set_value(new_value);
   }
 
  private:
   const multibody::MultibodyPlant<double>& plant_;
   int robot_state_index_{};
 
   Eigen::Vector3d trans_;
   Eigen::Vector3d rpy_;
}; 

class BaggingGripperController : public systems::LeafSystem<double> {
  public:
   BaggingGripperController() {
     this->DeclareVectorOutputPort("desired state", BasicVector<double>(48),
                                    &BaggingGripperController::CalcDesiredState);
   }
  
  static constexpr double visual_coeff = 0.8;
  static constexpr double gripper_xy = 0.1 * visual_coeff; // visual effect to make gripper looks less penetration
  static constexpr double gripper_z = 0.04;
  static constexpr double gripper_density = 10000.0;
 
  static constexpr double l_x = 0.34 - 0.03 - 0.01;
  static constexpr double h_x = 0.66 + 0.027 + 0.01;
  static constexpr double l_z = 0.39-2e-4 - 0.01;
  static constexpr double h_z = 0.41+2e-4 + 0.01;
 
  static constexpr double initial_free_duration = 0.25;
  static constexpr double initial_loose_duration = 0.25;
  static constexpr double free_duration = 8.0;
  static constexpr double bagging_duration = 1.0 - initial_loose_duration;
  static constexpr double static_duration = 1.0;
  static constexpr double final_loose_duration = 0.4;
  static constexpr double bagging_v = 0.1;
 
  static ModelInstanceIndex AddGripperInstance(MultibodyPlant<double>* plant, ProximityProperties rigid_proximity_props) {
   IllustrationProperties illustration_props_up;
   illustration_props_up.AddProperty("phong", "diffuse", Vector4d(0.6, 0.6, 0.6, 0.8));
   IllustrationProperties illustration_props_down;
   illustration_props_down.AddProperty("phong", "diffuse", Vector4d(0.4, 0.4, 0.4, 0.8));
 
   Box gripper_shape(gripper_xy, gripper_xy, gripper_z);
   const auto &gripper_inertia = SpatialInertia<double>::SolidBoxWithDensity(gripper_density, gripper_xy, gripper_xy, gripper_z);
 
   ModelInstanceIndex gripper_instance = plant->AddModelInstance("gripper_instance");
 
   const auto &add_single_gripper = [&](std::string name, double x, double y, double z, bool is_up) {
     const RigidBody<double>& x_body = plant->AddRigidBody(name + "_x", gripper_instance, gripper_inertia);
     const auto& x_joint = plant->AddJoint<PrismaticJoint>(name + "_x", plant->world_body(), 
           RigidTransformd::Identity(), x_body, std::nullopt, Vector3d::UnitX());
 
     const RigidBody<double>& y_body = plant->AddRigidBody(name + "_y", gripper_instance, gripper_inertia);
     const auto& y_joint = plant->AddJoint<PrismaticJoint>(name + "_y", x_body, 
           RigidTransformd::Identity(), y_body, std::nullopt, Vector3d::UnitY());
 
     const RigidBody<double>& z_body = plant->AddRigidBody(name + "_z", gripper_instance, gripper_inertia);
     const auto& z_joint = plant->AddJoint<PrismaticJoint>(name + "_z", y_body, 
           RigidTransformd::Identity(), z_body, std::nullopt, Vector3d::UnitZ());
 
     if (use_mpm_gripper) {
      unused(rigid_proximity_props);
     } else {
      plant->RegisterCollisionGeometry(z_body, RigidTransformd::Identity(), gripper_shape, name + "_collision", rigid_proximity_props);
     }
     if (is_up) {
        plant->RegisterVisualGeometry   (z_body, RigidTransformd::Identity(), gripper_shape, name + "_visual"   , illustration_props_up);
     } else {
        plant->RegisterVisualGeometry   (z_body, RigidTransformd::Identity(), gripper_shape, name + "_visual"   , illustration_props_down);
     }
 
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
 
   add_single_gripper("gll_up",  l_x, l_x, h_z, true);
   add_single_gripper("glh_up", l_x, h_x, h_z, true);
   add_single_gripper("ghl_up",  h_x, l_x,  h_z, true);
   add_single_gripper("ghh_up", h_x, h_x, h_z, true);
   add_single_gripper("gll_down",  l_x, l_x, l_z, false);
   add_single_gripper("glh_down", l_x, h_x, l_z, false);
   add_single_gripper("ghl_down",  h_x, l_x,  l_z, false);
   add_single_gripper("ghh_down", h_x, h_x, l_z, false);
 
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
     if (t < initial_free_duration) {
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
    }
     else if (t < initial_free_duration + initial_loose_duration) {
      double dt = t - initial_free_duration;
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
    }
     else if (t < free_duration + initial_loose_duration + initial_free_duration) {
      double dt = initial_loose_duration;
      gll_up_p = Vector3d(l_x + dt * bagging_v, l_x + dt * bagging_v, h_z);
      glh_up_p = Vector3d(l_x + dt * bagging_v, h_x - dt * bagging_v, h_z);
      ghl_up_p = Vector3d(h_x - dt * bagging_v, l_x + dt * bagging_v,  h_z);
      ghh_up_p = Vector3d(h_x - dt * bagging_v, h_x - dt * bagging_v, h_z);
      gll_down_p = Vector3d(l_x + dt * bagging_v, l_x + dt * bagging_v, l_z);
      glh_down_p = Vector3d(l_x + dt * bagging_v, h_x - dt * bagging_v, l_z);
      ghl_down_p = Vector3d(h_x - dt * bagging_v, l_x + dt * bagging_v,  l_z);
      ghh_down_p = Vector3d(h_x - dt * bagging_v, h_x - dt * bagging_v, l_z);
 
       gll_up_v = Vector3d(0, 0, 0);
       glh_up_v = Vector3d(0, 0, 0);
       ghl_up_v = Vector3d(0, 0, 0);
       ghh_up_v = Vector3d(0, 0, 0);
       gll_down_v = Vector3d(0, 0, 0);
       glh_down_v = Vector3d(0, 0, 0);
       ghl_down_v = Vector3d(0, 0, 0);
       ghh_down_v = Vector3d(0, 0, 0);
     } else if (t < free_duration + bagging_duration + initial_loose_duration + initial_free_duration) {
       double dt = (t - free_duration - initial_free_duration);
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
     } else if (t < free_duration + bagging_duration + initial_loose_duration + initial_free_duration + static_duration) {
        double total_dur = initial_loose_duration + bagging_duration;
       gll_up_p = Vector3d(l_x + total_dur * bagging_v, l_x + total_dur * bagging_v, h_z);
       glh_up_p = Vector3d(l_x + total_dur * bagging_v, h_x - total_dur * bagging_v, h_z);
       ghl_up_p = Vector3d(h_x - total_dur * bagging_v, l_x + total_dur * bagging_v,  h_z);
       ghh_up_p = Vector3d(h_x - total_dur * bagging_v, h_x - total_dur * bagging_v, h_z);
       gll_down_p = Vector3d(l_x + total_dur * bagging_v, l_x + total_dur * bagging_v, l_z);
       glh_down_p = Vector3d(l_x + total_dur * bagging_v, h_x - total_dur * bagging_v, l_z);
       ghl_down_p = Vector3d(h_x - total_dur * bagging_v, l_x + total_dur * bagging_v,  l_z);
       ghh_down_p = Vector3d(h_x - total_dur * bagging_v, h_x - total_dur * bagging_v, l_z);
 
       gll_up_v = Vector3d(0, 0, 0);
       glh_up_v = Vector3d(0, 0, 0);
       ghl_up_v = Vector3d(0, 0, 0);
       ghh_up_v = Vector3d(0, 0, 0);
       gll_down_v = Vector3d(0, 0, 0);
       glh_down_v = Vector3d(0, 0, 0);
       ghl_down_v = Vector3d(0, 0, 0);
       ghh_down_v = Vector3d(0, 0, 0);
     } else {
      double total_dur = initial_loose_duration + bagging_duration;
      double dt = 100.0;
      gll_up_p = Vector3d(l_x + total_dur * bagging_v, l_x + total_dur * bagging_v, h_z);
      glh_up_p = Vector3d(l_x + total_dur * bagging_v, h_x - total_dur * bagging_v, h_z);
      ghl_up_p = Vector3d(h_x - (total_dur - dt) * bagging_v, l_x + (total_dur - dt) * bagging_v,  h_z + dt * bagging_v);
      ghh_up_p = Vector3d(h_x - (total_dur - dt) * bagging_v, h_x - (total_dur - dt) * bagging_v, h_z + dt * bagging_v);
      gll_down_p = Vector3d(l_x + total_dur * bagging_v, l_x + total_dur * bagging_v, l_z);
      glh_down_p = Vector3d(l_x + total_dur * bagging_v, h_x - total_dur * bagging_v, l_z);
      ghl_down_p = Vector3d(h_x - (total_dur - dt) * bagging_v, l_x + (total_dur - dt) * bagging_v,  l_z - dt * bagging_v);
      ghh_down_p = Vector3d(h_x - (total_dur - dt) * bagging_v, h_x - (total_dur - dt) * bagging_v, l_z - dt * bagging_v);

      gll_up_v = Vector3d(0, 0, 0);
       glh_up_v = Vector3d(0, 0, 0);
       ghl_up_v = Vector3d(bagging_v, -bagging_v, + bagging_v);
       ghh_up_v = Vector3d(bagging_v, +bagging_v, + bagging_v);
       gll_down_v = Vector3d(0, 0, 0);
       glh_down_v = Vector3d(0, 0, 0);
       ghl_down_v = Vector3d(bagging_v, -bagging_v, - bagging_v);
       ghh_down_v = Vector3d(bagging_v, +bagging_v, - bagging_v);
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
  plant_config.discrete_contact_approximation = "lagged";

  ProximityProperties rigid_proximity_props;
  ProximityProperties ground_proximity_props;
  const CoulombFriction<double> surface_friction(1.0, 1.0);
  AddCompliantHydroelasticProperties(1.0, 1e6, &rigid_proximity_props);
  AddRigidHydroelasticProperties(1.0, &ground_proximity_props);
  AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
  AddContactMaterial({}, {}, surface_friction, &ground_proximity_props);

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  // set up table and ground
  {
    /* Set up a ground. */
    Box ground{10, 10, 10};
    const RigidTransformd X_WG(Eigen::Vector3d{0, 0, -5 + 0.02});
    plant.RegisterCollisionGeometry(plant.world_body(), X_WG, ground,
                                    "ground_collision", rigid_proximity_props);
  }
  multibody::Parser ground_parser(&plant, "ground");
  const std::string table_file = FindResourceOrThrow(
      "drake/examples/multibody/deformable/"
      "models/table_wide.sdf");
  auto table = ground_parser.AddModels(table_file)[0];
  plant.WeldFrames(plant.world_frame(),
                   plant.GetBodyByName("table_body", table).body_frame(),
                   RigidTransformd(Eigen::Vector3d(0.25, 0.29 + 0.5, 0.02)));
  
  // free box
  const Vector4<double> red(1.0, 0.0, 0.0, 1.0);
  const Vector4<double> blue(0.0, 0.0, 1.0, 1.0);
  double box_width = 0.08;
  double visual_coeff = 0.9; // get less penetration visually
  if (add_free_box) {
      ModelInstanceIndex free_body_model_instance1 =
          plant.AddModelInstance("free_body_instance1");
      const SpatialInertia<double> free_body_box_spatial1 =
          SpatialInertia<double>::SolidBoxWithDensity(1000.0, box_width,
                                                      box_width, box_width);
      const RigidBody<double>& free_box1 = plant.AddRigidBody(
          "free_box1", free_body_model_instance1, free_body_box_spatial1);
        
        plant.RegisterVisualGeometry(free_box1, RigidTransformd::Identity(),
        Box(box_width * visual_coeff, box_width * visual_coeff, box_width * visual_coeff),
        "FreeCubeV1", red);
      plant.RegisterCollisionGeometry(free_box1, RigidTransformd::Identity(),
              Box(box_width, box_width, box_width),
              "FreeCube1", rigid_proximity_props);
      
        ModelInstanceIndex free_body_model_instance2 =
        plant.AddModelInstance("free_body_instance2");
    const SpatialInertia<double> free_body_box_spatial2 =
        SpatialInertia<double>::SolidBoxWithDensity(1000.0, box_width,
                                                    box_width, box_width);
      const RigidBody<double>& free_box2 = plant.AddRigidBody(
        "free_box2", free_body_model_instance2, free_body_box_spatial2);
      
      plant.RegisterVisualGeometry(free_box2, RigidTransformd::Identity(),
      Box(box_width * visual_coeff, box_width * visual_coeff, box_width * visual_coeff),
      "FreeCubeV2", blue);
    plant.RegisterCollisionGeometry(free_box2, RigidTransformd::Identity(),
            Box(box_width, box_width, box_width),
            "FreeCube2", rigid_proximity_props);
  }

  MultibodyPlant<double> iiwa_controller_plant =
      MultibodyPlant<double>(plant_config.time_step);

  multibody::Parser parser(&plant);
  const std::string filename = PackageMap{}.ResolveUrl("package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf");
  std::vector<drake::multibody::ModelInstanceIndex> instances =
      parser.AddModels(filename);
  auto iiwa = instances[0];
  Parser(&iiwa_controller_plant).AddModels(filename);

  std::string hand_filename = PackageMap{}.ResolveUrl("package://drake_models/allegro_hand_description/sdf/allegro_hand_description_right.sdf");
  std::vector<drake::multibody::ModelInstanceIndex> instances2 =
      parser.AddModels(hand_filename);
  auto allegro = instances2[0];

  RigidTransformd iiwa_position(Eigen::Vector3d(0, 1, 0));

  plant.WeldFrames(plant.world_frame(),
                   plant.GetBodyByName("iiwa_link_0").body_frame(),
                   iiwa_position);
  iiwa_controller_plant.WeldFrames(
      iiwa_controller_plant.world_frame(),
      iiwa_controller_plant.GetBodyByName("iiwa_link_0").body_frame(),
      iiwa_position);

  plant.WeldFrames(
      plant.GetBodyByName("iiwa_link_7").body_frame(),
      plant.GetBodyByName("hand_root").body_frame(),
      FromXyzRpyDegree(Eigen::Vector3d(0, -45, 0), Eigen::Vector3d(0, 0, 0)));

  // mpm stuff
  DeformableModel<double>& deformable_model = plant.mutable_deformable_model();
  AddCloth(&deformable_model, FLAGS_res, 0.4);

  MpmConfigParams mpm_config;
  mpm_config.substep_dt = FLAGS_substep;
  mpm_config.write_files = FLAGS_write_files;
  mpm_config.contact_stiffness = FLAGS_stiffness;
  mpm_config.contact_damping = FLAGS_damping;
  mpm_config.contact_friction_mu = FLAGS_friction;
  mpm_config.contact_query_frequency = 8;
  if (use_mpm_gripper) {
    mpm_config.mpm_bc = 114;
  } else {
    mpm_config.mpm_bc = -1;
  }
  mpm_config.ignore_face_contact = true;
  mpm_config.mdv_as_impulse = false;
  deformable_model.SetMpmConfig(std::move(mpm_config));

  const auto& gripper_instance = BaggingGripperController::AddGripperInstance(&plant, rigid_proximity_props);

  double Kp = 1000000.0;
  double Kd = 5.0 * std::sqrt(Kp);

  drake::multibody::PdControllerGains gain(Kp, Kd);
  for (int i = 0; i < plant.num_actuators(); ++i) {
    plant.get_mutable_joint_actuator(drake::multibody::JointActuatorIndex(i))
        .set_controller_gains(gain);
  }

  plant.Finalize();
  iiwa_controller_plant.Finalize();

  Eigen::VectorXd iiwa_initial_joint_values(7);
  iiwa_initial_joint_values << 0, 0.7, 0, -1.6, 0, 0.8, 0;
  // Eigen::VectorXd iiwa_velocity_limits(7);
  // iiwa_velocity_limits << 1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3;

  // hand controller
  auto hand_pose_controller = builder.template AddSystem<HandPoseController>(
      plant);  // put gripper timing here
  std::vector<std::string> preferred_joint_ordering =
      hand_pose_controller->GetPreferredJointOrdering();

  int nu_allegro = plant.num_actuated_dofs(allegro);
  int nq_allegro = plant.num_positions(allegro);
  int nv_allegro = plant.num_velocities(allegro);
  unused(nv_allegro);

  Eigen::MatrixXd state_selector =
      Eigen::MatrixXd::Zero(2 * nu_allegro, 2 * nu_allegro);
  int u = 0;
  std::vector<std::string> actuated_joints;
  for (int a = 0; a < plant.num_actuators(); ++a) {
    auto& actuator =
        plant.get_joint_actuator(drake::multibody::JointActuatorIndex(a));

    for (int i = 0; i < static_cast<int>(preferred_joint_ordering.size());
         ++i) {
      if (actuator.joint().name() == preferred_joint_ordering[i]) {
        actuated_joints.push_back(actuator.joint().name());
        state_selector(u, i) = 1;
        state_selector(nu_allegro + u, nq_allegro + i) = 1;
        ++u;
        break;
      }
    }
  }
  
  auto actuated_states_selector =
      builder.template AddSystem<drake::systems::MatrixGain<double>>(
          state_selector);
  builder.Connect(hand_pose_controller->get_output_port(),
                  actuated_states_selector->get_input_port());
  builder.Connect(actuated_states_selector->get_output_port(),
                  plant.get_desired_state_input_port(allegro));

  // iiwa controller
  // first find the initial pose for the current joint values
  std::unique_ptr<Context<double>> temp_context = plant.CreateDefaultContext();
  plant.SetPositions(temp_context.get(), iiwa, iiwa_initial_joint_values);
  RigidTransformd iiwa_controller_initial_pose =
      plant.GetBodyByName("iiwa_link_7")
          .body_frame()
          .CalcPoseInWorld(*(temp_context.get()));

  auto iiwa_controller = builder.template AddSystem<IiwaController>(
      plant, iiwa_controller_initial_pose);
  int nq_iiwa = plant.num_positions(iiwa);
  int nv_iiwa = plant.num_velocities(iiwa);
  int nu_iiwa = plant.num_actuated_dofs(iiwa);
  unused(nu_iiwa);

  DifferentialInverseKinematicsParameters params(nq_iiwa, nv_iiwa);
  params.set_nominal_joint_position(iiwa_initial_joint_values);

  auto diff_ik =
      builder.template AddSystem<DifferentialInverseKinematicsIntegrator>(
          iiwa_controller_plant,
          iiwa_controller_plant.GetFrameByName("iiwa_link_7"),
          plant_config.time_step, params);

  std::vector<int> input_sizes;
  input_sizes.push_back(nq_iiwa);
  input_sizes.push_back(nv_iiwa);
  auto mux = builder.template AddSystem<drake::systems::Multiplexer<double>>(
      input_sizes);

  auto zero_vs =
      builder.template AddSystem<drake::systems::ConstantVectorSource>(
          Eigen::VectorXd::Zero(nv_iiwa));
  builder.Connect(plant.get_state_output_port(iiwa),
                  iiwa_controller->robot_state_input_port());
  builder.Connect(iiwa_controller->get_output_port(),
                  diff_ik->GetInputPort("X_WE_desired"));
  builder.Connect(plant.get_state_output_port(iiwa),
                  diff_ik->GetInputPort("robot_state"));
  builder.Connect(diff_ik->GetOutputPort("joint_positions"),
                  mux->get_input_port(0));
  builder.Connect(zero_vs->get_output_port(), mux->get_input_port(1));
  builder.Connect(mux->get_output_port(),
                  plant.get_desired_state_input_port(iiwa));

  // bag controller
  builder.Connect(builder.AddSystem<BaggingGripperController>()->get_output_port(), plant.get_desired_state_input_port(gripper_instance));

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams visualize_params;
  visualize_params.show_mpm = drake::geometry::DrakeVisualizerParams::ShowMpmOpt::kClothMpm;
  auto& visualizer = geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph, nullptr, visualize_params);

  // NOTE (changyu): MPM shortcut port shuould be explicit connected for visualization.
  builder.Connect(plant.get_output_port(
    plant.deformable_model().mpm_output_port_index()), 
    visualizer.mpm_input_port());
  
  auto meshcat = std::make_shared<geometry::Meshcat>();
  auto meshcat_params = drake::geometry::MeshcatVisualizerParams();
  meshcat_params.show_mpm = drake::geometry::MeshcatVisualizerParams::ShowMpmOpt::kClothMpm;
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
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  auto& mutable_context = simulator.get_mutable_context();
  auto& plant_context = plant.GetMyMutableContextFromRoot(&mutable_context);
  //   auto& diff_ik_context =
  //       diff_ik->GetMyMutableContextFromRoot(&mutable_context);

  // set free box initial position
  if (add_free_box) {
    plant.SetFreeBodyPose(
        &plant_context, plant.GetBodyByName("free_box1"),
        math::RigidTransformd{Vector3d(0.6, 1.0, box_width / 2.0 + 0.05)});
    
    plant.SetFreeBodyPose(
      &plant_context, plant.GetBodyByName("free_box2"),
      math::RigidTransformd{Vector3d(0.6, 1.2, box_width / 2.0 + 0.05)});
  }

  plant.SetPositions(&plant_context, iiwa, iiwa_initial_joint_values);

  meshcat->StartRecording();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);
  meshcat->StopRecording();
  meshcat->PublishRecording();

  std::ofstream htmlFile("/home/changyu/drake/allegro_bagging.html");
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