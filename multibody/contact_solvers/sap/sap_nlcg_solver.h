#pragma once

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/multibody/contact_solvers/sap/sap_model.h"
#include "drake/multibody/contact_solvers/sap/sap_solver.h"
#include "drake/multibody/contact_solvers/sap/sap_solver_results.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

// Parameters for a nonlinear conjugate-gradient (NLCG) SAP solve.
//
// This solver minimizes SAP's strictly convex primal objective
//   ℓ(v) = 1/2 (v-v*)ᵀ A (v-v*) + rᵀ (v-v*) + ℓc(Jv)
// using a Polak–Ribière+ (PR+) update and either a strong-Wolfe or an exact
// (one-dimensional) line search.
struct SapNlcgSolverParameters {
  enum class LineSearchType {
    kStrongWolfe,
    kExact,
  };

  enum class PreconditionerType {
    // uses a simple Jacobi preconditioner M ~ diag(A).
    kJacobi,
    // uses an (approximate) Hessian Jacobi preconditioner
    //   M ~ diag(H) ~ diag(A) + diag(J^T diag(G) J).
    // This option is intended to improve conditioning when contact
    // stiffness dominates.
    kDiagH,
    kNone,
  };

  // Parameters for the exact line search.
  // Ignored if line_search_type != LineSearchType::kExact.
  struct ExactLineSearchParameters {
    int max_iterations{100};
    double alpha_max{1.5};
  };

  // Strong-Wolfe line search parameters.
  // Ignored if line_search_type != LineSearchType::kStrongWolfe.
  struct StrongWolfeLineSearchParameters {
    int max_iterations{25};
    double c1{1.0e-4};        // sufficient decrease.
    double c2{0.9};           // strong curvature.
    double alpha0{1.0};       // initial trial step.
    double alpha_max{2.0};    // maximum step.
    double min_alpha{1.0e-16};
    double expansion{2.0};    // trial step expansion factor.
  };

  LineSearchType line_search_type{LineSearchType::kExact};
  ExactLineSearchParameters exact_line_search;
  StrongWolfeLineSearchParameters strong_wolfe;

  // Optimality condition: same definition as SapSolver (scaled momentum
  // residual). See SapSolverParameters for rationale.
  double abs_tolerance{1.e-14};
  double rel_tolerance{1.e-2};

  // Cost-stall condition (round-off detection).
  double cost_abs_tolerance{1.e-30};
  double cost_rel_tolerance{1.e-15};

  int max_iterations{200};

  // Restart heuristic threshold in PR+: restart if
  //   g_{k+1}ᵀ g_k / (g_kᵀ g_k) > restart_threshold.
  double restart_threshold{0.2};

  PreconditionerType preconditioner_type{PreconditionerType::kDiagH};
  // Minimum diagonal used to invert diag(H) (safety clamp).
  double diag_h_min_diagonal{1.0e-30};

  // Monotonicity sanity check slop.
  double relative_slop{1000 * std::numeric_limits<double>::epsilon()};
  bool nonmonotonic_convergence_is_error{false};
};

// Nonlinear conjugate-gradient solver for SAP's convex stage-2 minimization.
//
// Currently, only T = double is fully supported when constraints are non-empty.
// @tparam_nonsymbolic_scalar
template <typename T>
class SapNlcgSolver {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SapNlcgSolver);

  struct SolverStats {
    void Reset() {
      num_iters = 0;
      num_line_search_iters = 0;
      optimality_criterion_reached = false;
      cost_criterion_reached = false;
      momentum_residual.clear();
      momentum_scale.clear();
      cost.clear();
      alpha.clear();
    }
    int num_iters{0};
    int num_line_search_iters{0};
    bool optimality_criterion_reached{false};
    bool cost_criterion_reached{false};
    std::vector<double> cost;
    std::vector<double> alpha;
    std::vector<double> momentum_residual;
    std::vector<double> momentum_scale;
  };

  SapNlcgSolver() = default;

  SapSolverStatus SolveWithGuess(const SapContactProblem<T>& problem,
                                     const VectorX<T>& v_guess,
                                     SapSolverResults<T>* result);

  void set_parameters(const SapNlcgSolverParameters& parameters);

  const SolverStats& get_statistics() const;

 private:
  // @pre context was created by the underlying SapModel.
  void PackSapSolverResults(const systems::Context<T>& context,
                            SapSolverResults<T>* results) const;

  // @pre context was created by the underlying SapModel.
  void CalcStoppingCriteriaResidual(const systems::Context<T>& context,
                                    T* momentum_residual,
                                    T* momentum_scale) const;

  // Strong-Wolfe line search along `direction`, starting at the state stored
  // in `context`. Uses `scratch` for trial evaluations.
  // @returns (alpha, num_iterations).
  std::pair<double, int> PerformStrongWolfeLineSearch(
      const systems::Context<double>& context,
      const VectorX<double>& direction, double phi0, double dphi0,
      systems::Context<double>* scratch) const;

  // Exact line search by finding the root of dℓ(α)/dα.
  // @returns (alpha, num_iterations).
  std::pair<double, int> PerformExactLineSearch(
      const systems::Context<double>& context,
      const VectorX<double>& direction,
      systems::Context<double>* scratch) const;


  // Armijo backtracking fallback.
  std::pair<double, int> PerformBacktrackingArmijo(
      const systems::Context<double>& context,
      const VectorX<double>& direction, double phi0, double dphi0,
      systems::Context<double>* scratch) const;

  // Zoom phase for strong-Wolfe line search.
  double StrongWolfeZoom(const systems::Context<double>& context,
                         const VectorX<double>& direction, double phi0,
                         double dphi0, double alpha_lo, double alpha_hi,
                         double phi_lo, double phi_hi,
                         systems::Context<double>* scratch,
                         int* iters) const;

  // Evaluates (phi(alpha), dphi(alpha)) for v(alpha) = v + alpha*direction.
  std::pair<double, double> EvalPhiAndDeriv(
      const systems::Context<double>& context,
      const VectorX<double>& direction, double alpha,
      systems::Context<double>* scratch) const;

  std::unique_ptr<SapModel<T>> model_;
  SapNlcgSolverParameters parameters_;
  mutable SolverStats stats_;
};

// Specializations.

using drake::systems::Context;

template <>
SapSolverStatus SapNlcgSolver<double>::SolveWithGuess(
    const SapContactProblem<double>&, const VectorX<double>&,
    SapSolverResults<double>*);
  
template <>
std::pair<double, double> SapNlcgSolver<double>::EvalPhiAndDeriv(
    const Context<double>& context, const VectorX<double>& direction,
    double alpha, Context<double>* scratch) const;

template <>
std::pair<double, int> SapNlcgSolver<double>::PerformBacktrackingArmijo(
    const Context<double>& context, const VectorX<double>& direction,
    double phi0, double dphi0, Context<double>* scratch) const;

template <>
double SapNlcgSolver<double>::StrongWolfeZoom(
    const Context<double>& context, const VectorX<double>& direction,
    double phi0, double dphi0, double alpha_lo, double alpha_hi,
    double phi_lo, double phi_hi, Context<double>* scratch, int* iters) const;

template <>
std::pair<double, int> SapNlcgSolver<double>::PerformStrongWolfeLineSearch(
    const Context<double>& context, const VectorX<double>& direction,
    double phi0, double dphi0, Context<double>* scratch) const;

template <>
std::pair<double, int> SapNlcgSolver<double>::PerformExactLineSearch(
      const systems::Context<double>& context,
      const VectorX<double>& direction,
      systems::Context<double>* scratch) const;

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::internal::SapNlcgSolver);
