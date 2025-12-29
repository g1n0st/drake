#include "drake/multibody/contact_solvers/sap/sap_nlcg_solver.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <iostream>

#include "drake/common/default_scalars.h"
#include "drake/multibody/contact_solvers/newton_with_bisection.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

using drake::systems::Context;

template <typename T>
void SapNlcgSolver<T>::set_parameters(const SapNlcgSolverParameters& p) {
  parameters_ = p;
}

template <typename T>
const typename SapNlcgSolver<T>::SolverStats& SapNlcgSolver<T>::
get_statistics() const {
  return stats_;
}

template <typename T>
void SapNlcgSolver<T>::PackSapSolverResults(const Context<T>& context,
                                            SapSolverResults<T>* results) const {
  DRAKE_DEMAND(results != nullptr);
  results->Resize(model_->problem().num_velocities(),
                  model_->num_constraint_equations());

  // For non-participating velocities the solutions is v = v*. Therefore we
  // first initialize to v = v* and overwrite with the non-trivial participating
  // values in the following line.
  results->v = model_->problem().v_star();
  const VectorX<T>& v_participating = model_->GetVelocities(context);
  model_->velocities_permutation().ApplyInverse(v_participating, &results->v);

  // Constraints equations are clustered (essentially their order is permuted
  // for a better sparsity structure). Therefore constraint velocities and
  // impulses are evaluated in this clustered order and permuted into the
  // original order described by the model right after.
  const VectorX<T>& vc_clustered = model_->EvalConstraintVelocities(context);
  model_->impulses_permutation().ApplyInverse(vc_clustered, &results->vc);
  const VectorX<T>& gamma_clustered = model_->EvalImpulses(context);
  model_->impulses_permutation().ApplyInverse(gamma_clustered, &results->gamma);

  // For non-participating velocities we have v=v* and the generalized impulses
  // are zero. Therefore we first zero-out all generalized impulses and
  // overwrite with the non-trivial non-zero values for the participating DOFs
  // right after.
  const VectorX<T>& tau_participating =
      model_->EvalGeneralizedImpulses(context);
  results->j.setZero();
  model_->velocities_permutation().ApplyInverse(tau_participating, &results->j);
}

template <typename T>
void SapNlcgSolver<T>::CalcStoppingCriteriaResidual(const Context<T>& context,
                                                    T* momentum_residual,
                                                    T* momentum_scale) const {
  using std::max;
  const VectorX<T>& inv_sqrt_A = model_->inv_sqrt_dynamics_matrix();
  const VectorX<T>& p = model_->EvalMomentum(context);
  const VectorX<T>& jc = model_->EvalGeneralizedImpulses(context);
  const VectorX<T>& ell_grad = model_->EvalCostGradient(context);

  // Scale generalized momentum quantities using inv_sqrt_A so that all entries
  // have the same units and we can weigh them equally.
  const VectorX<T> ell_grad_tilde = inv_sqrt_A.asDiagonal() * ell_grad;
  const VectorX<T> p_tilde = inv_sqrt_A.asDiagonal() * p;
  const VectorX<T> jc_tilde = inv_sqrt_A.asDiagonal() * jc;

  *momentum_residual = ell_grad_tilde.norm();
  *momentum_scale = max(p_tilde.norm(), jc_tilde.norm());
}

template <typename T>
SapSolverStatus SapNlcgSolver<T>::SolveWithGuess(
    const SapContactProblem<T>& problem, const VectorX<T>&,
    SapSolverResults<T>* results) {
  if (problem.num_constraints() == 0) {
    results->Resize(problem.num_velocities(), problem.num_constraint_equations());
    results->v = problem.v_star();
    results->j.setZero();
    return SapSolverStatus::kSuccess;
  }
  throw std::logic_error(
      "SapNlcgSolver::SolveWithGuess(): Only T = double is supported when the "
      "set of constraints is non-empty.");
}

namespace {
inline bool IsFinite(double x) {
  return std::isfinite(x);
}

struct NlcgSearchDirectionData {
  const VectorX<double>& dv;
  const VectorX<double>& dp;
  const VectorX<double>& dvc;
  double d2ellA_dalpha2{NAN};
};

// Mirrors SapSolver::CalcCostAlongLine() exactly (for T = double), but takes a
// direction (dv) provided externally (e.g. NLCG search direction).
double CalcCostAlongLine(
    const SapModel<double>& model, const Context<double>& context,
    const NlcgSearchDirectionData& search_direction_data, const double& alpha,
    Context<double>* scratch, double* dell_dalpha = nullptr,
    double* d2ell_dalpha2 = nullptr,
    VectorX<double>* d2ell_dalpha2_scratch = nullptr) {
  DRAKE_DEMAND(scratch != nullptr);
  DRAKE_DEMAND(scratch != &context);
  if (d2ell_dalpha2 != nullptr) DRAKE_DEMAND(d2ell_dalpha2_scratch != nullptr);

  // Data.
  const VectorX<double>& v_star = model.v_star();
  const VectorX<double>& r = model.momentum_bias();

  // Search direction quantities at state v.
  const VectorX<double>& dv = search_direction_data.dv;
  const VectorX<double>& dp = search_direction_data.dp;
  const VectorX<double>& dvc = search_direction_data.dvc;
  const double& d2ellA_dalpha2 = search_direction_data.d2ellA_dalpha2;

  // State at v(alpha).
  Context<double>& context_alpha = *scratch;
  const VectorX<double>& v = model.GetVelocities(context);
  model.GetMutableVelocities(&context_alpha) = v + alpha * dv;

  if (d2ell_dalpha2 != nullptr) {
    // Since it is more efficient to calculate impulses (gamma) and their
    // derivatives (G) together, this evaluation avoids calculating the impulses
    // twice.
    model.EvalConstraintsHessian(context_alpha);
  }

  // Update velocities and impulses at v(alpha).
  // N.B. This evaluation should be cheap given we called EvalConstraintsHessian()
  // at the very start of the scope of this function.
  const VectorX<double>& gamma = model.EvalImpulses(context_alpha);

  // Regularizer cost.
  const double ellR = model.EvalConstraintsCost(context_alpha);

  // Momentum cost. We use the O(n) strategy described in [Castro et al., 2021].
  double ellA = model.EvalMomentumCost(context);
  ellA += alpha * dp.dot(v - v_star);
  ellA += alpha * r.dot(dv);
  ellA += 0.5 * alpha * alpha * d2ellA_dalpha2;
  const double ell = ellA + ellR;

  // Compute first derivative.
  if (dell_dalpha != nullptr) {
    const VectorX<double>& v_alpha = model.GetVelocities(context_alpha);

    // Momentum term + linear residual term rᵀΔv.
    const double dellA_dalpha = dp.dot(v_alpha - v_star) + r.dot(dv);
    const double dellR_dalpha = -dvc.dot(gamma);  // Regularizer term.
    *dell_dalpha = dellA_dalpha + dellR_dalpha;
  }

  // Compute second derivative.
  if (d2ell_dalpha2 != nullptr) {
    // N.B. This evaluation should be cheap given we called EvalConstraintsHessian()
    // at the very start of the scope of this function.
    const std::vector<MatrixX<double>>& G =
        model.EvalConstraintsHessian(context_alpha);

    // First compute d2ell_dalpha2_scratch = G⋅Δvc.
    d2ell_dalpha2_scratch->resize(model.num_constraint_equations());
    const int nc = model.num_constraints();
    int constraint_start = 0;
    for (int i = 0; i < nc; ++i) {
      const MatrixX<double>& G_i = G[i];
      const int ni = G_i.rows();
      const auto dvc_i = dvc.segment(constraint_start, ni);
      d2ell_dalpha2_scratch->segment(constraint_start, ni) = G_i * dvc_i;
      constraint_start += ni;
    }

    const double d2ellR_dalpha2 = dvc.dot(*d2ell_dalpha2_scratch);
    *d2ell_dalpha2 = d2ellA_dalpha2 + d2ellR_dalpha2;

    // Sanity check these terms are all positive.
    DRAKE_DEMAND(d2ellR_dalpha2 >= 0.0);
    DRAKE_DEMAND(d2ellA_dalpha2 > 0.0);
    DRAKE_DEMAND(*d2ell_dalpha2 > 0.0);
  }

  return ell;
}

}  // namespace

template <>
SapSolverStatus SapNlcgSolver<double>::SolveWithGuess(
    const SapContactProblem<double>& problem, const VectorX<double>& v_guess,
    SapSolverResults<double>* results) {
  DRAKE_DEMAND(results != nullptr);

  if (problem.num_constraints() == 0) {
    results->Resize(problem.num_velocities(), problem.num_constraint_equations());
    results->v = problem.v_star();
    results->j.setZero();
    return SapSolverStatus::kSuccess;
  }

  // Build model.
  model_ = std::make_unique<SapModel<double>>(&problem);
  // const int nv = model_->num_velocities();

  auto context = model_->MakeContext();
  auto scratch = model_->MakeContext();
  stats_.Reset();

  // Set initial guess (participating velocities only).
  {
    Eigen::VectorBlock<VectorX<double>> v =
        model_->GetMutableVelocities(context.get());
    model_->velocities_permutation().Apply(v_guess, &v);
  }

  const bool use_diag_h_preconditioner = 
    parameters_.preconditioner_type == SapNlcgSolverParameters::PreconditionerType::kDiagH;
  const bool use_preconditioner = 
    parameters_.preconditioner_type != SapNlcgSolverParameters::PreconditionerType::kNone;

  // Optional Jacobi preconditioners.
  //  - diag(A)^{-1} (cheap, constant)
  //  - diag(H)^{-1} with H ~ A + J^T diag(G) J (more expensive, refreshed)
  VectorX<double> inv_diag_A;
  VectorX<double> diag_A;
  MatrixX<double> J_squared;
  if (use_preconditioner) {
    const VectorX<double>& inv_sqrt_A = model_->inv_sqrt_dynamics_matrix();
    inv_diag_A = inv_sqrt_A.array().square().matrix();
    if (use_diag_h_preconditioner) {
      diag_A = inv_diag_A.cwiseInverse();
      const MatrixX<double> J =
          model_->constraints_bundle().J().MakeDenseMatrix();
      J_squared = J.array().square().matrix();
    }
  }

  auto ApplyPreconditioner = [&](const Context<double>& c,
                                const VectorX<double>& g_in) -> VectorX<double> {
    if (!use_preconditioner) return g_in;
    if (!use_diag_h_preconditioner) return inv_diag_A.array() * g_in.array();
    const int nk = model_->num_constraint_equations();
    VectorX<double> diagG(nk);
    int offset = 0;
    const std::vector<MatrixX<double>>& G = model_->EvalConstraintsHessian(c);
    for (const auto& Gi : G) {
      const int ni = Gi.rows();
      diagG.segment(offset, ni) = Gi.diagonal();
      offset += ni;
    }
    DRAKE_DEMAND(offset == nk);
    DRAKE_DEMAND(J_squared.rows() == nk);
    VectorX<double> diag_H = diag_A + J_squared.transpose() * diagG;
    diag_H = diag_H.array().max(parameters_.diag_h_min_diagonal).matrix();
    return g_in.array() / diag_H.array();
  };

  // Initialize objective and gradient.
  double ell = model_->EvalCost(*context);
  double ell_previous = ell;
  VectorX<double> g = model_->EvalCostGradient(*context);  // copy
  VectorX<double> z = ApplyPreconditioner(*context, g);
  VectorX<double> d = -z;

  double alpha = 1.0;
  bool converged = false;
  int k = 0;
  for (;; ++k) {
    // Stopping criteria check (before any expensive line search work).
    double momentum_residual{}, momentum_scale{};
    CalcStoppingCriteriaResidual(*context, &momentum_residual, &momentum_scale);
    stats_.optimality_criterion_reached =
        momentum_residual <= parameters_.abs_tolerance +
                                parameters_.rel_tolerance * momentum_scale;
    stats_.cost.push_back(ell);
    stats_.alpha.push_back(alpha);
    stats_.momentum_residual.push_back(momentum_residual);
    stats_.momentum_scale.push_back(momentum_scale);

    if (stats_.optimality_criterion_reached || stats_.cost_criterion_reached) {
      converged = true;
      break;
    }

    if (k == parameters_.max_iterations) break;

    // Sanity check: expected monotonic decrease; allow small slop.
    {
      const double ell_scale = 0.5 * (std::abs(ell) + std::abs(ell_previous));
      const double ell_slop =
          parameters_.relative_slop * std::max(1.0, ell_scale);
      if (ell > ell_previous + ell_slop &&
          parameters_.nonmonotonic_convergence_is_error) {
        throw std::runtime_error(
            "SapNlcgSolver: Non-monotonic convergence detected.");
      }
    }

    // Ensure descent direction.
    if (g.dot(d) >= 0.0) d = -z;
    const double dphi0 = g.dot(d);
    if (!(dphi0 < 0.0)) {
      // If we cannot find a descent direction, the best we can do is stop.
      break;
    }

    std::cout << "\033[34m"
          << fmt::format("k={}, dphi0={} ell={} alpha={}\n momentum_residual={}, momentum_scale={}, threshold={}", 
            k, dphi0, ell, alpha, 
            momentum_residual, momentum_scale, parameters_.abs_tolerance + parameters_.rel_tolerance * momentum_scale)
          << "\033[0m"
          << std::endl;

    // Line search.
    int ls_iters = 0;
    switch (parameters_.line_search_type) {
      case SapNlcgSolverParameters::LineSearchType::kStrongWolfe:
        std::tie(alpha, ls_iters) = PerformStrongWolfeLineSearch(
            *context, d, ell, dphi0, scratch.get());
        break;
      case SapNlcgSolverParameters::LineSearchType::kExact:
        std::tie(alpha, ls_iters) =
            PerformExactLineSearch(*context, d, scratch.get());
        break;
    }
    stats_.num_line_search_iters += ls_iters;

    if (!IsFinite(alpha) || alpha <= 0.0) break;
    if (parameters_.line_search_type ==
            SapNlcgSolverParameters::LineSearchType::kStrongWolfe &&
        alpha < parameters_.strong_wolfe.min_alpha) {
      // Step is numerically too small; treat as stall.
      stats_.cost_criterion_reached = true;
      continue;
    }

    // Take step.
    model_->GetMutableVelocities(context.get()) += alpha * d;

    // Update objective and gradient.
    ell_previous = ell;
    ell = model_->EvalCost(*context);
    const VectorX<double> g_prev = std::move(g);
    const VectorX<double> z_prev = std::move(z);
    const VectorX<double> d_prev = d;
    g = model_->EvalCostGradient(*context);  // copy
    z = ApplyPreconditioner(*context, g);

    // Cost-stall check (round-off detection).
    {
      const double ell_scale = 0.5 * (std::abs(ell) + std::abs(ell_previous));
      const double ell_decrement = std::abs(ell_previous - ell);
      stats_.cost_criterion_reached =
          ell_decrement < parameters_.cost_abs_tolerance +
                              parameters_.cost_rel_tolerance * ell_scale &&
          alpha > 0.5;
    }

    // PR+ update with restart heuristic.
    double beta = 0.0;
    if (use_preconditioner) {
      const VectorX<double> z_diff = z - z_prev;
      const double denom = g_prev.dot(z_prev);
      if (denom > 0.0) {
        beta = std::max(0.0, g.dot(z_diff) / denom);
      }
    } else {
      const VectorX<double> y = g - g_prev;
      const double denom = g_prev.squaredNorm();
      if (denom > 0.0) {
        beta = std::max(0.0, g.dot(y) / denom);
      }
    }
    {
      const double denom = g_prev.squaredNorm();
      if (denom > 0.0 &&
          (g.dot(g_prev) / denom) > parameters_.restart_threshold) {
        beta = 0.0;
      }
    }

    d = -z + beta * d_prev;
    if (g.dot(d) >= 0.0) d = -z;
  }

  if (!converged) return SapSolverStatus::kFailure;

  PackSapSolverResults(*context, results);
  stats_.num_iters = k;
  return SapSolverStatus::kSuccess;
}

template <>
std::pair<double, double> SapNlcgSolver<double>::EvalPhiAndDeriv(
    const Context<double>& context, const VectorX<double>& direction,
    double alpha, Context<double>* scratch) const {
  DRAKE_DEMAND(scratch != nullptr);
  DRAKE_DEMAND(scratch != &context);

  // v(α) = v + α d.
  auto v_alpha = model_->GetMutableVelocities(scratch);
  v_alpha = model_->GetVelocities(context);
  v_alpha.noalias() += alpha * direction;

  const double phi = model_->EvalCost(*scratch);
  const VectorX<double>& grad = model_->EvalCostGradient(*scratch);
  const double dphi = grad.dot(direction);
  return {phi, dphi};
}

template <>
std::pair<double, int> SapNlcgSolver<double>::PerformBacktrackingArmijo(
    const Context<double>& context, const VectorX<double>& direction,
    double phi0, double dphi0, Context<double>* scratch) const {
  // Simple Armijo backtracking fallback; parameters chosen for robustness.
  const double c1 = parameters_.strong_wolfe.c1;
  const double rho = 0.5;
  const int max_iters = std::max(1, parameters_.strong_wolfe.max_iterations);

  double alpha = std::min(parameters_.strong_wolfe.alpha0,
                          parameters_.strong_wolfe.alpha_max);
  int it = 0;
  for (; it < max_iters; ++it) {
    const auto [phi, dphi] = EvalPhiAndDeriv(context, direction, alpha, scratch);
    (void)dphi;
    if (phi <= phi0 + c1 * alpha * dphi0) return {alpha, it + 1};
    alpha *= rho;
    if (alpha < parameters_.strong_wolfe.min_alpha) break;
  }
  return {alpha, it};
}

template <>
double SapNlcgSolver<double>::StrongWolfeZoom(
    const Context<double>& context, const VectorX<double>& direction,
    double phi0, double dphi0, double alpha_lo, double alpha_hi,
    double phi_lo, double phi_hi, Context<double>* scratch, int* iters) const {
  DRAKE_DEMAND(iters != nullptr);
  (void)phi_hi;
  const double c1 = parameters_.strong_wolfe.c1;
  const double c2 = parameters_.strong_wolfe.c2;

  double a_lo = alpha_lo;
  double a_hi = alpha_hi;
  double f_lo = phi_lo;

  const int max_iters = std::max(1, parameters_.strong_wolfe.max_iterations);
  for (int j = 0; j < max_iters; ++j) {
    ++(*iters);

    // Bisection for robustness.
    const double a = 0.5 * (a_lo + a_hi);
    const auto [f, dphi] = EvalPhiAndDeriv(context, direction, a, scratch);

    if ((f > phi0 + c1 * a * dphi0) || (f >= f_lo)) {
      a_hi = a;
    } else {
      if (std::abs(dphi) <= -c2 * dphi0) return a;
      if (dphi * (a_hi - a_lo) >= 0.0) {
        a_hi = a_lo;
      }
      a_lo = a;
      f_lo = f;
    }

    if (std::abs(a_hi - a_lo) < parameters_.strong_wolfe.min_alpha) break;
  }
  return a_lo;
}

template <>
std::pair<double, int> SapNlcgSolver<double>::PerformStrongWolfeLineSearch(
    const Context<double>& context, const VectorX<double>& direction,
    double phi0, double dphi0, Context<double>* scratch) const {
  DRAKE_DEMAND(scratch != nullptr);
  DRAKE_DEMAND(scratch != &context);
  DRAKE_DEMAND(dphi0 < 0.0);

  const auto& p = parameters_.strong_wolfe;
  const double c1 = p.c1;
  const double c2 = p.c2;
  DRAKE_DEMAND(c1 > 0.0 && c2 > c1 && c2 < 1.0);

  double alpha_prev = 0.0;
  double phi_prev = phi0;

  double alpha = std::min(p.alpha0, p.alpha_max);
  int iters = 0;

  for (int i = 0; i < std::max(1, p.max_iterations); ++i) {
    ++iters;
    const auto [phi, dphi] = EvalPhiAndDeriv(context, direction, alpha, scratch);

    if ((phi > phi0 + c1 * alpha * dphi0) ||
        (i > 0 && phi >= phi_prev)) {
      const double a = StrongWolfeZoom(context, direction, phi0, dphi0,
                                      alpha_prev, alpha, phi_prev, phi,
                                      scratch, &iters);
      return {a, iters};
    }

    if (std::abs(dphi) <= -c2 * dphi0) {
      return {alpha, iters};
    }

    if (dphi >= 0.0) {
      const double a = StrongWolfeZoom(context, direction, phi0, dphi0,
                                      alpha, alpha_prev, phi, phi_prev,
                                      scratch, &iters);
      return {a, iters};
    }

    alpha_prev = alpha;
    phi_prev = phi;

    // Expand trial step.
    const double alpha_next = std::min(alpha * p.expansion, p.alpha_max);
    if (alpha_next <= alpha) break;
    alpha = alpha_next;
  }

  // Fallback to Armijo backtracking.
  auto [a, back_iters] =
      PerformBacktrackingArmijo(context, direction, phi0, dphi0, scratch);
  return {a, iters + back_iters};
}

template <>
std::pair<double, int> SapNlcgSolver<double>::PerformExactLineSearch(
    const Context<double>& context, const VectorX<double>& direction,
    Context<double>* scratch) const {
  DRAKE_DEMAND(parameters_.line_search_type ==
               SapNlcgSolverParameters::LineSearchType::kExact);
  DRAKE_DEMAND(scratch != nullptr);
  DRAKE_DEMAND(scratch != &context);

  // dℓ/dα(α = 0) = ∇ᵥℓ(α = 0)⋅Δv.
  const VectorX<double>& ell_grad_v0 = model_->EvalCostGradient(context);
  const VectorX<double>& dv = direction;
  const double dell_dalpha0 = ell_grad_v0.dot(dv);

  // dℓ/dα(α = 0) is guaranteed to be strictly negative given the Hessian of
  // the cost is positive definite. Only round-off errors for ill-conditioned
  // systems can destroy this property.
  if (dell_dalpha0 >= 0) {
    throw std::runtime_error(
        "The cost does not decrease along the search direction. This is "
        "usually caused by an excessive accumulation round-off errors for "
        "ill-conditioned systems. Consider revisiting your model.");
  }

  // Precompute search direction quantities at state v.
  VectorX<double> dp(model_->num_velocities());
  VectorX<double> dvc(model_->num_constraint_equations());
  model_->constraints_bundle().J().Multiply(dv, &dvc);
  model_->MultiplyByDynamicsMatrix(dv, &dp);
  NlcgSearchDirectionData search_direction_data{dv, dp, dvc, dv.dot(dp)};

  const double alpha_max = parameters_.exact_line_search.alpha_max;
  double dell{NAN};
  double d2ell{NAN};
  VectorX<double> vec_scratch;
  const double ell0 = CalcCostAlongLine(*model_, context, search_direction_data,
                                       alpha_max, scratch, &dell, &d2ell,
                                       &vec_scratch);

  // If the cost is still decreasing at alpha_max, we accept this value.
  if (dell <= 0) return std::make_pair(alpha_max, 0);

  // If the user requests very tight tolerances, we might enter the line search
  // with a very small gradient. Close to machine epsilon, Newton might return
  // inaccurate results. Therefore return early if we detect this situation.
  if (-dell_dalpha0 <
      parameters_.cost_abs_tolerance + parameters_.cost_rel_tolerance * ell0)
    return std::make_pair(1.0, 0);

  // N.B. We place the data needed to evaluate cost and gradients into a single
  // struct so that cost_and_gradient only needs to capture a single pointer.
  // This avoids heap allocations when passing the lambda to
  // DoNewtonWithBisectionFallback().
  struct EvalData {
    const SapNlcgSolver<double>& solver;
    const Context<double>& context0;  // Context at alpha = 0.
    const NlcgSearchDirectionData& search_direction_data;
    Context<double>& scratch;  // Context at alpha != 0.
    // N.B. We normalize the gradient to minimize round-off errors as f(alpha) =
    // −ℓ'(α)/dell_scale.
    const double dell_scale;
    VectorX<double> vec_scratch;
  };

  // N.B. At this point we know that dell_dalpha0 < 0. Also, if the line search
  // was called it is because the residual is non-zero. Therefore we can safely
  // divide by dell_dalpha0.
  const double dell_scale = -dell_dalpha0;
  EvalData data{*this, context, search_direction_data, *scratch, dell_scale};

  // Cost and gradient of f(α) = −ℓ'(α)/ℓ'₀.
  auto cost_and_gradient = [&data](double x) {
    double dell_dalpha;
    double d2ell_dalpha2;
    CalcCostAlongLine(*data.solver.model_, data.context0,
                      data.search_direction_data, x, &data.scratch,
                      &dell_dalpha, &d2ell_dalpha2, &data.vec_scratch);
    return std::make_pair(dell_dalpha / data.dell_scale,
                          d2ell_dalpha2 / data.dell_scale);
  };

  // To estimate a guess, we approximate the cost as being quadratic around
  // alpha = 0.
  const double alpha_guess = std::min(-dell_dalpha0 / d2ell, alpha_max);

  // N.B. If we are here, then we already know that dell_dalpha0 < 0 and dell >
  // 0, and therefore [0, alpha_max] is a valid bracket.
  const Bracket bracket(0., dell_dalpha0 / dell_scale, alpha_max,
                        dell / dell_scale);

  const double f_tolerance = 1.0e-8;  // f = −ℓ'(α)/ℓ'₀ is dimensionless.
  const double alpha_tolerance = f_tolerance * alpha_guess;
  const auto [alpha, iters] = DoNewtonWithBisectionFallback(
      cost_and_gradient, bracket, alpha_guess, alpha_tolerance, f_tolerance,
      parameters_.exact_line_search.max_iterations);

  return std::make_pair(alpha, iters);
}


}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::internal::SapNlcgSolver)
