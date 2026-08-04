// CountTestLambdaEtaAlphafixri.cpp
// Count data model with log-normal latent variables:
//   Given counts Y_count(i,t,j), introduce latent log-scale Y(i,t,j) such that
//     Y(i,t,j) | params ~ N( r_it + alpha_ijt + Lambda_j * eta_it, sigma2 )
//     Y_count(i,t,j) = y  implies Y(i,t,j) in [log(y), log(y+1)) (and y=0 implies (-inf, 0]).
// Sampling step: Y(i,t,j) is drawn from the corresponding truncated normal each MCMC iteration.
// Remaining updates (r with fixed u2_r = 1, alpha (AR1 + DP), eta (AR1), Lambda, sigma2, rho, phi, V) are
// the same as the Gaussian model but conditioning on the latent Y.

#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <float.h>  

using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// ── Embedded truncnorm (Bornkamp & Mersmann) ────────────────────────────────
static const double t1 = 0.15;
static const double t2 = 2.18;
static const double t3 = 0.725;
static const double t4 = 0.45;


static inline double ers_a_inf(double a) {
  const double ainv = 1.0 / a;
  double x, rho;
  do {
    x = R::rexp(ainv) + a;
    rho = exp(-0.5 * (x - a) * (x - a));
  } while (R::runif(0, 1) > rho);
  return x;
}

static inline double ers_a_b(double a, double b) {
  const double ainv = 1.0 / a;
  double x, rho;
  do {
    x = R::rexp(ainv) + a;
    rho = exp(-0.5 * (x - a) * (x - a));
  } while (R::runif(0, 1) > rho || x > b);
  return x;
}

static inline double nrs_a_b(double a, double b) {
  double x = -DBL_MAX;
  while (x < a || x > b) {
    x = R::rnorm(0, 1);
  }
  return x;
}

static inline double nrs_a_inf(double a) {
  double x = -DBL_MAX;
  while (x < a) {
    x = R::rnorm(0, 1);
  }
  return x;
}

static inline double hnrs_a_b(double a, double b) {
  double x = a - 1.0;
  while (x < a || x > b) {
    x = R::rnorm(0, 1);
    x = fabs(x);
  }
  return x;
}

static inline double urs_a_b(double a, double b) {
  const double phi_a = R::dnorm(a, 0.0, 1.0, false);
  double x, u;
  const double ub = (a < 0 && b > 0) ? M_1_SQRT_2PI : phi_a;
  do {
    x = R::runif(a, b);
  } while (R::runif(0, 1) * ub > R::dnorm(x, 0.0, 1.0, false));
  return x;
}

static inline double r_lefttruncnorm(double a, double mean, double sd) {
  const double alpha = (a - mean) / sd;
  if (alpha < t4) {
    return mean + sd * nrs_a_inf(alpha);
  } else {
    return mean + sd * ers_a_inf(alpha);
  }
}

static inline double r_righttruncnorm(double b, double mean, double sd) {
  const double beta = (b - mean) / sd;
  return mean - sd * r_lefttruncnorm(-beta, 0.0, 1.0);
}

static inline double rtruncnorm_cpp(double a, double b, double mean, double sd) {
  const double alpha = (a - mean) / sd;
  const double beta  = (b - mean) / sd;
  const double phi_a = R::dnorm(alpha, 0.0, 1.0, false);
  const double phi_b = R::dnorm(beta,  0.0, 1.0, false);
  
  if (beta <= alpha) {
    Rcpp::stop("r_truncnorm: a >= b after standardizing");
  } else if (alpha <= 0 && 0 <= beta) {
    if (phi_a <= t1 || phi_b <= t1) {
      return mean + sd * nrs_a_b(alpha, beta);
    } else {
      return mean + sd * urs_a_b(alpha, beta);
    }
  } else if (alpha > 0) {
    if (phi_a / phi_b <= t2) {
      return mean + sd * urs_a_b(alpha, beta);
    } else if (alpha < t3) {
      return mean + sd * hnrs_a_b(alpha, beta);
    } else {
      return mean + sd * ers_a_b(alpha, beta);
    }
  } else {
    if (phi_b / phi_a <= t2) {
      return mean - sd * urs_a_b(-beta, -alpha);
    } else if (beta > -t3) {
      return mean - sd * hnrs_a_b(-beta, -alpha);
    } else {
      return mean - sd * ers_a_b(-beta, -alpha);
    }
  }
}



// Safe MVN sampler via Cholesky; avoids mvnrnd() symmetry/PD issues
// caller: optional label to trace which call triggered chol warning (pass NULL to skip)
arma::vec safe_mvnrnd(const arma::vec& mu, arma::mat Sigma, double diag_jitter,
                      const char* caller = nullptr) {
  int n = mu.n_elem;
  // Build a clean symmetric matrix: average then force lower = upper so chol() never warns
  arma::mat S = (Sigma + Sigma.t()) * 0.5;
  for (arma::uword j = 0; j < S.n_cols; j++)
    for (arma::uword i = j + 1; i < S.n_rows; i++)
      S(i, j) = S(j, i);
  S.diag() += diag_jitter;
  arma::mat L;
  for (int attempt = 0; attempt < 25; attempt++) {
    if (arma::chol(L, S, "lower")) {
      arma::vec z(n);
      for (int i = 0; i < n; i++) z(i) = R::rnorm(0, 1);
      return mu + L * z;
    }
    (void)caller;
    S.diag() += 0.5;
  }
  arma::vec d = S.diag();
  for (arma::uword i = 0; i < d.n_elem; i++) if (d(i) <= 0) d(i) = 1e-10;
  return mu + arma::sqrt(d) % arma::randn(n);
}

Rcpp::NumericVector compute_covY_LL_part_cpp(const Rcpp::NumericVector& Lambda_samples,
                                             const Rcpp::NumericVector& rho_samples,
                                             int T) {
  // Lambda_samples has dim (nkeep, J, K) as returned from lodi_mcmc_cpp
  Rcpp::IntegerVector dims = Lambda_samples.attr("dim");
  if (dims.size() != 3)
    Rcpp::stop("Lambda_samples must be a 3D array with dim (nkeep, J, K).");

  int nkeep = dims[0];
  int J     = dims[1];
  int K     = dims[2];

  if (rho_samples.size() != nkeep)
    Rcpp::stop("rho_samples length must equal nkeep.");

  // Wrap output buffer as an Armadillo cube: (J, J, T)
  Rcpp::NumericVector Cov_LL_R(T * J * J);
  arma::cube Cov_LL(reinterpret_cast<double*>(Cov_LL_R.begin()),
                    J, J, T, false);
  Cov_LL.fill(0.0);

  // Temporary matrices for Lambda_s and Lambda_s Lambda_s'
  arma::mat L(J, K);
  arma::mat LLt(J, J);

  for (int s = 0; s < nkeep; ++s) {
    // Fill L with Lambda_s (J x K) from flattened (nkeep, J, K) array
    for (int j = 0; j < J; ++j) {
      for (int k = 0; k < K; ++k) {
        int idx = s + j * nkeep + k * nkeep * J; // (s, j, k)
        L(j, k) = Lambda_samples[idx];
      }
    }

    // Matrix multiply to get Lambda_s Lambda_s'
    LLt = L * L.t(); // (J x K) * (K x J) = (J x J)

    double rho_s = rho_samples[s];
    double denom = 1.0 - rho_s * rho_s;
    if (std::fabs(denom) < 1e-10)
      denom = (rho_s >= 0 ? 1e-10 : -1e-10);

    for (int tt = 0; tt < T; ++tt) {
      int t1 = tt + 1; // time index in formula is 1-based
      double pow_term = std::pow(rho_s, 2 * t1);
      double scale_s_t = (1.0 - pow_term) / denom;
      Cov_LL.slice(tt) += scale_s_t * LLt;
    }
  }

  // Divide by number of kept iterations to get expectation
  Cov_LL /= static_cast<double>(nkeep);

  // Attach dim attribute (J, J, T) to match Armadillo cube memory layout
  Cov_LL_R.attr("dim") = Rcpp::IntegerVector::create(J, J, T);
  return Cov_LL_R;
}

// FFBS for eta (time-varying observation variance sigma2_t)
arma::mat ffbs_eta_cpp(const arma::mat& y_subject,
                       const arma::vec& r_subject,
                       const arma::mat& alpha_subject,
                       const arma::mat& Lambda,
                       double rho,
                       double Q,
                       const arma::vec& sigma2_t) {
  int Time = y_subject.n_rows;
  int J = y_subject.n_cols;
  int K = Lambda.n_cols;
  arma::mat m_filt(Time, K, arma::fill::zeros);
  arma::cube C_filt(Time, K, K, arma::fill::zeros);
  arma::mat a_pred(Time, K, arma::fill::zeros);
  arma::cube R_pred(Time, K, K, arma::fill::zeros);
  arma::mat I_K = arma::eye(K, K);
  arma::mat Q_mat = Q * I_K;

  for (int t = 0; t < Time; t++) {
    if (t == 0) {
      for (int k = 0; k < K; k++) a_pred(t, k) = 0.0;
      for (int k1 = 0; k1 < K; k1++)
        for (int k2 = 0; k2 < K; k2++)
          R_pred(t, k1, k2) = I_K(k1, k2);
      for (int k = 0; k < K; k++) R_pred(t, k, k) += 1e-10;
    } else {
      for (int k = 0; k < K; k++) a_pred(t, k) = rho * m_filt(t-1, k);
      arma::mat C_prev(K, K);
      for (int k1 = 0; k1 < K; k1++)
        for (int k2 = 0; k2 < K; k2++)
          C_prev(k1, k2) = C_filt(t-1, k1, k2);
      arma::mat R_t = rho * rho * C_prev + Q_mat;
      R_t.diag() += 1e-10;
      for (int k1 = 0; k1 < K; k1++)
        for (int k2 = 0; k2 < K; k2++)
          R_pred(t, k1, k2) = R_t(k1, k2);
    }
    arma::mat R_t(K, K);
    for (int k1 = 0; k1 < K; k1++)
      for (int k2 = 0; k2 < K; k2++)
        R_t(k1, k2) = R_pred(t, k1, k2);
    R_t.diag() += 1e-10;
    arma::mat inv_R = arma::inv(R_t);

    // time-specific observation precision: (Lambda' Lambda) / sigma2_t
    arma::mat LtL_sig_inv = Lambda.t() * Lambda / sigma2_t(t);
    arma::mat sum_inv = inv_R + LtL_sig_inv;
    sum_inv.diag() += 1e-10;
    arma::mat C_t = arma::inv(sum_inv);
    for (int k1 = 0; k1 < K; k1++)
      for (int k2 = 0; k2 < K; k2++)
        C_filt(t, k1, k2) = C_t(k1, k2);
    arma::rowvec obs_resid = y_subject.row(t) - r_subject(t) - alpha_subject.row(t);
    arma::vec a_vec(K), resid_vec(J);
    for (int k = 0; k < K; k++) a_vec(k) = a_pred(t, k);
    for (int j = 0; j < J; j++) resid_vec(j) = obs_resid(j);
    arma::mat C_t_m(K, K);
    for (int k1 = 0; k1 < K; k1++)
      for (int k2 = 0; k2 < K; k2++)
        C_t_m(k1, k2) = C_filt(t, k1, k2);
    arma::vec m_vec = C_t_m * (inv_R * a_vec + Lambda.t() * resid_vec / sigma2_t(t));
    for (int k = 0; k < K; k++) m_filt(t, k) = m_vec(k);
  }

  arma::mat eta_sampled(Time, K, arma::fill::zeros);
  arma::vec m_final(K);
  for (int k = 0; k < K; k++) m_final(k) = m_filt(Time-1, k);
  arma::mat C_final(K, K);
  for (int k1 = 0; k1 < K; k1++)
    for (int k2 = 0; k2 < K; k2++)
      C_final(k1, k2) = C_filt(Time-1, k1, k2);
  C_final = (C_final + C_final.t()) * 0.5;
  C_final = arma::symmatu(C_final);
  arma::rowvec er = safe_mvnrnd(m_final, C_final, 1e-10, "eta_final").t();
  for (int k = 0; k < K; k++) eta_sampled(Time-1, k) = er(k);

  for (int t = Time-2; t >= 0; t--) {
    arma::mat R_next(K, K), C_t(K, K);
    for (int k1 = 0; k1 < K; k1++)
      for (int k2 = 0; k2 < K; k2++) {
        R_next(k1, k2) = R_pred(t+1, k1, k2);
        C_t(k1, k2) = C_filt(t, k1, k2);
      }
    R_next.diag() += 1e-10;
    arma::mat inv_R_next = arma::inv(R_next);
    arma::mat H_t = C_t * (rho * I_K) * inv_R_next;
    arma::vec eta_next(K), a_next(K), m_t(K);
    for (int k = 0; k < K; k++) {
      eta_next(k) = eta_sampled(t+1, k);
      a_next(k) = a_pred(t+1, k);
      m_t(k) = m_filt(t, k);
    }
    arma::vec m_cond = m_t + H_t * (eta_next - a_next);
    arma::mat R_next_b(K, K);
    for (int k1 = 0; k1 < K; k1++)
      for (int k2 = 0; k2 < K; k2++)
        R_next_b(k1, k2) = R_pred(t+1, k1, k2);
    arma::mat C_cond = C_t - H_t * R_next_b * H_t.t();
    C_cond = (C_cond + C_cond.t()) * 0.5;
    C_cond = arma::symmatu(C_cond);
    er = safe_mvnrnd(m_cond, C_cond, 1e-10, "eta_cond").t();
    for (int k = 0; k < K; k++) eta_sampled(t, k) = er(k);
  }
  return eta_sampled;
}

// FFBS for alpha with DP cluster-specific prior on alpha_ij1
arma::vec ffbs_alpha_dp_cpp(int alpha_g_idx,
                            const arma::vec& alpha_mu,
                            const arma::vec& alpha_sig2,
                            double phi_i,
                            double V,
                            const arma::vec& Y_vec,
                            const arma::vec& r_vec,
                            const arma::vec& Lambda_j,
                            const arma::mat& eta_mat,
                            const arma::vec& sigma2_t) {
  int Time = Y_vec.n_elem;
  arma::vec m(Time, arma::fill::zeros);
  arma::vec C(Time, arma::fill::zeros);
  arma::vec a(Time, arma::fill::zeros);
  arma::vec R(Time, arma::fill::zeros);

  for (int t = 0; t < Time; t++) {
    if (t == 0) {
      a(t) = alpha_mu(alpha_g_idx - 1);
      R(t) = alpha_sig2(alpha_g_idx - 1);
    } else {
      a(t) = phi_i * m(t-1);
      R(t) = phi_i * phi_i * C(t-1) + V;
    }
    double factor_effect = arma::dot(Lambda_j, eta_mat.row(t).t());
    double obs_resid = Y_vec(t) - r_vec(t) - factor_effect;
    double K_gain = R(t) / (R(t) + sigma2_t(t));
    m(t) = a(t) + K_gain * (obs_resid - a(t));
    C(t) = R(t) - K_gain * R(t);
  }

  arma::vec alpha_st(Time, arma::fill::zeros);
  alpha_st(Time-1) = R::rnorm(m(Time-1), std::sqrt(std::max(C(Time-1), 1e-10)));
  for (int t = Time-2; t >= 0; t--) {
    double H_t = (C(t) * phi_i) / (R(t+1) + 1e-12);
    double m_smooth = m(t) + H_t * (alpha_st(t+1) - a(t+1));
    double C_smooth = C(t) - H_t * H_t * R(t+1);
    if (C_smooth < 1e-10) C_smooth = 1e-10;
    alpha_st(t) = R::rnorm(m_smooth, std::sqrt(C_smooth));
  }
  return alpha_st;
}

// [[Rcpp::export]]
Rcpp::List lodi_mcmc_cpp(Rcpp::NumericVector Y_vec,
                                     int seed,
                                     int niter,
                                     int nburn,
                                     int thin,
                                     int K,
                                     int L,
                                     Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
                                     Rcpp::Nullable<Rcpp::NumericMatrix> r_fixed = R_NilValue,
                                     Rcpp::Nullable<Rcpp::NumericMatrix> Lambda_fixed = R_NilValue,
                                     Rcpp::Nullable<Rcpp::NumericVector> eta_fixed = R_NilValue,
                                     Rcpp::Nullable<Rcpp::NumericVector> sigma2_t_fixed = R_NilValue,
                                     Rcpp::Nullable<Rcpp::NumericVector> rho_fixed = R_NilValue,
                                     Rcpp::Nullable<Rcpp::NumericVector> alpha_fixed = R_NilValue,
                                     Rcpp::Nullable<Rcpp::NumericVector> phi_g_fixed = R_NilValue,
                                     Rcpp::Nullable<Rcpp::NumericVector> V_fixed = R_NilValue) {
  Rcpp::Environment base_env("package:base");
  Rcpp::Function set_seed_r = base_env["set.seed"];
  set_seed_r(seed);

  Rcpp::IntegerVector dims = Y_vec.attr("dim");
  int N = dims[0], T = dims[1], J = dims[2];
  arma::cube Y_count(N, T, J);
  arma::cube Y(N, T, J);
  for (int i = 0; i < N; i++)
    for (int t = 0; t < T; t++)
      for (int j = 0; j < J; j++)
        Y_count(i, t, j) = Y_vec[i + t*N + j*N*T];

  // Prior mean for r_it: mu_it = 0 for all i,t
  arma::mat mu_r(N, T, arma::fill::zeros);

  // Genotype index x_i in {1, ..., G}; used for genotype-specific phi_g
  arma::ivec x_id(N, arma::fill::ones);
  int G = 1;
  if (X.isNotNull()) {
    Rcpp::NumericMatrix Xmat(X.get());
    if (Xmat.nrow() != N || Xmat.ncol() < 1)
      Rcpp::stop("X must be an N x 1 matrix (or matrix with first column as genotype).");
    for (int i = 0; i < N; i++) {
      int gi = static_cast<int>(std::round(Xmat(i, 0)));
      if (gi < 1) Rcpp::stop("Genotype labels in X must be >= 1.");
      x_id(i) = gi;
      if (gi > G) G = gi;
    }
  }

  std::vector<int> keep_vec;
  for (int it = nburn + 1; it <= niter; it += thin) keep_vec.push_back(it);
  int nkeep = keep_vec.size();

  double a0 = 2.0, b0 = 1.0;
  double rho_prior_mean = 0.0, rho_prior_var = 1.0;
  double phi_prior_mean = 0.0, phi_prior_var = 1.0;
  double V_prior_shape = 2.0, V_prior_rate = 1.0;
  double c_prior = 1.0;
  double alpha_prior_mean = 0.0;
  double alpha_prior_var = 25.0;
  double alpha_a0 = 2.0;
  double alpha_b0 = 1.0;

  bool fix_Lambda  = Lambda_fixed.isNotNull();
  bool fix_eta     = eta_fixed.isNotNull();
  bool fix_sigma2  = sigma2_t_fixed.isNotNull();
  bool fix_rho     = rho_fixed.isNotNull();
  bool fix_alpha   = alpha_fixed.isNotNull();
  bool fix_phi_g   = phi_g_fixed.isNotNull();
  bool fix_V       = V_fixed.isNotNull();

  arma::mat Lambda(J, K, arma::fill::zeros);
  if (fix_Lambda) {
    Rcpp::NumericMatrix Lf(Lambda_fixed.get());
    if (Lf.nrow() != J || Lf.ncol() != K)
      Rcpp::stop("Lambda_fixed must be a J x K matrix.");
    for (int j = 0; j < J; j++)
      for (int k = 0; k < K; k++)
        Lambda(j, k) = Lf(j, k);
  } else {
    for (int j = 0; j < J; j++) if (j < K) Lambda(j, j) = 1.0;
  }

  // time-specific residual variances sigma2_t
  arma::vec sigma2(T);
  if (fix_sigma2) {
    Rcpp::NumericVector s2 = sigma2_t_fixed.get();
    if ((int)s2.size() != T)
      Rcpp::stop("sigma2_t_fixed must be a vector of length T.");
    for (int t = 0; t < T; t++) {
      sigma2(t) = std::max((double)s2[t], 1e-8);
    }
  } else {
    sigma2.fill(1.0);
  }

  arma::cube eta(T, K, N);
  if (fix_eta) {
    Rcpp::NumericVector eflat = eta_fixed.get();
    Rcpp::IntegerVector edims = eflat.attr("dim");
    if (edims.size() != 3)
      Rcpp::stop("eta_fixed must be an array with dim (N, T, K).");
    int eN = edims[0], eT = edims[1], eK = edims[2];
    if (eN != N || eT != T || eK != K)
      Rcpp::stop("eta_fixed must have dim (N, T, K) consistent with Y.");
    for (int i = 0; i < N; i++)
      for (int t = 0; t < T; t++)
        for (int k = 0; k < K; k++) {
          int idx = i + t * N + k * N * T;
          eta(t, k, i) = eflat[idx];
        }
  } else {
    for (int i = 0; i < N; i++)
      for (int t = 0; t < T; t++)
        for (int k = 0; k < K; k++)
          eta(t, k, i) = R::rnorm(0, 1);
  }
  arma::cube alpha(N, T, J, arma::fill::zeros);
  if (fix_alpha) {
    Rcpp::NumericVector aflat(alpha_fixed.get());
    Rcpp::IntegerVector adims = aflat.attr("dim");
    if (adims.size() != 3)
      Rcpp::stop("alpha_fixed must be an array with dim (N, T, J).");
    int aN = adims[0], aT = adims[1], aJ = adims[2];
    if (aN != N || aT != T || aJ != J)
      Rcpp::stop("alpha_fixed must have dim (N, T, J) consistent with Y.");
    for (int i = 0; i < N; i++)
      for (int t = 0; t < T; t++)
        for (int j = 0; j < J; j++) {
          int idx = i + t * N + j * N * T;
          alpha(i, t, j) = aflat[idx];
        }
  }
  arma::imat alpha_g(N, J);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < J; j++)
      alpha_g(i, j) = R::rbinom(2, 0.5) + 1; // 1..3
  double rho = 0.5;
  if (fix_rho) {
    Rcpp::NumericVector rv = rho_fixed.get();
    if (rv.size() < 1) Rcpp::stop("rho_fixed must have at least one element.");
    rho = rv[0];
    if (rho <= -0.999999) rho = -0.999999;
    if (rho >=  0.999999) rho =  0.999999;
  }
  double Q = 1.0; // fixed
  arma::vec phi_g(G, arma::fill::ones);
  phi_g *= 0.5;
  if (fix_phi_g) {
    Rcpp::NumericVector pv(phi_g_fixed.get());
    if ((int)pv.size() != G)
      Rcpp::stop("phi_g_fixed must be a vector of length G (max genotype label in X).");
    for (int g = 0; g < G; g++) {
      double val = pv[g];
      if (val <= -0.999999) val = -0.999999;
      if (val >=  0.999999) val =  0.999999;
      phi_g(g) = val;
    }
  }

  double V = 1.0;
  if (fix_V) {
    Rcpp::NumericVector vv(V_fixed.get());
    if (vv.size() < 1) Rcpp::stop("V_fixed must have at least one element.");
    V = std::max((double)vv[0], 1e-10);
  }

  arma::vec weight(L);
  double sum_gamma = 0.0;
  for (int l = 0; l < L; l++) {
    weight(l) = R::rgamma(1.0, 1.0);
    sum_gamma += weight(l);
  }
  weight = weight / sum_gamma;
  arma::vec v(L-1, arma::fill::ones);
  v = v * 0.5;
  arma::vec alpha_mu(L, arma::fill::zeros);
  arma::vec alpha_sig2(L, arma::fill::ones);

  Rcpp::NumericVector Lambda_samples_vec(nkeep * J * K);
  Rcpp::NumericVector eta_samples_vec(nkeep * N * T * K);
  Rcpp::NumericVector alpha_samples_vec(nkeep * N * T * J);
  Rcpp::NumericVector r_samples_vec(nkeep * N * T);
  Rcpp::NumericVector Y_latent_samples_vec(nkeep * N * T * J);
  Rcpp::NumericVector sigma2_t_samples_vec(nkeep * T);    // time-specific sigma2_t draws
  Rcpp::NumericVector rho_samples_vec(nkeep);
  Rcpp::NumericVector phi_g_samples_vec(nkeep * G);
  Rcpp::NumericVector phi_samples_vec(nkeep);
  Rcpp::NumericVector V_samples_vec(nkeep);

  auto in_keep = [&](int iter) -> bool {
    for (int k = 0; k < nkeep; k++) if (keep_vec[k] == iter + 1) return true;
    return false;
  };
  auto get_keep_idx = [&](int iter) -> int {
    for (int k = 0; k < nkeep; k++) if (keep_vec[k] == iter + 1) return k;
    return -1;
  };

  bool fix_r = r_fixed.isNotNull();
  arma::mat r(N, T);
  const double u2_r = 1.0;
  if (fix_r) {
    Rcpp::NumericMatrix Rr(r_fixed.get());
    for (int i = 0; i < N; i++)
      for (int t = 0; t < T; t++)
        r(i, t) = Rr(i, t);
  } else {
    for (int i = 0; i < N; i++)
      for (int t = 0; t < T; t++)
        r(i, t) = R::rnorm(mu_r(i, t), std::sqrt(u2_r));
  }

  // Initialize latent Y on the log scale from the truncation region implied by counts.
  // Using current (initial) parameters as the mean for the truncated normal draw.
  for (int i = 0; i < N; i++) {
    for (int t = 0; t < T; t++) {
      for (int j = 0; j < J; j++) {
        double eta_dot = 0.0;
        for (int k = 0; k < K; k++) eta_dot += Lambda(j, k) * eta(t, k, i);
        double mu_ijt = r(i, t) + alpha(i, t, j) + eta_dot;
        double lower_bound, upper_bound;
        if (Y_count(i, t, j) <= 0.0) {
          lower_bound = -1e10;
          upper_bound = 0.0;
        } else {
          lower_bound = std::log(Y_count(i, t, j));
          upper_bound = std::log(Y_count(i, t, j) + 1.0);
        }
        Y(i, t, j) = rtruncnorm_cpp(lower_bound, upper_bound, mu_ijt, std::sqrt(sigma2(t)));
      }
    }
  }

  for (int iter = 0; iter < niter; iter++) {
    // 0) Sample latent log-normal Y given counts and current parameters
    arma::cube mu(N, T, J);
    for (int t = 0; t < T; t++) {
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < J; j++) {
          arma::rowvec eta_row_t(K);
          for (int k = 0; k < K; k++) eta_row_t(k) = eta(t, k, i);
          double factor_effect = arma::dot(Lambda.row(j), eta_row_t.t());
          mu(i, t, j) = r(i, t) + alpha(i, t, j) + factor_effect;
        }
      }
    }
    for (int i = 0; i < N; i++) {
      for (int t = 0; t < T; t++) {
        for (int j = 0; j < J; j++) {
          double lower_bound, upper_bound;
          if (Y_count(i, t, j) <= 0.0) {
            lower_bound = -1e10;
            upper_bound = 0.0;
          } else {
            lower_bound = std::log(Y_count(i, t, j));
            upper_bound = std::log(Y_count(i, t, j) + 1.0);
          }
          Y(i, t, j) = rtruncnorm_cpp(lower_bound, upper_bound, mu(i, t, j), std::sqrt(sigma2(t)));
        }
      }
    }

    // 1) Update r_it
    // Prior r_it ~ N(mu_it, u2_r), likelihood Y_ijt - r_it - alpha - Lambda*eta ~ N(0, sigma2)
    // (When this runs first in the sweep, it conditions on previous-iteration alpha/Lambda/eta.)
    if (!fix_r) {
      for (int i = 0; i < N; i++) {
        for (int t = 0; t < T; t++) {
          double r_res_sum = 0.0;
          for (int j = 0; j < J; j++) {
            double fe = 0.0;
            for (int k = 0; k < K; k++) fe += Lambda(j, k) * eta(t, k, i);
            r_res_sum += Y(i, t, j) - alpha(i, t, j) - fe;
          }
          double prec_post = 1.0 / u2_r + J / sigma2(t);
          double var_post = 1.0 / prec_post;
          double mean_post = (mu_r(i, t) / u2_r + r_res_sum / sigma2(t)) / prec_post;
          r(i, t) = R::rnorm(mean_post, std::sqrt(var_post));
        }
      }
    }

    // 2) Alpha-related updates (DP assignments/atoms + alpha FFBS + phi/V)
    if (!fix_alpha) {
      // 2a) Update alpha_g (group assignments for DP prior on alpha_ij1)
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < J; j++) {
          arma::vec log_probs(L);
          for (int l = 0; l < L; l++) {
            log_probs(l) = std::log(weight(l)) +
              R::dnorm(alpha(i, 0, j), alpha_mu(l), std::sqrt(alpha_sig2(l)), 1);
          }
          double max_log = log_probs.max();
          arma::vec log_probs_shifted = log_probs - max_log;
          arma::vec probs = arma::exp(log_probs_shifted) / arma::sum(arma::exp(log_probs_shifted));
          double u = R::runif(0, 1);
          double cumsum = 0.0;
          int sampled_l = L - 1;
          for (int l = 0; l < L; l++) {
            cumsum += probs(l);
            if (u <= cumsum) { sampled_l = l; break; }
          }
          alpha_g(i, j) = sampled_l + 1;
        }
      }

      // 2b) Update stick-breaking weights
      for (int l = 0; l < L-1; l++) {
        int n_l = 0, m_l = 0;
        for (int i = 0; i < N; i++) {
          for (int j = 0; j < J; j++) {
            if (alpha_g(i, j) == l + 1) n_l++;
            if (alpha_g(i, j) >  l + 1) m_l++;
          }
        }
        v(l) = R::rbeta(1.0 + n_l, c_prior + m_l);
      }
      weight(0) = v(0);
      double cum_stick = 1.0 - v(0);
      for (int l = 1; l < L-1; l++) {
        weight(l) = v(l) * cum_stick;
        cum_stick *= (1.0 - v(l));
      }
      weight(L-1) = cum_stick;

      // 2c) Update cluster parameters alpha_mu, alpha_sig2 using alpha_ij1
      arma::mat alpha_init(N, J);
      for (int i = 0; i < N; i++)
        for (int j = 0; j < J; j++)
          alpha_init(i, j) = alpha(i, 0, j);

      for (int l = 0; l < L; l++) {
        std::vector<double> y_l;
        for (int i = 0; i < N; i++) {
          for (int j = 0; j < J; j++) {
            if (alpha_g(i, j) == l + 1) y_l.push_back(alpha_init(i, j));
          }
        }
        int n_l = y_l.size();
        if (n_l > 0) {
          double sum_y = 0.0;
          for (double val : y_l) sum_y += val;
          double prec_mu_post = 1.0/alpha_prior_var + n_l/alpha_sig2(l);
          double mean_mu_post = (alpha_prior_mean/alpha_prior_var + sum_y/alpha_sig2(l)) / prec_mu_post;
          alpha_mu(l) = R::rnorm(mean_mu_post, std::sqrt(1.0/prec_mu_post));
          double sum_sq = 0.0;
          for (double val : y_l)
            sum_sq += (val - alpha_mu(l)) * (val - alpha_mu(l));
          double shape_sig_post = alpha_a0 + n_l/2.0;
          double rate_sig_post  = alpha_b0 + sum_sq/2.0;
          alpha_sig2(l) = 1.0 / R::rgamma(shape_sig_post, 1.0/rate_sig_post);
        } else {
          alpha_mu(l) = R::rnorm(alpha_prior_mean, std::sqrt(alpha_prior_var));
          alpha_sig2(l) = 1.0 / R::rgamma(alpha_a0, 1.0/alpha_b0);
        }
      }

      // 2d) Update alpha via FFBS with DP prior for each (i,j)
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < J; j++) {
          arma::vec Y_vec(T);
          arma::vec r_vec(T);
          arma::vec Lambda_j(K);
          for (int t = 0; t < T; t++) {
            Y_vec(t) = Y(i, t, j);
            r_vec(t) = r(i, t);
          }
          for (int k = 0; k < K; k++) Lambda_j(k) = Lambda(j, k);
          arma::mat eta_mat(T, K);
          for (int t = 0; t < T; t++)
            for (int k = 0; k < K; k++)
              eta_mat(t, k) = eta(t, k, i);

          double phi_i = phi_g(x_id(i) - 1);
          arma::vec alpha_ij = ffbs_alpha_dp_cpp(alpha_g(i, j), alpha_mu, alpha_sig2,
                                                 phi_i, V, Y_vec, r_vec, Lambda_j,
                                                 eta_mat, sigma2);
          for (int t = 0; t < T; t++) alpha(i, t, j) = alpha_ij(t);
        }
      }

      // 2e) Update phi_g (genotype-specific AR1 coefficient for alpha)
      if (!fix_phi_g) {
        for (int g = 1; g <= G; g++) {
          double SS_prev_a = 0.0, SS_pcur_a = 0.0;
          for (int i = 0; i < N; i++) {
            if (x_id(i) != g) continue;
            for (int j = 0; j < J; j++)
              for (int t = 0; t < T-1; t++) {
                double a_prev = alpha(i, t, j);
                double a_curr = alpha(i, t+1, j);
                SS_prev_a += a_prev * a_prev;
                SS_pcur_a += a_prev * a_curr;
              }
          }
          double phi_var = 1.0 / ((1.0 / phi_prior_var) + (SS_prev_a / V));
          double phi_mean = phi_var * ((phi_prior_mean / phi_prior_var) + (SS_pcur_a / V));
          phi_g(g - 1) = rtruncnorm_cpp(-1.0, 1.0, phi_mean, std::sqrt(phi_var));
        }
      }

      // 2f) Update V (innovation variance for alpha)
      if (!fix_V) {
        double V_res_sum = 0.0;
        for (int i = 0; i < N; i++)
          for (int j = 0; j < J; j++)
            for (int t = 0; t < T-1; t++) {
              double phi_i = phi_g(x_id(i) - 1);
              double resid = alpha(i, t+1, j) - phi_i * alpha(i, t, j);
              V_res_sum += resid * resid;
            }
        double V_shape = V_prior_shape + N * (T-1) * J / 2.0;
        double V_rate = V_prior_rate + V_res_sum / 2.0;
        V = 1.0 / R::rgamma(V_shape, 1.0 / V_rate);
      }
    }

    if (!fix_Lambda) {
      // 3) Lambda-related updates (time-specific sigma2_t via row-wise weighting)
      for (int j = 0; j < J; j++) {
        
        int k_limit = std::min(j + 1, K);
        
        // ----- 1. Build weighted regression -----
        arma::mat E_j(N * T, k_limit);
        arma::vec y_j(N * T);
        
        int idx = 0;
        for (int i = 0; i < N; i++) {
          for (int t = 0; t < T; t++) {
            double w_t = 1.0 / std::sqrt(sigma2(t));
            
            for (int kk = 0; kk < k_limit; kk++) {
              E_j(idx, kk) = eta(t, kk, i) * w_t;
            }
            
            double resid = Y(i, t, j) - alpha(i, t, j) - r(i, t);
            y_j(idx) = resid * w_t;
            idx++;
          }
        }
        
        // ----- 2. Posterior moments -----
        double Lambda_prior_var = 1.0;
        
        arma::mat V_j_inv =
          E_j.t() * E_j +
          (1.0 / Lambda_prior_var) * arma::eye(k_limit, k_limit);
        
        V_j_inv.diag() += 1e-6;
        V_j_inv = arma::symmatu(V_j_inv);
        
        arma::mat V_j;
        if (!arma::inv_sympd(V_j, V_j_inv)) {
          V_j = arma::inv(V_j_inv);
        }
        V_j = arma::symmatu(V_j);
        
        arma::vec m_j = V_j * (E_j.t() * y_j);
        
        arma::vec lambda_row(k_limit);
        
        // =====================================================
        // ===== 3. SAMPLING
        // =====================================================
        
        if (j < K) {
          
          int diag_idx = k_limit - 1;
          bool accepted = false;
          
          // ==========================================
          // ===== Path A: Rejection sampling (fast)
          // ==========================================
          for (int attempt = 0; attempt < 1000; attempt++) {
            
            arma::vec candidate =
              safe_mvnrnd(m_j, V_j, 1e-10, "Lambda_Vj_reject");
            
            if (candidate(diag_idx) > 0) {
              lambda_row = candidate;
              accepted = true;
              break;
            }
          }
          
          // ==========================================
          // ===== Path B: Exact Gibbs fallback
          // ==========================================
          if (!accepted) {
            
            // ----- STEP 1: sample diagonal (marginal truncated) -----
            double mu_jj  = m_j(diag_idx);
            double var_jj = V_j(diag_idx, diag_idx);
            
            lambda_row(diag_idx) =
              r_lefttruncnorm(0.0, mu_jj, std::sqrt(var_jj));
            
            // ----- STEP 2: conditional for off-diagonals -----
            if (diag_idx > 0) {
              
              arma::vec m1 =
                m_j.subvec(0, diag_idx - 1);
              
              arma::vec V12 =
                V_j(arma::span(0, diag_idx - 1), diag_idx);
              
              arma::mat V11 =
                V_j(arma::span(0, diag_idx - 1),
                    arma::span(0, diag_idx - 1));
              
              arma::vec m_cond =
                m1 + V12 * (lambda_row(diag_idx) - mu_jj) / var_jj;
              
              arma::mat V_cond =
                V11 - (V12 * V12.t()) / var_jj;
              
              V_cond = arma::symmatu(V_cond);
              
              lambda_row.subvec(0, diag_idx - 1) =
                safe_mvnrnd(m_cond, V_cond, 1e-10, "Lambda_Vcond");
            }
          }
          
        } else {
          // ----- unconstrained row -----
          lambda_row =
            safe_mvnrnd(m_j, V_j, 1e-10, "Lambda_Vj_full");
        }
        
        // ----- 4. Store -----
        Lambda.row(j).head(k_limit) = lambda_row.t();
        
        if (k_limit < K) {
          Lambda.row(j).tail(K - k_limit).zeros();
        }
      }
    }

    if (!fix_eta) {
      // 4) Update eta via FFBS
      for (int i = 0; i < N; i++) {
        arma::mat y_subject(T, J);
        arma::mat alpha_subject(T, J);
        arma::vec r_subject(T);
        for (int t = 0; t < T; t++) {
          r_subject(t) = r(i, t);
          for (int j = 0; j < J; j++) {
            y_subject(t, j) = Y(i, t, j);
            alpha_subject(t, j) = alpha(i, t, j);
          }
        }
        arma::mat eta_i = ffbs_eta_cpp(y_subject, r_subject, alpha_subject, Lambda, rho, Q, sigma2);
        for (int t = 0; t < T; t++)
          for (int k = 0; k < K; k++)
            eta(t, k, i) = eta_i(t, k);
      }

      // Center eta so (1/N)*sum_i eta_itk = 0 for each (t,k)
      for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
          arma::vec v(N);
          for (int i = 0; i < N; i++) v(i) = eta(t, k, i);
          double m = arma::mean(v);
          for (int i = 0; i < N; i++) eta(t, k, i) -= m;
        }
      }
    }

    if (!fix_rho) {
      // 5) Update rho (AR1 coefficient for eta), Q fixed at 1
      double SS_xx = 0.0, SS_xy = 0.0;
      for (int i = 0; i < N; i++)
        for (int t = 0; t < T-1; t++)
          for (int k = 0; k < K; k++) {
            double ep = eta(t, k, i), ec = eta(t+1, k, i);
            SS_xx += ep * ep;
            SS_xy += ep * ec;
          }
      double rho_mean = (SS_xy/Q + rho_prior_mean/rho_prior_var) / (SS_xx/Q + 1.0/rho_prior_var);
      double rho_sd = std::sqrt(1.0 / (SS_xx/Q + 1.0/rho_prior_var));
      rho = rtruncnorm_cpp(-1.0, 1.0, rho_mean, rho_sd);
    }

    if (!fix_sigma2) {
      // 6) Update sigma2_t independently for each time t
      for (int t = 0; t < T; t++) {
        double ssr_t = 0.0;
        for (int i = 0; i < N; i++)
          for (int j = 0; j < J; j++) {
            double fe = 0.0;
            for (int k = 0; k < K; k++) fe += Lambda(j, k) * eta(t, k, i);
            double resid = Y(i, t, j) - r(i, t) - alpha(i, t, j) - fe;
            ssr_t += resid * resid;
          }
        double shape_t = a0 + (N * J) / 2.0;
        double rate_t  = b0 + ssr_t / 2.0;
        sigma2(t) = 1.0 / R::rgamma(shape_t, 1.0 / rate_t);
      }
    }

    // Store samples
    if (in_keep(iter)) {
      int s = get_keep_idx(iter);
      for (int j = 0; j < J; j++)
        for (int k = 0; k < K; k++)
          Lambda_samples_vec[s + j*nkeep + k*nkeep*J] = Lambda(j, k);
      for (int i = 0; i < N; i++)
        for (int t = 0; t < T; t++)
          for (int k = 0; k < K; k++)
            eta_samples_vec[s + i*nkeep + t*nkeep*N + k*nkeep*N*T] = eta(t, k, i);
      for (int i = 0; i < N; i++)
        for (int t = 0; t < T; t++)
          for (int j = 0; j < J; j++)
            alpha_samples_vec[s + i*nkeep + t*nkeep*N + j*nkeep*N*T] = alpha(i, t, j);
      for (int i = 0; i < N; i++)
        for (int t = 0; t < T; t++)
          r_samples_vec[s + i*nkeep + t*nkeep*N] = r(i, t);
      for (int i = 0; i < N; i++)
        for (int t = 0; t < T; t++)
          for (int j = 0; j < J; j++)
            Y_latent_samples_vec[s + i*nkeep + t*nkeep*N + j*nkeep*N*T] = Y(i, t, j);
      // store full time-specific sigma2_t draws
      for (int t = 0; t < T; t++)
        sigma2_t_samples_vec[s + t * nkeep] = sigma2(t);
      rho_samples_vec[s] = rho;
      for (int g = 0; g < G; g++) {
        phi_g_samples_vec[s + g * nkeep] = phi_g(g);
      }
      phi_samples_vec[s] = arma::mean(phi_g);
      V_samples_vec[s] = V;
    }

    if ((iter + 1) % 1000 == 0) {
      double pct = 100.0 * (iter + 1) / niter;
      Rcpp::Rcout << "LambdaEtaAlphaRi Seed " << seed << " Iter " << iter + 1
                  << "/" << niter << " (" << pct << "%)" << std::endl;
      Rcpp::Rcout.flush();
    }
  }

  // Wrap samples and means
  Rcpp::IntegerVector Lambda_dim = Rcpp::IntegerVector::create(nkeep, J, K);
  Rcpp::IntegerVector eta_dim = Rcpp::IntegerVector::create(nkeep, N, T, K);
  Rcpp::IntegerVector alpha_dim = Rcpp::IntegerVector::create(nkeep, N, T, J);
  Rcpp::IntegerVector r_dim = Rcpp::IntegerVector::create(nkeep, N, T);
  Rcpp::IntegerVector Y_latent_dim = Rcpp::IntegerVector::create(nkeep, N, T, J);
  Rcpp::IntegerVector sigma2_t_dim = Rcpp::IntegerVector::create(nkeep, T);
  Rcpp::IntegerVector phi_g_dim = Rcpp::IntegerVector::create(nkeep, G);
  Rcpp::NumericVector Lambda_samples_array = Rcpp::NumericVector(Lambda_samples_vec);
  Lambda_samples_array.attr("dim") = Lambda_dim;
  Rcpp::NumericVector eta_samples_array = Rcpp::NumericVector(eta_samples_vec);
  eta_samples_array.attr("dim") = eta_dim;
  Rcpp::NumericVector alpha_samples_array = Rcpp::NumericVector(alpha_samples_vec);
  alpha_samples_array.attr("dim") = alpha_dim;
  Rcpp::NumericVector r_samples_array = Rcpp::NumericVector(r_samples_vec);
  r_samples_array.attr("dim") = r_dim;
  Rcpp::NumericVector Y_latent_samples_array = Rcpp::NumericVector(Y_latent_samples_vec);
  Y_latent_samples_array.attr("dim") = Y_latent_dim;
  Rcpp::NumericVector sigma2_t_samples_array = Rcpp::NumericVector(sigma2_t_samples_vec);
  sigma2_t_samples_array.attr("dim") = sigma2_t_dim;
  Rcpp::NumericVector phi_g_samples_array = Rcpp::NumericVector(phi_g_samples_vec);
  phi_g_samples_array.attr("dim") = phi_g_dim;

  return Rcpp::List::create(
    Rcpp::Named("Lambda_samples") = Lambda_samples_array,
    Rcpp::Named("eta_samples") = eta_samples_array,
    Rcpp::Named("alpha_samples") = alpha_samples_array,
    Rcpp::Named("r_samples") = r_samples_array,
    Rcpp::Named("Y_latent_samples") = Y_latent_samples_array,
    Rcpp::Named("sigma2_t_samples") = sigma2_t_samples_array,
    Rcpp::Named("rho_samples") = rho_samples_vec,
    Rcpp::Named("phi_g_samples") = phi_g_samples_array,
    Rcpp::Named("phi_samples") = phi_samples_vec,
    Rcpp::Named("V_samples") = V_samples_vec
  );
}

