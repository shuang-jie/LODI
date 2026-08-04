#' Simulate a longitudinal count table from the LODI generative model
#'
#' Draws a synthetic dataset matching the "Simulation 1: matched truth"
#' setting of Zhang, Ok, and Ni (submitted): rounded log-normal observation
#' model, three-cluster Dirichlet-process mixture initial baselines, AR(1)
#' per-feature drift, and AR(1) shared factors with lower-triangular
#' identifiable loadings.
#'
#' @param N Number of subjects.
#' @param T Number of timepoints.
#' @param J Number of features.
#' @param K True number of latent factors used to generate the data.
#' @param G Number of genotype groups (subjects are split evenly).
#' @param phi Per-feature AR(1) coefficient (shared across genotypes here).
#' @param V Innovation variance of the drift process.
#' @param rho AR(1) coefficient of the shared factor process.
#' @param sigma2 Global residual variance for the log-latent.
#' @param mu_cluster Length-3 vector of DP-mixture cluster means for
#'   \eqn{\alpha_{ij1}}.
#' @param w_cluster Length-3 vector of DP-mixture cluster weights.
#' @param r_mean,r_sd Mean and sd of the per-(subject, time) intercept.
#' @param seed Random seed.
#'
#' @return A list with elements
#' \describe{
#'   \item{`Y`}{integer array `N x T x J`, the simulated counts.}
#'   \item{`Y_star`}{numeric array `N x T x J`, the log-latent used to
#'     generate the counts (before flooring).}
#'   \item{`Lambda`,`eta`,`alpha`,`r`}{the parameters used to generate the data.}
#'   \item{`genotype`}{integer vector of length `N` giving each subject's
#'     genotype label in \eqn{\{1, ..., G\}}.}
#'   \item{`Sigma_true`}{array `J x J x T`, the true within-time cross-feature
#'     covariance \eqn{\Lambda\Lambda^\top + \sigma^2 I} at each timepoint.}
#'   \item{`settings`}{list of the simulation settings used.}
#' }
#'
#' @examples
#' sim <- simulate_lodi(N = 10, T = 4, J = 15, K = 2, seed = 1)
#' dim(sim$Y)
#' mean(sim$Y == 0)
#'
#' @importFrom stats rnorm rbinom
#' @export
simulate_lodi <- function(N = 20, T = 7, J = 30, K = 3, G = 1,
                          phi = 0.8, V = 0.2,
                          rho = 0.7, sigma2 = 0.5,
                          mu_cluster = c(-4, 2, 8),
                          w_cluster  = c(0.5, 0.2, 0.3),
                          r_mean = 0, r_sd = 1,
                          seed = 1L) {

  stopifnot(length(mu_cluster) == 3, length(w_cluster) == 3,
            all.equal(sum(w_cluster), 1))
  set.seed(seed)

  # Genotype assignment: split subjects into G roughly equal groups
  genotype <- as.integer(rep(seq_len(G), length.out = N))

  # Lower-triangular Lambda with positive diagonal
  Lambda <- matrix(0, nrow = J, ncol = K)
  for (j in seq_len(J)) {
    kmax <- min(j, K)
    Lambda[j, seq_len(kmax)] <- stats::rnorm(kmax)
    if (j <= K) Lambda[j, j] <- abs(Lambda[j, j])  # positive diagonal
  }

  # Latent factor AR(1) process, one path per subject
  eta <- array(0, dim = c(T, K, N))
  for (i in seq_len(N)) {
    eta[1, , i] <- stats::rnorm(K)
    for (t in seq_len(T - 1L))
      eta[t + 1L, , i] <- rho * eta[t, , i] + stats::rnorm(K)
  }

  # Per-feature drift alpha with DP-mixture initial baseline
  alpha <- array(0, dim = c(N, T, J))
  for (i in seq_len(N)) {
    for (j in seq_len(J)) {
      cl <- sample.int(3L, size = 1L, prob = w_cluster)
      alpha[i, 1L, j] <- stats::rnorm(1L, mean = mu_cluster[cl], sd = 1)
      for (t in seq_len(T - 1L))
        alpha[i, t + 1L, j] <- phi * alpha[i, t, j] +
                                 stats::rnorm(1L, sd = sqrt(V))
    }
  }

  # Sample-scale factor r
  r <- matrix(stats::rnorm(N * T, mean = r_mean, sd = r_sd),
              nrow = N, ncol = T)

  # Assemble log-latent Y_star and count Y
  Y_star <- array(0, dim = c(N, T, J))
  Y      <- array(0L, dim = c(N, T, J))
  for (i in seq_len(N))
    for (t in seq_len(T))
      for (j in seq_len(J)) {
        lin <- r[i, t] + alpha[i, t, j] + sum(Lambda[j, ] * eta[t, , i])
        Y_star[i, t, j] <- lin + stats::rnorm(1L, sd = sqrt(sigma2))
        Y[i, t, j] <- as.integer(floor(exp(Y_star[i, t, j])))
      }

  # True within-time covariance Sigma_t = Lambda Lambda^T + sigma2 I
  LLt <- Lambda %*% t(Lambda)
  Sigma_true <- array(0, dim = c(J, J, T))
  for (t in seq_len(T)) Sigma_true[, , t] <- LLt + sigma2 * diag(J)

  list(
    Y = Y,
    Y_star = Y_star,
    Lambda = Lambda,
    eta = eta,
    alpha = alpha,
    r = r,
    genotype = genotype,
    Sigma_true = Sigma_true,
    settings = list(N = N, T = T, J = J, K = K, G = G,
                    phi = phi, V = V, rho = rho, sigma2 = sigma2,
                    mu_cluster = mu_cluster, w_cluster = w_cluster,
                    seed = seed)
  )
}
