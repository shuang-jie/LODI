#' Fit the LODI dynamic factor model
#'
#' Runs the LODI MCMC on an \eqn{N \times T \times J} array of counts.
#' Given the observed counts, the sampler jointly infers the per-(subject,
#' time) log sample-scale factor \eqn{r_{it}}, the genotype-specific AR(1)
#' per-feature drift \eqn{\alpha_{ijt}} (with a Dirichlet-process mixture
#' prior on the initial baseline \eqn{\alpha_{ij1}}), the shared cross-feature
#' loading matrix \eqn{\Lambda} (identified via a lower-triangular constraint
#' with positive diagonal), the AR(1) latent factors \eqn{\eta_{it}}, and the
#' per-time residual variance \eqn{\sigma^2_t}.
#'
#' @param Y Integer array of counts with dimension \eqn{N \times T \times J}
#'   (subjects x timepoints x features). Missing values are not supported.
#' @param K Integer, the number of latent factors used in the fitted model.
#' @param niter Total number of MCMC iterations.
#' @param nburn Number of initial iterations discarded as burn-in.
#' @param thin Thinning interval: only every `thin`-th post-burn-in iteration
#'   is retained.
#' @param L Truncation level for the stick-breaking approximation to the
#'   Dirichlet-process mixture on \eqn{\alpha_{ij1}}.
#' @param genotype Optional integer vector of length \eqn{N} giving the
#'   genotype/group label of each subject in \eqn{\{1, \dots, G\}}, used for
#'   the genotype-specific AR(1) coefficient \eqn{\phi_g}. Default `NULL`
#'   places all subjects in a single group.
#' @param seed Integer random seed passed to the C++ sampler.
#'
#' @return A list of posterior draws with elements
#' \describe{
#'   \item{`Lambda`}{array `nkeep x J x K`, factor loadings.}
#'   \item{`eta`}{array `nkeep x T x K x N`, latent factor scores.}
#'   \item{`alpha`}{array `nkeep x N x T x J`, per-feature drift trajectories.}
#'   \item{`r`}{array `nkeep x N x T`, sample-scale factors.}
#'   \item{`sigma2`}{matrix `nkeep x T`, per-time residual variance.}
#'   \item{`rho`}{numeric vector `nkeep`, AR(1) coefficient of the factor process.}
#'   \item{`phi_g`}{matrix `nkeep x G`, genotype-specific drift AR(1) coefficients.}
#'   \item{`V`}{numeric vector `nkeep`, drift innovation variance.}
#'   \item{`niter`, `nburn`, `thin`, `K`, `L`}{sampler settings for reference.}
#' }
#'
#' @examples
#' \donttest{
#' sim <- simulate_lodi(N = 20, T = 5, J = 20, K = 3, seed = 1)
#' fit <- lodi(sim$Y, K = 3, niter = 200, nburn = 100, thin = 2, seed = 1)
#' str(fit, max.level = 1)
#' }
#'
#' @export
lodi <- function(Y,
                 K,
                 niter = 4000L,
                 nburn = 2000L,
                 thin  = 1L,
                 L     = 10L,
                 genotype = NULL,
                 seed  = 1L) {

  if (!is.array(Y) || length(dim(Y)) != 3L)
    stop("Y must be a 3-dimensional array of dimension N x T x J.")
  if (!is.numeric(Y) || any(Y < 0) || any(Y != round(Y)))
    stop("Y must contain non-negative integer counts.")

  N <- dim(Y)[1L]; T <- dim(Y)[2L]; J <- dim(Y)[3L]
  if (K < 1L || K > J)
    stop("K must be between 1 and J.")
  if (niter <= nburn)
    stop("niter must exceed nburn.")

  if (is.null(genotype)) {
    X <- matrix(1L, nrow = N, ncol = 1L)
  } else {
    if (length(genotype) != N)
      stop("genotype must be a length-N vector of positive integers.")
    if (any(genotype < 1L) || any(genotype != round(genotype)))
      stop("genotype labels must be positive integers >= 1.")
    X <- matrix(as.integer(genotype), nrow = N, ncol = 1L)
  }

  storage.mode(Y) <- "double"
  Y_vec <- as.vector(Y)
  attr(Y_vec, "dim") <- c(N, T, J)

  raw <- lodi_mcmc_cpp(
    Y_vec = Y_vec,
    seed  = as.integer(seed),
    niter = as.integer(niter),
    nburn = as.integer(nburn),
    thin  = as.integer(thin),
    K     = as.integer(K),
    L     = as.integer(L),
    X     = X
  )

  out <- list(
    Lambda   = raw$Lambda_samples,
    eta      = raw$eta_samples,
    alpha    = raw$alpha_samples,
    r        = raw$r_samples,
    sigma2   = raw$sigma2_t_samples,
    rho      = raw$rho_samples,
    phi_g    = raw$phi_g_samples,
    V        = raw$V_samples,
    niter    = niter,
    nburn    = nburn,
    thin     = thin,
    K        = K,
    L        = L
  )
  class(out) <- c("lodi_fit", "list")
  out
}
