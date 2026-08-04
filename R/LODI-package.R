#' LODI: LOngitudinal Dynamic Interaction Model for Microbiome Count Data
#'
#' A Bayesian dynamic factor model for longitudinal microbiome count data.
#' LODI models the observed counts through a rounded multivariate log-normal
#' kernel and decomposes the within-subject temporal signal into a per-feature
#' drift term (a genotype-specific AR(1) process with a Dirichlet-process
#' mixture prior on the initial baseline) and a shared cross-feature factor
#' term (a static loading matrix paired with an AR(1) latent factor process).
#'
#' The main entry points are:
#'
#' * [lodi()] --- fit the model by MCMC.
#' * [simulate_lodi()] --- generate a synthetic longitudinal count table from
#'   the LODI generative model.
#' * [posterior_Sigma()] --- posterior within-time cross-feature covariance.
#' * [posterior_rho()], [posterior_phi()] --- posterior of the AR(1)
#'   coefficients for the shared factor process and per-feature drift.
#'
#' @keywords internal
#' @name LODI-package
#' @aliases LODI
#' @useDynLib LODI, .registration = TRUE
#' @importFrom Rcpp sourceCpp
"_PACKAGE"
