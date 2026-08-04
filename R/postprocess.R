#' Posterior within-time cross-feature covariance
#'
#' Computes the posterior mean of the within-time cross-feature covariance
#' \eqn{\Sigma_t = \Lambda \Lambda^\top + \sigma^2_t I_J} at each timepoint.
#'
#' @param fit A fitted object returned by [lodi()].
#' @param timepoint Optional integer in `1:T`. If given, returns a `J x J`
#'   matrix for that timepoint; otherwise returns a `J x J x T` array with the
#'   posterior mean at every timepoint.
#' @return Either a `J x J` matrix or a `J x J x T` array.
#' @export
posterior_Sigma <- function(fit, timepoint = NULL) {
  stopifnot(inherits(fit, "lodi_fit"))
  Lam <- fit$Lambda        # nkeep x J x K
  sig2 <- fit$sigma2       # nkeep x T
  nkeep <- dim(Lam)[1L]
  J <- dim(Lam)[2L]
  Tn <- ncol(sig2)

  # Posterior mean of Lambda Lambda^T
  LLt_mean <- matrix(0, J, J)
  for (s in seq_len(nkeep)) {
    L_s <- Lam[s, , , drop = TRUE]
    if (is.matrix(L_s)) LLt_mean <- LLt_mean + tcrossprod(L_s)
  }
  LLt_mean <- LLt_mean / nkeep

  sig2_mean <- colMeans(sig2)
  Iden <- diag(J)

  if (!is.null(timepoint)) {
    if (timepoint < 1L || timepoint > Tn)
      stop("timepoint out of range 1:", Tn)
    return(LLt_mean + sig2_mean[timepoint] * Iden)
  }

  out <- array(0, dim = c(J, J, Tn))
  for (t in seq_len(Tn)) out[, , t] <- LLt_mean + sig2_mean[t] * Iden
  out
}

#' Posterior summary of the shared-factor AR(1) coefficient
#'
#' @param fit A fitted object returned by [lodi()].
#' @return A numeric vector with elements `mean`, `sd`, `q025`, `q500`, `q975`.
#' @importFrom stats sd quantile
#' @export
posterior_rho <- function(fit) {
  stopifnot(inherits(fit, "lodi_fit"))
  .summary_vec(fit$rho)
}

#' Posterior summary of the per-feature drift AR(1) coefficients
#'
#' @param fit A fitted object returned by [lodi()].
#' @return A matrix with one row per genotype group and columns `mean`, `sd`,
#'   `q025`, `q500`, `q975`.
#' @importFrom stats sd quantile
#' @export
posterior_phi <- function(fit) {
  stopifnot(inherits(fit, "lodi_fit"))
  phi_g <- fit$phi_g
  if (is.null(dim(phi_g))) phi_g <- matrix(phi_g, ncol = 1L)
  G <- ncol(phi_g)
  out <- t(vapply(seq_len(G), function(g) .summary_vec(phi_g[, g]),
                  numeric(5L)))
  rownames(out) <- paste0("genotype_", seq_len(G))
  out
}

.summary_vec <- function(x) {
  qs <- stats::quantile(x, probs = c(0.025, 0.5, 0.975), names = FALSE)
  c(mean = mean(x), sd = stats::sd(x),
    q025 = qs[1L], q500 = qs[2L], q975 = qs[3L])
}
