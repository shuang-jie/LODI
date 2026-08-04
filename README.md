# LODI: LOngitudinal Dynamic Interaction Model for Microbiome Counts

<!-- badges: start -->
[![R-CMD-check](https://github.com/shuang-jie/LODI/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/shuang-jie/LODI/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

`LODI` fits a Bayesian dynamic factor model for longitudinal microbiome
count data. The observed counts are modelled through a rounded multivariate
log-normal kernel, and the within-subject temporal signal is decomposed
into two interpretable pieces:

* a **per-feature drift** term, given a genotype-specific AR(1) process
  with a Dirichlet-process mixture prior on the initial baseline, and
* a **shared cross-feature factor** term, given a static loading matrix
  (identified via the standard lower-triangular constraint with positive
  diagonal) paired with an AR(1) latent factor process.

The model returns a time-indexed cross-feature covariance
$\Sigma_t = \Lambda \Lambda^\top + \sigma^2_t I_J$ and cleanly separates
individual taxon trajectories from the shared dynamics that drive
interactions.

For more information, see the paper: **Zhang, Ok, and Ni (submitted)**.

Contact: Shuangjie Zhang <shuangjie.zhang@austin.utexas.edu>.

## Installation

Install R (>= 4.1) and a C++17 toolchain (Xcode command-line tools on
macOS, Rtools on Windows, or the system compiler on Linux). Then:

```r
# install.packages("remotes")
remotes::install_github("shuang-jie/LODI")
```

## Exported functions

| Function | Purpose |
| --- | --- |
| `lodi(Y, K, ...)` | Fit the LODI dynamic factor model by MCMC. Returns a list of posterior draws with class `lodi_fit`. |
| `simulate_lodi(N, T, J, K, ...)` | Draw a synthetic longitudinal count table and the ground-truth parameters from the LODI generative model. |
| `posterior_Sigma(fit, timepoint = NULL)` | Posterior mean of the within-time cross-feature covariance $\Sigma_t$ at one timepoint or across all timepoints. |
| `posterior_rho(fit)` | Posterior summary (mean, sd, 95% CI) of the shared-factor AR(1) coefficient $\rho$. |
| `posterior_phi(fit)` | Posterior summary of the per-genotype drift AR(1) coefficients $\phi_g$. |

## Quick start

```r
library(LODI)

# 1. Simulate a small dataset from the LODI generative model
sim <- simulate_lodi(N = 15, T = 5, J = 20, K = 3, seed = 1)

# 2. Fit the model with a short MCMC run
fit <- lodi(sim$Y, K = 3, niter = 400, nburn = 200, thin = 2, seed = 1)

# 3. Summarise the posterior
posterior_rho(fit)                          # shared-factor AR(1) coefficient
Sigma_hat <- posterior_Sigma(fit, timepoint = 1)  # J x J covariance at t = 1
```

A slightly longer worked example is provided as a package vignette
(`vignette("simulation1", package = "LODI")`).

## Reproducing the paper

This repository ships only the core sampler and a compact worked example
(Simulation 1). The full simulation study (Simulation 2, Steps K and L)
and the real-data analysis reported in the paper live outside the R
package and are available on request.

## License

MIT. See `LICENSE.md`.
