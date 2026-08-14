# LODI: LOngitudinal Dynamic Interaction Model for Microbiome Count Data

A Bayesian dynamic factor model for longitudinal microbiome count data.
LODI models the observed counts through a rounded multivariate
log-normal kernel and decomposes the within-subject temporal signal into
a per-feature drift term (a genotype-specific AR(1) process with a
Dirichlet-process mixture prior on the initial baseline) and a shared
cross-feature factor term (a static loading matrix paired with an AR(1)
latent factor process).

## Details

The main entry points are:

- [`lodi()`](https://shuang-jie.github.io/LODI/reference/lodi.md) — fit
  the model by MCMC.

- [`simulate_lodi()`](https://shuang-jie.github.io/LODI/reference/simulate_lodi.md)
  — generate a synthetic longitudinal count table from the LODI
  generative model.

- [`posterior_Sigma()`](https://shuang-jie.github.io/LODI/reference/posterior_Sigma.md)
  — posterior within-time cross-feature covariance.

- [`posterior_rho()`](https://shuang-jie.github.io/LODI/reference/posterior_rho.md),
  [`posterior_phi()`](https://shuang-jie.github.io/LODI/reference/posterior_phi.md)
  — posterior of the AR(1) coefficients for the shared factor process
  and per-feature drift.

## See also

Useful links:

- <https://github.com/shuang-jie/LODI>

- Report bugs at <https://github.com/shuang-jie/LODI/issues>

## Author

**Maintainer**: Shuangjie Zhang <shuangjie.zhang@austin.utexas.edu>
