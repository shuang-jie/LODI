# Package index

## Fit

Run the LODI MCMC on a longitudinal count table.

- [`lodi()`](https://shuang-jie.github.io/LODI/reference/lodi.md) : Fit
  the LODI dynamic factor model

## Simulate

Generate a matched-truth longitudinal count dataset from the LODI
generative model.

- [`simulate_lodi()`](https://shuang-jie.github.io/LODI/reference/simulate_lodi.md)
  : Simulate a longitudinal count table from the LODI generative model

## Posterior summaries

Post-process a fitted `lodi_fit` into interpretable estimates.

- [`posterior_Sigma()`](https://shuang-jie.github.io/LODI/reference/posterior_Sigma.md)
  : Posterior within-time cross-feature covariance
- [`posterior_rho()`](https://shuang-jie.github.io/LODI/reference/posterior_rho.md)
  : Posterior summary of the shared-factor AR(1) coefficient
- [`posterior_phi()`](https://shuang-jie.github.io/LODI/reference/posterior_phi.md)
  : Posterior summary of the per-feature drift AR(1) coefficients
