# Simulate a longitudinal count table from the LODI generative model

Draws a synthetic dataset matching the "Simulation 1: matched truth"
setting of Zhang, Ok, and Ni (submitted), "Bayesian Nonparametric
Longitudinal Factor Model for Microbiome Counts": rounded log-normal
observation model, three-cluster Dirichlet-process mixture initial
baselines, AR(1) per-feature drift, and AR(1) shared factors with
lower-triangular identifiable loadings.

## Usage

``` r
simulate_lodi(
  N = 20,
  T = 7,
  J = 30,
  K = 3,
  G = 1,
  phi = 0.8,
  V = 0.2,
  rho = 0.7,
  sigma2 = 0.5,
  mu_cluster = c(-4, 2, 8),
  w_cluster = c(0.5, 0.2, 0.3),
  r_mean = 0,
  r_sd = 1,
  seed = 1L
)
```

## Arguments

- N:

  Number of subjects.

- T:

  Number of timepoints.

- J:

  Number of features.

- K:

  True number of latent factors used to generate the data.

- G:

  Number of genotype groups (subjects are split evenly).

- phi:

  Per-feature AR(1) coefficient (shared across genotypes here).

- V:

  Innovation variance of the drift process.

- rho:

  AR(1) coefficient of the shared factor process.

- sigma2:

  Global residual variance for the log-latent.

- mu_cluster:

  Length-3 vector of DP-mixture cluster means for \\\alpha\_{ij1}\\.

- w_cluster:

  Length-3 vector of DP-mixture cluster weights.

- r_mean, r_sd:

  Mean and sd of the per-(subject, time) intercept.

- seed:

  Random seed.

## Value

A list with elements

- `Y`:

  integer array `N x T x J`, the simulated counts.

- `Y_star`:

  numeric array `N x T x J`, the log-latent used to generate the counts
  (before flooring).

- `Lambda`,`eta`,`alpha`,`r`:

  the parameters used to generate the data.

- `genotype`:

  integer vector of length `N` giving each subject's genotype label in
  \\\\1, ..., G\\\\.

- `Sigma_true`:

  array `J x J x T`, the true within-time cross-feature covariance
  \\\Lambda\Lambda^\top + \sigma^2 I\\ at each timepoint.

- `settings`:

  list of the simulation settings used.

## Examples

``` r
sim <- simulate_lodi(N = 10, T = 4, J = 15, K = 2, seed = 1)
dim(sim$Y)
#> [1] 10  4 15
mean(sim$Y == 0)
#> [1] 0.4916667
```
