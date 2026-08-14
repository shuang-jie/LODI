# Fit the LODI dynamic factor model

Runs the LODI MCMC on an \\N \times T \times J\\ array of counts. Given
the observed counts, the sampler jointly infers the per-(subject, time)
log sample-scale factor \\r\_{it}\\, the genotype-specific AR(1)
per-feature drift \\\alpha\_{ijt}\\ (with a Dirichlet-process mixture
prior on the initial baseline \\\alpha\_{ij1}\\), the shared
cross-feature loading matrix \\\Lambda\\ (identified via a
lower-triangular constraint with positive diagonal), the AR(1) latent
factors \\\eta\_{it}\\, and the per-time residual variance
\\\sigma^2_t\\.

## Usage

``` r
lodi(
  Y,
  K,
  niter = 4000L,
  nburn = 2000L,
  thin = 1L,
  L = 10L,
  genotype = NULL,
  seed = 1L
)
```

## Arguments

- Y:

  Integer array of counts with dimension \\N \times T \times J\\
  (subjects x timepoints x features). Missing values are not supported.

- K:

  Integer, the number of latent factors used in the fitted model.

- niter:

  Total number of MCMC iterations.

- nburn:

  Number of initial iterations discarded as burn-in.

- thin:

  Thinning interval: only every `thin`-th post-burn-in iteration is
  retained.

- L:

  Truncation level for the stick-breaking approximation to the
  Dirichlet-process mixture on \\\alpha\_{ij1}\\.

- genotype:

  Optional integer vector of length \\N\\ giving the genotype/group
  label of each subject in \\\\1, \dots, G\\\\, used for the
  genotype-specific AR(1) coefficient \\\phi_g\\. Default `NULL` places
  all subjects in a single group.

- seed:

  Integer random seed passed to the C++ sampler.

## Value

A list of posterior draws with elements

- `Lambda`:

  array `nkeep x J x K`, factor loadings.

- `eta`:

  array `nkeep x T x K x N`, latent factor scores.

- `alpha`:

  array `nkeep x N x T x J`, per-feature drift trajectories.

- `r`:

  array `nkeep x N x T`, sample-scale factors.

- `sigma2`:

  matrix `nkeep x T`, per-time residual variance.

- `rho`:

  numeric vector `nkeep`, AR(1) coefficient of the factor process.

- `phi_g`:

  matrix `nkeep x G`, genotype-specific drift AR(1) coefficients.

- `V`:

  numeric vector `nkeep`, drift innovation variance.

- `niter`, `nburn`, `thin`, `K`, `L`:

  sampler settings for reference.

## Examples

``` r
# \donttest{
sim <- simulate_lodi(N = 20, T = 5, J = 20, K = 3, seed = 1)
fit <- lodi(sim$Y, K = 3, niter = 200, nburn = 100, thin = 2, seed = 1)
str(fit, max.level = 1)
#> List of 13
#>  $ Lambda: num [1:50, 1:20, 1:3] 0.399 0.345 0.499 0.397 0.469 ...
#>  $ eta   : num [1:50, 1:20, 1:5, 1:3] -1.26 -1.45 -1.8 -2.01 -1.41 ...
#>  $ alpha : num [1:50, 1:20, 1:5, 1:20] -4.63 -5.4 -4.83 -4.93 -5.01 ...
#>  $ r     : num [1:50, 1:20, 1:5] 0.269 0.373 0.555 0.508 0.485 ...
#>  $ sigma2: num [1:50, 1:5] 0.475 0.526 0.627 0.619 0.605 ...
#>  $ rho   : num [1:50] 0.527 0.675 0.59 0.67 0.675 ...
#>  $ phi_g : num [1:50, 1] 0.79 0.793 0.785 0.791 0.784 ...
#>  $ V     : num [1:50] 0.449 0.439 0.44 0.412 0.402 ...
#>  $ niter : num 200
#>  $ nburn : num 100
#>  $ thin  : num 2
#>  $ K     : num 3
#>  $ L     : int 10
#>  - attr(*, "class")= chr [1:2] "lodi_fit" "list"
# }
```
