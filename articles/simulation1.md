# Simulation 1: matched truth

This vignette reproduces a small version of “Simulation 1: matched
truth” from Zhang, Ok, and Ni (submitted), “Bayesian Nonparametric
Longitudinal Factor Model for Microbiome Counts.” It generates counts
from LODI’s own generative model and shows that the MCMC recovers the
within-time cross-feature covariance
$`\Sigma_t = \Lambda\Lambda^\top + \sigma^2_t I_J`$ under matched
conditions.

Settings are chosen small so the vignette runs in under a minute; larger
runs simply increase `N`, `T`, `J`, and `niter`.

``` r

library(LODI)
set.seed(1)
```

## 1. Simulate a matched-truth dataset

``` r

sim <- simulate_lodi(
  N = 15, T = 5, J = 20, K = 3,
  phi = 0.8, V = 0.2, rho = 0.7, sigma2 = 0.5,
  seed = 1
)
dim(sim$Y)          # N x T x J
#> [1] 15  5 20
mean(sim$Y == 0)    # simulated zero rate
#> [1] 0.528
```

## 2. Fit LODI

Short-run MCMC settings for the vignette; production runs typically use
tens of thousands of iterations.

``` r

fit <- lodi(
  Y = sim$Y,
  K = 3,
  niter = 400,
  nburn = 200,
  thin  = 2,
  seed  = 1
)
```

## 3. Recovery of the shared-factor AR(1) coefficient

Truth is `rho = 0.7`:

``` r

posterior_rho(fit)
#>       mean         sd       q025       q500       q975 
#> 0.68163786 0.08588373 0.52579189 0.67292704 0.84157349
```

## 4. Recovery of the within-time covariance

Element-wise RMSE between posterior mean and truth (Frobenius-style,
over the whole $`J\times J`$ matrix):

``` r

Sigma_hat  <- posterior_Sigma(fit, timepoint = 1)
Sigma_true <- sim$Sigma_true[, , 1]
sqrt(mean((Sigma_hat - Sigma_true)^2))
#> [1] 0.7773142
```

## 5. Full simulation

To reproduce the full Simulation 1 in the paper, increase to `N = 20`,
`T = 7`, `J = 50`, `niter = 30000`, `nburn = 15000`, `thin = 5` and
repeat over 10 seeds.
