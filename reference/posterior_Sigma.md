# Posterior within-time cross-feature covariance

Computes the posterior mean of the within-time cross-feature covariance
\\\Sigma_t = \Lambda \Lambda^\top + \sigma^2_t I_J\\ at each timepoint.

## Usage

``` r
posterior_Sigma(fit, timepoint = NULL)
```

## Arguments

- fit:

  A fitted object returned by
  [`lodi()`](https://shuang-jie.github.io/LODI/reference/lodi.md).

- timepoint:

  Optional integer in `1:T`. If given, returns a `J x J` matrix for that
  timepoint; otherwise returns a `J x J x T` array with the posterior
  mean at every timepoint.

## Value

Either a `J x J` matrix or a `J x J x T` array.
