# Posterior summary of the per-feature drift AR(1) coefficients

Posterior summary of the per-feature drift AR(1) coefficients

## Usage

``` r
posterior_phi(fit)
```

## Arguments

- fit:

  A fitted object returned by
  [`lodi()`](https://shuang-jie.github.io/LODI/reference/lodi.md).

## Value

A matrix with one row per genotype group and columns `mean`, `sd`,
`q025`, `q500`, `q975`.
