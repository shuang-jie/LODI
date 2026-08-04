test_that("simulate_lodi returns the expected shape and non-negative integers", {
  sim <- simulate_lodi(N = 8, T = 4, J = 10, K = 2, seed = 42)
  expect_equal(dim(sim$Y), c(8, 4, 10))
  expect_true(all(sim$Y >= 0))
  expect_true(all(sim$Y == round(sim$Y)))
  expect_equal(dim(sim$Lambda), c(10, 2))
  # Lower-triangular Lambda: entry (1, 2) must be 0
  expect_equal(sim$Lambda[1, 2], 0)
  # Positive diagonal
  expect_true(sim$Lambda[1, 1] > 0)
  expect_true(sim$Lambda[2, 2] > 0)
})

test_that("lodi runs a short chain without error and returns expected objects", {
  sim <- simulate_lodi(N = 8, T = 4, J = 10, K = 2, seed = 42)
  fit <- lodi(sim$Y, K = 2, niter = 80, nburn = 40, thin = 2, seed = 42)
  expect_s3_class(fit, "lodi_fit")
  expect_setequal(names(fit),
                  c("Lambda", "eta", "alpha", "r", "sigma2", "rho",
                    "phi_g", "V", "niter", "nburn", "thin", "K", "L"))
  # Lambda draws: nkeep x J x K, with nkeep = (niter - nburn)/thin = 20
  expect_equal(dim(fit$Lambda)[2:3], c(10, 2))
})

test_that("posterior_Sigma and posterior_rho give sensible output on a tiny run", {
  sim <- simulate_lodi(N = 8, T = 4, J = 10, K = 2, seed = 42)
  fit <- lodi(sim$Y, K = 2, niter = 80, nburn = 40, thin = 2, seed = 42)

  Sigma_all <- posterior_Sigma(fit)
  expect_equal(dim(Sigma_all), c(10, 10, 4))
  # Diagonal must be positive
  expect_true(all(diag(Sigma_all[, , 1]) > 0))

  Sigma_t1 <- posterior_Sigma(fit, timepoint = 1)
  expect_equal(dim(Sigma_t1), c(10, 10))

  rho_summary <- posterior_rho(fit)
  expect_named(rho_summary, c("mean", "sd", "q025", "q500", "q975"))
  expect_true(rho_summary["mean"] >= -1 && rho_summary["mean"] <= 1)
})

test_that("input validation rejects malformed Y and K", {
  Y_ok <- simulate_lodi(N = 5, T = 3, J = 6, K = 1, seed = 1)$Y
  expect_error(lodi(as.numeric(Y_ok), K = 1),
               "3-dimensional")
  Y_bad <- Y_ok; Y_bad[1] <- -1
  expect_error(lodi(Y_bad, K = 1),
               "non-negative")
  expect_error(lodi(Y_ok, K = 10),
               "K must be between 1 and J")
})
