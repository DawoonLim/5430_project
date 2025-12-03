library(mcmc)        # metrop (random-walk Metropolis)
library(adaptMCMC)   # adaptive Metropolis
library(rstan)       # HMC / NUTS
library(coda)        # ESS, diagnostics
library(posterior)   # tidy draws from stan
library(ggplot2)
library(gridExtra)
library(sgmcmc)
sgmcmc::sghmc

?set.seed(123)

# ---------------------------
# 1) 2D correlated MVN
# ---------------------------
mu <- c(0,0)
Sigma <- matrix(c(1, 0.9,
                  0.9, 1), 2, 2)
Sigma_inv <- solve(Sigma)
log_target <- function(x) {
  # x: numeric vector length 2
  z <- as.numeric(x - mu)
  -0.5 * crossprod(z, Sigma_inv %*% z)
}

# Helper: wrapper to adapt signatures
log_target_for_adapt <- function(x, ...) {
  # adaptMCMC expects logpdf(x, ...).
  return(as.numeric(log_target(x)))
}

# ---------------------------
# 2) 1) Random-walk Metropolis (mcmc::metrop)
# ---------------------------
init <- c(0,0)
niter <- 5000

cat("Running Random-walk Metropolis (mcmc::metrop)...\n")
t0 <- Sys.time()
rw_out <- metrop(log_target, initial = init, nbatch = niter, scale = 0.5, debug = FALSE)
t1 <- Sys.time()
time_rw <- as.numeric(difftime(t1, t0, units = "secs"))
# metrop returns $batch: matrix of samples (nbatch x dim)
samples_rw <- rw_out$batch
colnames(samples_rw) <- c("x1", "x2")

# ---------------------------
# 3) Adaptive Metropolis (adaptMCMC::MCMC)
# ---------------------------
cat("Running Adaptive Metropolis (adaptMCMC::MCMC)...\n")
t0 <- Sys.time()
am_out <- MCMC(log_target_for_adapt, n = niter, init = init, adapt = TRUE, acc.rate = 0.234, list = TRUE)
t1 <- Sys.time()
time_am <- as.numeric(difftime(t1, t0, units = "secs"))
# adaptMCMC returns $samples matrix (n x dim)
samples_am <- am_out$samples
colnames(samples_am) <- c("x1", "x2")

# ---------------------------
# 4) HMC / NUTS via rstan
# ---------------------------
cat("Compiling simple Stan model (2D MVN) and sampling via NUTS...\n")
stan_code <- "
data {
  vector[2] mu;
  matrix[2,2] Sigma;
}
parameters {
  vector[2] x;
}
model {
  x ~ multi_normal(mu, Sigma);
}
"
sm <- stan_model(model_code = stan_code)
t0 <- Sys.time()
fit <- sampling(sm, data = list(mu = mu, Sigma = Sigma),
                iter = 2000, warmup = 1000, chains = 2, refresh = 0)
t1 <- Sys.time()
time_hmc <- as.numeric(difftime(t1, t0, units = "secs"))

# posterior
samps_hmc <- as.matrix(fit, pars = "x")   
colnames(samps_hmc) <- c("x1", "x2")     
samples_hmc <- samps_hmc

# ---------------------------
# 5) Compute diagnostics: ESS, ESS/sec, acceptance-ish info
# ---------------------------
# Convert to mcmc objects for coda functions
mcmc_rw  <- as.mcmc(samples_rw)
mcmc_am  <- as.mcmc(samples_am)
mcmc_hmc <- as.mcmc(samples_hmc)

ess_rw  <- effectiveSize(mcmc_rw)   # vector per parameter
ess_am  <- effectiveSize(mcmc_am)
ess_hmc <- effectiveSize(mcmc_hmc)

esssec_rw  <- ess_rw / time_rw
esssec_am  <- ess_am / time_am
esssec_hmc <- ess_hmc / time_hmc

# Summarize
summary_df <- data.frame(
  Algorithm = rep(c("RWM", "AdaptiveMH", "NUTS"), each = 2),
  Param = rep(c("x1", "x2"), times = 3),
  ESS = c(ess_rw, ess_am, ess_hmc),
  Time_sec = c(rep(time_rw, 2), rep(time_am, 2), rep(time_hmc, 2)),
  ESS_per_sec = c(esssec_rw, esssec_am, esssec_hmc)
)

print("=== Summary (ESS, Time, ESS/sec) ===")
print(summary_df)

# ---------------------------
# 6) Plots: trace, ACF, scatter (combined)
# ---------------------------
# Helper plotting functions for ggplot
df_rw  <- data.frame(iter = 1:nrow(samples_rw), samples_rw, Algo = "RWM")
df_am  <- data.frame(iter = 1:nrow(samples_am), samples_am, Algo = "AdaptiveMH")
df_hmc <- data.frame(iter = 1:nrow(samples_hmc), samples_hmc, Algo = "NUTS")

# TRACE plots for x1
p_trace1 <- ggplot(rbind(df_rw, df_am, df_hmc), aes(x = iter, y = x1, color = Algo)) +
  geom_line(alpha = 0.6) + theme_minimal() + labs(title = "Trace plot (x1)", x = "Iteration", y = "x1")

p_trace2 <- ggplot(rbind(df_rw, df_am, df_hmc), aes(x = iter, y = x2, color = Algo)) +
  geom_line(alpha = 0.6) + theme_minimal() + labs(title = "Trace plot (x2)", x = "Iteration", y = "x2")

# Scatter plots of samples (thin for visibility)
thin_rate <- 5
samp_comb <- rbind(df_rw[seq(1,nrow(df_rw),thin_rate), c("x1","x2","Algo")],
                   df_am[seq(1,nrow(df_am),thin_rate), c("x1","x2","Algo")],
                   df_hmc[seq(1,nrow(df_hmc),thin_rate), c("x1","x2","Algo")])
p_scatter <- ggplot(samp_comb, aes(x = x1, y = x2, color = Algo)) +
  geom_point(alpha = 0.6) + theme_minimal() + labs(title = "Samples scatter (thinned)", x = "x1", y = "x2") +
  coord_equal()

# ACF plots (use base acf for clarity)
# We'll create small ACF plots for x1 of each algorithm and use grid.arrange
par(mfrow = c(3,1))
acf(as.numeric(samples_rw[,"x1"]), main = "ACF: RWM (x1)")
acf(as.numeric(samples_am[,"x1"]), main = "ACF: AdaptiveMH (x1)")
acf(as.numeric(samples_hmc[,"x1"]), main = "ACF: NUTS (x1)")
par(mfrow = c(1,1))

# Show ggplots side by side
grid.arrange(p_trace1, p_trace2, p_scatter, ncol = 1)

# ---------------------------
# 7) Print runtime + ESS summary
# ---------------------------
cat("\nRuntimes (seconds):\n")
cat(sprintf(" - RWM: %.2f s\n - AdaptiveMH: %.2f s\n - NUTS: %.2f s\n", time_rw, time_am, time_hmc))

cat("\nESS (per param) and ESS/sec:\n")
print(summary_df)

# End

# SGHMC with diagonal B_hat (simple implementation)
# grad_fn(theta, minibatch_idx) -> gradient vector (length D) computed on that minibatch
# sample_minibatch_indices() -> function that returns indices for one minibatch
# data, model specifics are user-provided

sghmc_custom <- function(theta0, grad_fn, sample_minibatch_indices,
                         D, n_iters = 5000, m = 128, epsilon = 1e-3,
                         K_est = 200, alpha_C = 1.2, mass_inv = rep(1, D),
                         reestimate_B_every = 1000, verbose = TRUE) {
  
  theta <- theta0
  r <- rep(0, D)                      # momentum
  chain <- matrix(NA, nrow = n_iters, ncol = D)
  
  # 1) 초기 B_hat 추정 (대각)
  grads <- matrix(0, nrow = K_est, ncol = D)
  for (k in 1:K_est) {
    idx <- sample_minibatch_indices()
    grads[k, ] <- grad_fn(theta, idx)
  }
  Bhat_diag <- apply(grads, 2, var)  # per-dim variance of minibatch gradients
  
  # 2) set friction C (diagonal) conservatively
  C_diag <- pmax(alpha_C * Bhat_diag, 1e-10)  # alpha>1 권장
  
  for (t in 1:n_iters) {
    # optional: re-estimate B_hat periodically
    if (t %% reestimate_B_every == 0 && t > 0) {
      K2 <- min(200, K_est)
      G2 <- matrix(0, nrow = K2, ncol = D)
      for (k in 1:K2) {
        idx <- sample_minibatch_indices()
        G2[k, ] <- grad_fn(theta, idx)
      }
      Bhat_diag <- apply(G2, 2, var)
      C_diag <- pmax(alpha_C * Bhat_diag, C_diag * 0.5) # keep stable floor
    }
    
    # compute gradient on one minibatch
    idx <- sample_minibatch_indices()
    g <- grad_fn(theta, idx)   # vector length D: stochastic gradient (minibatch)
    
    # compute noise standard deviation vector
    noise_cov_diag <- 2 * (C_diag - Bhat_diag) * epsilon
    # numerical safety: ensure non-negative
    noise_cov_diag[noise_cov_diag < 0] <- 0
    noise_sd <- sqrt(noise_cov_diag)
    
    # SGHMC momentum update (diagonal mass_inv)
    # r <- r - epsilon * g - epsilon * C * mass_inv * r + N(0, noise_cov)
    r <- r - epsilon * g - epsilon * (C_diag * (mass_inv * r)) + rnorm(D, 0, noise_sd)
    
    # theta update
    theta <- theta + epsilon * (mass_inv * r)
    
    chain[t, ] <- theta
    
    if (verbose && (t %% 500 == 0)) {
      cat(sprintf("iter %d: mean theta = %s\n", t, paste0(round(mean(theta),4), "")))
    }
  }
  
  list(chain = chain, Bhat_diag = Bhat_diag, C_diag = C_diag)
}
