# hmc vs sghmc vs splitting sghmc vs adaptive mass sghmc vs cyclical(recent method)
# r implementation
# find simulation that sghmc with rcpp works best in perspective of precision and effiency
# proper simulaton: standard benchmark needed (before detailed alalysis, sghmc provides baseline)
# & unimodal case
# Find proper simulation where sghmc_rcpp is the best
# splitting: PyTorch 
# adaptive: python
# cyclical: PyTorch
# ===================================================================
# SGHMC paper simulation reproduction
# ===================================================================

library(ggplot2)
library(gridExtra)

# ===================================================================
# SECTION 1: SGHMC Density Comparison (Fig 1 Reproduction)
# ===================================================================

sghmc <- function(grad_U_fn, theta_init, n_iter, epsilon, C, V_hat = 0, M = 1) {
  samples <- numeric(n_iter)
  theta <- theta_init
  r <- rnorm(1, 0, sqrt(M))
  
  B_hat <- 0.5 * epsilon * V_hat
  
  for (t in 1:n_iter) {
    theta <- theta + epsilon * (r / M)
    grad <- grad_U_fn(theta)
    noise_std <- sqrt(2 * (C - B_hat) * epsilon)
    if(is.nan(noise_std)) noise_std <- 0 
    
    noise <- rnorm(1, 0, noise_std)
    r <- r - epsilon * grad - epsilon * C * (r / M) + noise
    samples[t] <- theta
  }
  return(samples)
}

# -------------------------------------------------------------------
# (Double Well Potential)
# -------------------------------------------------------------------
grad_U_noisy_double_well <- function(theta) {
  true_grad <- -4 * theta + 4 * theta^3
  noise <- rnorm(1, mean = 0, sd = 2) 
  return(true_grad + noise)
}

target_density <- function(x) {
  U <- -2 * x^2 + x^4
  exp(-U)
}
normalization_const <- integrate(target_density, -2.5, 2.5)$value

set.seed(42)
n_iter_fig1 <- 10000
epsilon_fig1 <- 0.1
samples_sghmc <- sghmc(grad_U_noisy_double_well, 0, n_iter_fig1, epsilon_fig1, C = 3, V_hat = 0)
samples_naive <- sghmc(grad_U_noisy_double_well, 0, n_iter_fig1, epsilon_fig1, C = 0, V_hat = 0)

df_res_fig1 <- data.frame(
  theta = c(samples_sghmc, samples_naive),
  Method = rep(c("SGHMC (With Friction)", "Naive (No Friction)"), each = n_iter_fig1)
)

x_vals <- seq(-2, 2, length.out = 200)
y_vals <- target_density(x_vals) / normalization_const
df_true_fig1 <- data.frame(x = x_vals, y = y_vals)

plot_fig1 <- ggplot() +
  geom_density(data = df_res_fig1, aes(x = theta, color = Method, fill = Method), alpha = 0.3) +
  geom_line(data = df_true_fig1, aes(x = x, y = y), color = "black", size = 1, linetype = "dashed") +
  labs(title = "Figure 1: Density Comparison", x = "Theta", y = "Density") +
  theme_minimal() +
  scale_x_continuous(limits = c(-2.5, 2.5)) +
  theme(legend.position = "bottom")


# ===================================================================
# SECTION 2: Phase Space Trajectories (Fig 2 Reproduction)
# ===================================================================

simulate_trajectory <- function(method, n_iter, epsilon, theta_init = 0, r_init = 1) {
  
  theta_vals <- numeric(n_iter)
  r_vals <- numeric(n_iter)
  theta <- theta_init
  r <- r_init
  
  grad_noise_sd <- 2
  C_friction <- 0.5 * epsilon * (grad_noise_sd^2) 
  
  for (t in 1:n_iter) {
    true_grad <- theta
    if (method == "exact") {
      grad <- true_grad
    } else {
      grad <- true_grad + rnorm(1, 0, grad_noise_sd)
    }
    
    theta <- theta + epsilon * r
    
    if (method == "exact") {
      r <- r - epsilon * grad
    } else if (method == "sghmc") {
      r <- r - epsilon * grad - epsilon * C_friction * r
    } else {
      r <- r - epsilon * grad
    }
    
    if (method == "noisy_resample" && t %% 50 == 0) {
      r <- rnorm(1, 0, 1) 
    }
    
    theta_vals[t] <- theta
    r_vals[t] <- r
  }
  
  
  return(data.frame(
    step = 1:n_iter, 
    theta = theta_vals, 
    r = r_vals, 
    method = as.character(method), 
    stringsAsFactors = FALSE
  ))
}

# -------------------------------------------------------------------
# simulation 2
# -------------------------------------------------------------------
set.seed(123)
n_iter_fig2 <- 2000 
eps_fig2 <- 0.1

traj_exact <- simulate_trajectory("exact", n_iter_fig2, eps_fig2)
traj_naive <- simulate_trajectory("noisy_naive", n_iter_fig2, eps_fig2)
traj_resample <- simulate_trajectory("noisy_resample", n_iter_fig2, eps_fig2)
traj_sghmc <- simulate_trajectory("sghmc", n_iter_fig2, eps_fig2)

df_fig2 <- rbind(traj_naive, traj_resample, traj_sghmc, traj_exact)

# Factor 
df_fig2$method <- factor(df_fig2$method, 
                         levels = c("exact", "noisy_naive", "noisy_resample", "sghmc"))

# -------------------------------------------------------------------
# Figure 2 visual
# -------------------------------------------------------------------

plot_fig2 <- ggplot(df_fig2, aes(x = theta, y = r, color = method, shape = method)) +
  
  geom_point(aes(size = method, alpha = method)) +
  
  geom_path(data = subset(df_fig2, method == "exact"), 
            size = 1.2, alpha = 0.8) +
  
  
  # (1) (Color)
  scale_color_manual(
    name = "Method",
    breaks = c("exact", "noisy_naive", "noisy_resample", "sghmc"),
    labels = c("Exact HMC (Ground Truth)", "Naive Noisy HMC (Diverging)", 
               "Resampling HMC (Inaccurate)", "SGHMC (Correct)"),
    values = c("exact" = "black", "noisy_naive" = "red", 
               "noisy_resample" = "blue", "sghmc" = "forestgreen")
  ) +
  
  # (2) (Shape)
  scale_shape_manual(
    name = "Method",
    breaks = c("exact", "noisy_naive", "noisy_resample", "sghmc"),
    labels = c("Exact HMC (Ground Truth)", "Naive Noisy HMC (Diverging)", 
               "Resampling HMC (Inaccurate)", "SGHMC (Correct)"),
    values = c("exact" = NA, "noisy_naive" = 4,      # 4: X shape
               "noisy_resample" = 15, "sghmc" = 16)  # 15: Square, 16: Circle
  ) +
  
  # (3) (Size)
  scale_size_manual(
    name = "Method",
    breaks = c("exact", "noisy_naive", "noisy_resample", "sghmc"),
    labels = c("Exact HMC (Ground Truth)", "Naive Noisy HMC (Diverging)", 
               "Resampling HMC (Inaccurate)", "SGHMC (Correct)"),
    values = c("exact" = 0.1, "noisy_naive" = 1.5, 
               "noisy_resample" = 1.5, "sghmc" = 2.0)
  ) +
  
  # (4) (Alpha)
  scale_alpha_manual(
    name = "Method",
    breaks = c("exact", "noisy_naive", "noisy_resample", "sghmc"),
    labels = c("Exact HMC (Ground Truth)", "Naive Noisy HMC (Diverging)", 
               "Resampling HMC (Inaccurate)", "SGHMC (Correct)"),
    values = c("exact" = 0, "noisy_naive" = 0.6,    # exact 점은 완전 투명
               "noisy_resample" = 0.6, "sghmc" = 0.4)
  ) +
  
  labs(
    title = "Figure 2: Phase Space Trajectories",
    subtitle = "Correct: SGHMC (Green Circle) matches Exact dynamics (Black Line)",
    x = "Theta",
    y = "Momentum (r)"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom", legend.box = "vertical") +
  coord_fixed() + 
  scale_x_continuous(limits = c(-6, 6)) +
  scale_y_continuous(limits = c(-6, 6))

print(plot_fig1)
print(plot_fig2)







# -------------------------------------------------------------------------
# Multivariate Gaussian covariance; hmc vs naive hmc vs sghmc
# -------------------------------------------------------------------------
library(ggplot2)
library(gridExtra)
# -------------------------------------------------------------------------
# 1. (Algorithm 2 Implementation)
# -------------------------------------------------------------------------
sghmc_multidim <- function(grad_U_fn, theta_init, n_iter, epsilon, C, V_hat = 0, M = 1) {
  d <- length(theta_init)
  samples <- matrix(0, nrow = n_iter, ncol = d)
  
  theta <- theta_init
  r <- rnorm(d, 0, sqrt(M)) # Momentum r ~ N(0, M)
  
  B_hat <- 0.5 * epsilon * V_hat 
  
  # C > B_hat 
  noise_var_term <- 2 * (C - B_hat) * epsilon
  if(noise_var_term < 0) {
    warning("Friction C must be greater than estimated noise B_hat.")
    noise_var_term <- 0
  }
  noise_std <- sqrt(noise_var_term)
  
  for (t in 1:n_iter) {
    # (A) Theta Update: theta_{t+1} <- theta_t + epsilon * M^-1 * r_t
    theta <- theta + epsilon * (r / M)
    
    # (B) Gradient Update: grad U(theta_{t+1})
    grad <- grad_U_fn(theta)
    
    # (C) Momentum Update: r_{t+1} ...
    # Eq 13: - epsilon * grad U - epsilon * C * M^-1 * r + Gaussian Noise
    noise <- rnorm(d, 0, noise_std)
    r <- r - epsilon * grad - epsilon * C * (r / M) + noise
    
    samples[t, ] <- theta
  }
  return(samples)
}

# -------------------------------------------------------------------------
# 2. (Multivariate Gaussian Target)
# Target: N(0, Sigma)
# -------------------------------------------------------------------------

#d <- 50 # High Dimension
d <- 200
rho <- 0.5 # Correlation

# (AR(1)
Sigma <- matrix(0, nrow = d, ncol = d)
for (i in 1:d) {
  for (j in 1:d) {
    Sigma[i, j] <- rho^abs(i - j)
  }
}
Sigma_inv <- solve(Sigma) # Precision Matrix

# Stochastic Gradient 
grad_U_high_dim <- function(theta) {
  true_grad <- as.vector(Sigma_inv %*% theta)
  
  # Noise Scale 
  noise_scale <- 1.0 
  grad_noise <- rnorm(length(theta), 0, noise_scale)
  
  return(true_grad + grad_noise)
}

# -------------------------------------------------------------------------
# 3.(Simulation)
# -------------------------------------------------------------------------
set.seed(2024)
n_iter_hd <- 2000   
epsilon_hd <- 0.01
theta_init_hd <- rep(0, d)

# (1) Naive SGHMC (No Friction, C=0)
samples_naive_hd <- sghmc_multidim(grad_U_high_dim, theta_init_hd, n_iter_hd, epsilon_hd, C = 0)

# (2) SGHMC (With Friction, C=3)
samples_sghmc_hd <- sghmc_multidim(grad_U_high_dim, theta_init_hd, n_iter_hd, epsilon_hd, C = 3)

# (3) Exact HMC (Reference, No Noise in Gradient, C=0)
grad_U_exact <- function(theta) { as.vector(Sigma_inv %*% theta) }
samples_exact_hd <- sghmc_multidim(grad_U_exact, theta_init_hd, n_iter_hd, epsilon_hd, C = 0)

# -------------------------------------------------------------------------
# 4. (Evaluation)
# -------------------------------------------------------------------------
evaluate_performance <- function(samples, true_mean, true_cov) {
  n <- nrow(samples)
  check_points <- seq(100, n, by = 50)
  
  res_df <- data.frame()
  
  for (t in check_points) {
    current_samples <- samples[1:t, ]
    
    # Mean Error (MSE)
    est_mean <- colMeans(current_samples)
    err_mean <- mean((est_mean - true_mean)^2) 
    
    # Covariance Error (Frobenius Norm)
    est_cov <- cov(current_samples)
    err_cov <- sqrt(sum((est_cov - true_cov)^2))
    
    res_df <- rbind(res_df, data.frame(Iter = t, MeanError = err_mean, CovError = err_cov))
  }
  return(res_df)
}

true_mean <- rep(0, d)

df_naive <- evaluate_performance(samples_naive_hd, true_mean, Sigma)
df_naive$Method <- "Naive SGHMC (C=0)"

df_sghmc <- evaluate_performance(samples_sghmc_hd, true_mean, Sigma)
df_sghmc$Method <- "SGHMC (C=3)"

df_exact <- evaluate_performance(samples_exact_hd, true_mean, Sigma)
df_exact$Method <- "Exact HMC (Ref)"

df_total <- rbind(df_naive, df_sghmc, df_exact)

# -------------------------------------------------------------------------
# 5. visual
# -------------------------------------------------------------------------
p1 <- ggplot(df_total, aes(x = Iter, y = MeanError, color = Method)) +
  geom_line(linewidth = 1) +
  labs(title = paste0("Mean Estimation Error (d=", d, ")"),
       y = "MSE of Mean") +
  theme_minimal() + 
  theme(legend.position = "none")

p2 <- ggplot(df_total, aes(x = Iter, y = CovError, color = Method)) +
  geom_line(linewidth = 1) +
  labs(title = "Covariance Estimation Error (Frobenius Norm)",
       y = "Covariance Error") +
  theme_minimal() + 
  theme(legend.position = "bottom")

grid.arrange(p1, p2, ncol = 1)





