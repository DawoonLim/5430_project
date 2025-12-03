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

# [이전 코드와 동일한 sghmc 함수 유지]
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
# 실험 1 설정 (Double Well Potential)
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

# 실험 1 실행
set.seed(42)
n_iter_fig1 <- 10000
epsilon_fig1 <- 0.1
samples_sghmc <- sghmc(grad_U_noisy_double_well, 0, n_iter_fig1, epsilon_fig1, C = 3, V_hat = 0)
samples_naive <- sghmc(grad_U_noisy_double_well, 0, n_iter_fig1, epsilon_fig1, C = 0, V_hat = 0)

# 실험 1 데이터 프레임
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

# [수정됨] 데이터 프레임 생성 시 Factor 자동 변환 방지 (stringsAsFactors = FALSE)
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
  
  # 여기서 method를 character로 강제하고 stringsAsFactors=FALSE 사용
  return(data.frame(
    step = 1:n_iter, 
    theta = theta_vals, 
    r = r_vals, 
    method = as.character(method), 
    stringsAsFactors = FALSE
  ))
}

# -------------------------------------------------------------------
# 실험 2 실행
# -------------------------------------------------------------------
set.seed(123)
n_iter_fig2 <- 2000 
eps_fig2 <- 0.1

traj_exact <- simulate_trajectory("exact", n_iter_fig2, eps_fig2)
traj_naive <- simulate_trajectory("noisy_naive", n_iter_fig2, eps_fig2)
traj_resample <- simulate_trajectory("noisy_resample", n_iter_fig2, eps_fig2)
traj_sghmc <- simulate_trajectory("sghmc", n_iter_fig2, eps_fig2)

# 데이터 통합 (문자열 상태에서 병합하므로 안전함)
df_fig2 <- rbind(traj_naive, traj_resample, traj_sghmc, traj_exact)

# [중요] Factor 레벨 명시적 지정
df_fig2$method <- factor(df_fig2$method, 
                         levels = c("exact", "noisy_naive", "noisy_resample", "sghmc"))

# -------------------------------------------------------------------
# Figure 2 시각화 (개선된 버전)
# -------------------------------------------------------------------
# 1. 모든 점을 하나의 geom_point로 그립니다 (Size와 Alpha도 method에 따라 다르게 설정)
# 2. "exact"는 geom_path로 선을 그립니다.
# 3. Scale 함수에서 labels와 breaks를 통일하여 범례를 하나로 합칩니다.

plot_fig2 <- ggplot(df_fig2, aes(x = theta, y = r, color = method, shape = method)) +
  
  # 1. 점 그리기 (Size, Alpha도 매핑)
  geom_point(aes(size = method, alpha = method)) +
  
  # 2. 기준선 그리기 (Exact인 경우만 선으로 연결)
  geom_path(data = subset(df_fig2, method == "exact"), 
            size = 1.2, alpha = 0.8) +
  
  # --- 스케일 설정 (범례 이름 'Method'로 통일) ---
  
  # (1) 색상 (Color)
  scale_color_manual(
    name = "Method",
    breaks = c("exact", "noisy_naive", "noisy_resample", "sghmc"),
    labels = c("Exact HMC (Ground Truth)", "Naive Noisy HMC (Diverging)", 
               "Resampling HMC (Inaccurate)", "SGHMC (Correct)"),
    values = c("exact" = "black", "noisy_naive" = "red", 
               "noisy_resample" = "blue", "sghmc" = "forestgreen")
  ) +
  
  # (2) 모양 (Shape) - Exact는 NA로 설정하여 점을 숨김
  scale_shape_manual(
    name = "Method",
    breaks = c("exact", "noisy_naive", "noisy_resample", "sghmc"),
    labels = c("Exact HMC (Ground Truth)", "Naive Noisy HMC (Diverging)", 
               "Resampling HMC (Inaccurate)", "SGHMC (Correct)"),
    values = c("exact" = NA, "noisy_naive" = 4,      # 4: X shape
               "noisy_resample" = 15, "sghmc" = 16)  # 15: Square, 16: Circle
  ) +
  
  # (3) 크기 (Size) - SGHMC를 강조
  scale_size_manual(
    name = "Method",
    breaks = c("exact", "noisy_naive", "noisy_resample", "sghmc"),
    labels = c("Exact HMC (Ground Truth)", "Naive Noisy HMC (Diverging)", 
               "Resampling HMC (Inaccurate)", "SGHMC (Correct)"),
    values = c("exact" = 0.1, "noisy_naive" = 1.5, 
               "noisy_resample" = 1.5, "sghmc" = 2.0)
  ) +
  
  # (4) 투명도 (Alpha) - 겹침 방지
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
# 1. 다차원 지원 SGHMC 함수 (Algorithm 2 Implementation)
# -------------------------------------------------------------------------
sghmc_multidim <- function(grad_U_fn, theta_init, n_iter, epsilon, C, V_hat = 0, M = 1) {
  d <- length(theta_init)
  samples <- matrix(0, nrow = n_iter, ncol = d)
  
  theta <- theta_init
  r <- rnorm(d, 0, sqrt(M)) # Momentum r ~ N(0, M)
  
  # [Review Comment]
  # 논문의 Eq. 13 노이즈 항: N(0, 2(C - B_hat)epsilon)
  # 일반적으로 B_hat은 Gradient Noise의 추정치이므로 epsilon을 곱하지 않고 상수 취급하는 경우가 많으나,
  # 사용자 정의에 따라 V_hat의 스케일이 다를 수 있음. 여기서는 사용자 코드 로직을 유지하되
  # V_hat=0 입력 시 논문의 기본 friction 모드와 동일하게 작동함.
  B_hat <- 0.5 * epsilon * V_hat 
  
  # 노이즈 표준편차 계산
  # C > B_hat 이어야 함 (Friction이 Noise보다 커야 함)
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
    # 논문 Eq 13: - epsilon * grad U - epsilon * C * M^-1 * r + Gaussian Noise
    noise <- rnorm(d, 0, noise_std)
    r <- r - epsilon * grad - epsilon * C * (r / M) + noise
    
    samples[t, ] <- theta
  }
  return(samples)
}

# -------------------------------------------------------------------------
# 2. 고차원 실험 설정 (Multivariate Gaussian Target)
# Target: N(0, Sigma)
# -------------------------------------------------------------------------

#d <- 50 # High Dimension
d <- 200
rho <- 0.5 # Correlation

# 공분산 행렬 (AR(1) 구조)
Sigma <- matrix(0, nrow = d, ncol = d)
for (i in 1:d) {
  for (j in 1:d) {
    Sigma[i, j] <- rho^abs(i - j)
  }
}
Sigma_inv <- solve(Sigma) # Precision Matrix

# Stochastic Gradient 함수
# Minibatch 효과를 시뮬레이션하기 위해 True Gradient에 Gaussian Noise 추가
grad_U_high_dim <- function(theta) {
  true_grad <- as.vector(Sigma_inv %*% theta)
  
  # Noise Scale (논문의 Gradient Noise 시뮬레이션)
  noise_scale <- 1.0 
  grad_noise <- rnorm(length(theta), 0, noise_scale)
  
  return(true_grad + grad_noise)
}

# -------------------------------------------------------------------------
# 3. 실험 실행 (Simulation)
# -------------------------------------------------------------------------
set.seed(2024)
n_iter_hd <- 2000   # 반복 횟수 조정 (빠른 확인용)
epsilon_hd <- 0.01
theta_init_hd <- rep(0, d)

# (1) Naive SGHMC (No Friction, C=0)
# 이론적으로 발산하거나 온도가 너무 높게 측정됨
samples_naive_hd <- sghmc_multidim(grad_U_high_dim, theta_init_hd, n_iter_hd, epsilon_hd, C = 0)

# (2) SGHMC (With Friction, C=3)
# Friction이 Noise를 상쇄하여 적절한 분포 수렴 유도
samples_sghmc_hd <- sghmc_multidim(grad_U_high_dim, theta_init_hd, n_iter_hd, epsilon_hd, C = 3)

# (3) Exact HMC (Reference, No Noise in Gradient, C=0)
grad_U_exact <- function(theta) { as.vector(Sigma_inv %*% theta) }
samples_exact_hd <- sghmc_multidim(grad_U_exact, theta_init_hd, n_iter_hd, epsilon_hd, C = 0)

# -------------------------------------------------------------------------
# 4. 성능 평가 (Evaluation)
# -------------------------------------------------------------------------
evaluate_performance <- function(samples, true_mean, true_cov) {
  n <- nrow(samples)
  # 평가 간격 (Burn-in 고려하여 100부터 시작)
  check_points <- seq(100, n, by = 50)
  
  res_df <- data.frame()
  
  for (t in check_points) {
    # Burn-in 없이 처음부터 누적 (간단한 비교용)
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
# 5. 결과 시각화
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

# 결과 출력
grid.arrange(p1, p2, ncol = 1)





