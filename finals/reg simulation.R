#Basic/Splitting: ϵ을 조금만 키우면 폭발하고, 줄이면 도착을 못 합니다. (Tuning이 매우 까다로움)
#Cyclical: **"초반엔 큰 보폭, 후반엔 작은 보폭"**을 스케줄링하므로, 
#튜닝 없이도 먼 거리를 빨리 이동하고 정착도 잘합니다. 
#고차원 문제에서 왜 Cyclical SGHMC가 SOTA(State-of-the-art)인지 보여주는 완벽한 예시입니다.

# ==============================================================================
# HMC High-Dimensional Comparison: All Rcpp Implementation
# linear regression N = 500; d = 700 
# ==============================================================================

library(hmclearn)
library(Rcpp)
library(ggplot2)
library(dplyr)
library(gridExtra)
library(dplyr)
library(tidyr)
# ------------------------------------------------------------------------------
# 1. Rcpp functions
# ------------------------------------------------------------------------------

sourceCpp(code = '
  #include <Rcpp.h>
  #include <chrono>. 
  #include <cmath>
  
  using namespace Rcpp;

  // ---------------------------------------------------------------------------
  // Gradient Calculation
  // ---------------------------------------------------------------------------
  void calc_gradient(const NumericMatrix& X, const NumericVector& y, 
                     const NumericVector& theta, NumericVector& grad,
                     const IntegerVector& all_indices, int batch_size, 
                     double lambda, int N, int d) {
                     
    // 1. Prior Gradient (Regularization): lambda * theta
    for(int j = 0; j < d; j++) {
      grad[j] = lambda * theta[j];
    }
    
    // 2. Data Subsampling
    IntegerVector indices = sample(all_indices, batch_size);
    double scale = (double)N / batch_size;
    
    // 3. Mini-batch Gradient Accumulation
    for(int i = 0; i < batch_size; i++) {
      int idx = indices[i]; 
      double pred = 0;
      for(int j = 0; j < d; j++) {
        pred += X(idx, j) * theta[j];
      }
      double residual = y[idx] - pred;
      
      for(int j = 0; j < d; j++) {
        grad[j] += scale * (-residual * X(idx, j));
      }
    }
  }

  // ---------------------------------------------------------------------------
  // 1. SGHMC
  // ---------------------------------------------------------------------------
  // [[Rcpp::export]]
  List sghmc_basic_cpp(NumericMatrix X, NumericVector y, NumericVector theta_init, 
                       int n_iter, double epsilon, double C, double M, 
                       double lambda, int batch_size) {
    int N = X.nrow();
    int d = X.ncol();
    NumericMatrix samples(n_iter, d);
    NumericVector times(n_iter); 
    NumericVector theta = clone(theta_init);
    NumericVector r(d);
    NumericVector grad(d);
    IntegerVector all_indices = seq(0, N - 1);
    
    double noise_std = sqrt(2.0 * C * epsilon);
    
    // Init Momentum
    for(int j=0; j<d; j++) r[j] = R::rnorm(0, sqrt(M));

    auto start_time = std::chrono::high_resolution_clock::now();

    for(int t = 0; t < n_iter; t++) {
      // A: Theta Update
      for(int j = 0; j < d; j++) theta[j] += epsilon * (r[j] / M);
      
      // B: Gradient Update
      calc_gradient(X, y, theta, grad, all_indices, batch_size, lambda, N, d);
      
      // C: Momentum Update
      for(int j = 0; j < d; j++) {
        double noise = R::rnorm(0, noise_std);
        r[j] = r[j] - epsilon * grad[j] - epsilon * C * (r[j] / M) + noise;
      }
      
      samples(t, _) = theta;
      
      auto current_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = current_time - start_time;
      times[t] = elapsed.count();
    }
    return List::create(Named("Time") = times, Named("Theta") = samples);
  }

  // ---------------------------------------------------------------------------
  // 2. Splitting SGHMC
  // ---------------------------------------------------------------------------
  // [[Rcpp::export]]
  List sghmc_splitting_cpp(NumericMatrix X, NumericVector y, NumericVector theta_init, 
                           int n_iter, double epsilon, double C, double lambda, 
                           int batch_size, double M) {
    int N = X.nrow();
    int d = X.ncol();
    NumericMatrix samples(n_iter, d);
    NumericVector times(n_iter);
    NumericVector theta = clone(theta_init);
    NumericVector r(d);
    NumericVector grad(d);
    IntegerVector all_indices = seq(0, N - 1);

    // Init Momentum
    for(int j=0; j<d; j++) r[j] = R::rnorm(0, sqrt(M));

    double decay = exp(-C * epsilon);
    double noise_std = sqrt(1.0 - exp(-2.0 * C * epsilon));

    // Initial Gradient
    calc_gradient(X, y, theta, grad, all_indices, batch_size, lambda, N, d);

    auto start_time = std::chrono::high_resolution_clock::now();

    for(int t = 0; t < n_iter; t++) {
      // Momentum half-step
      for(int j=0; j<d; j++) r[j] -= 0.5 * epsilon * grad[j];
      
      // Theta half-step
      for(int j=0; j<d; j++) theta[j] += 0.5 * epsilon * (r[j] / M);
      
      // Noise/Friction step (Ornstein-Uhlenbeck)
      for(int j=0; j<d; j++) {
        double noise = R::rnorm(0, 1);
        r[j] = r[j] * decay + (sqrt(M) * noise_std * noise);
      }
      
      // Theta half-step
      for(int j=0; j<d; j++) theta[j] += 0.5 * epsilon * (r[j] / M);
      
      // Gradient Re-calculation (New Batch)
      // Note: Splitting requires gradient at new position for the next momentum step
      calc_gradient(X, y, theta, grad, all_indices, batch_size, lambda, N, d);
      
      // Momentum half-step
      for(int j=0; j<d; j++) r[j] -= 0.5 * epsilon * grad[j];

      samples(t, _) = theta;
      
      auto current_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = current_time - start_time;
      times[t] = elapsed.count();
    }
    return List::create(Named("Time") = times, Named("Theta") = samples);
  }

  // ---------------------------------------------------------------------------
  // 3. Adaptive SGHMC
  // ---------------------------------------------------------------------------
  // [[Rcpp::export]]
  List sghmc_adaptive_cpp(NumericMatrix X, NumericVector y, NumericVector theta_init, 
                          int n_iter, double epsilon, double C, double lambda, 
                          int batch_size, double mdecay) {
    int N = X.nrow();
    int d = X.ncol();
    NumericMatrix samples(n_iter, d);
    NumericVector times(n_iter);
    NumericVector theta = clone(theta_init);
    NumericVector v(d); // Velocity
    NumericVector grad(d);
    NumericVector g_sq(d); // Gradient squared accumulator
    IntegerVector all_indices = seq(0, N - 1);

    // Init v and g_sq
    for(int j=0; j<d; j++) {
        v[j] = R::rnorm(0, sqrt(epsilon));
        g_sq[j] = 1e-8; // Avoid division by zero
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for(int t = 0; t < n_iter; t++) {
      // Calculate Gradient
      calc_gradient(X, y, theta, grad, all_indices, batch_size, lambda, N, d);
      
      // Update Adaptive Preconditioner & Momentum
      for(int j=0; j<d; j++) {
        // Accumulate squared gradient
        g_sq[j] = (1.0 - mdecay) * g_sq[j] + mdecay * (grad[j] * grad[j]);
        
        double sigma_inv = 1.0 / sqrt(g_sq[j]);
        double noise_std_adaptive = sqrt(2.0 * epsilon * C) * sqrt(sigma_inv);
        double noise = R::rnorm(0, 1) * noise_std_adaptive;
        
        // Update velocity (momentum proxy)
        v[j] = v[j] - (epsilon * grad[j] * sigma_inv) - (epsilon * C * v[j] * sigma_inv) + noise;
        
        // Update Theta
        theta[j] += v[j];
      }

      samples(t, _) = theta;
      
      auto current_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = current_time - start_time;
      times[t] = elapsed.count();
    }
    return List::create(Named("Time") = times, Named("Theta") = samples);
  }

  // ---------------------------------------------------------------------------
  // 4. Cyclical SGHMC
  // ---------------------------------------------------------------------------
// [[Rcpp::export]]
List sghmc_cyclical_cpp(NumericMatrix X, NumericVector y, NumericVector theta_init, 
                        int n_iter, double epsilon_max, int cycle_length, 
                        int batch_size, double lambda, double beta) { // lambda 인자 추가
  int N = X.nrow();
  int d = X.ncol();
  NumericMatrix samples(n_iter, d);
  NumericVector times(n_iter);
  NumericVector theta = clone(theta_init);
  NumericVector m(d);
  NumericVector grad(d);
  IntegerVector all_indices = seq(0, N - 1);
  
  const double PI_VAL = 3.14159265358979323846;
  
  for(int j=0; j<d; j++) m[j] = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  for(int t = 0; t < n_iter; t++) {
    double r_cyc = (double)(t % cycle_length) / cycle_length;
    double epsilon_t = epsilon_max * (cos(PI_VAL * r_cyc) + 1.0) / 2.0;
    
    // lambda value
    for(int j = 0; j < d; j++) grad[j] = lambda * theta[j];
    
    IntegerVector indices = sample(all_indices, batch_size);
    double scale = (double)N / batch_size;
    for(int i = 0; i < batch_size; i++) {
      int idx = indices[i]; 
      double pred = 0;
      for(int j = 0; j < d; j++) pred += X(idx, j) * theta[j];
      double residual = y[idx] - pred;
      for(int j = 0; j < d; j++) grad[j] += scale * (-residual * X(idx, j));
    }
    
    double noise_scale = sqrt(1.0 * (1.0 - beta) * epsilon_t);
    for(int j=0; j<d; j++) {
       double noise = R::rnorm(0, 1) * noise_scale;
       m[j] = beta * m[j] - (epsilon_t / 2.0) * grad[j] + noise;
       theta[j] += m[j];
    }
    samples(t, _) = theta;
    auto current_time = std::chrono::high_resolution_clock::now();
    times[t] = std::chrono::duration<double>(current_time - start_time).count();
  }
  return List::create(Named("Time") = times, Named("Theta") = samples);
}
')


# ------------------------------------------------------------------------------
# 2. High-Dimensional Regression Setting
# ------------------------------------------------------------------------------
set.seed(1)

# 사용자 설정
N <- 500
d <- 700 
n_iter <- 2000
batch_size <- 50

# Sparsity level = 5/700
true_theta <- numeric(d)
true_theta[1:5] <- c(2, -1, 0.5, -0.5, 1) 

# Data Generating
X <- matrix(rnorm(N * d), nrow = N, ncol = d)
y <- as.vector(X %*% true_theta + rnorm(N, 0, 1.5))
theta_init <- rep(0, d) 


# ------------------------------------------------------------------------------
# 3. 알고리즘 실행 및 비교
# ------------------------------------------------------------------------------
print("Starting HMC Benchmark...")

# (1) hmclearn::hmc
t0 <- Sys.time()
linear_posterior <- function(theta, ...) {
  dots <- list(...)
  if (!is.null(dots$param)) { y <- dots$param$y; X <- dots$param$X }
  p <- length(theta) - 1
  beta <- theta[1:p]
  log_sigma_sq <- theta[p + 1]
  sigma2 <- exp(log_sigma_sq)
  resid <- y - X %*% beta
  n <- length(y)
  ll <- -0.5 * n * log(2 * pi * sigma2) - 0.5 * sum(resid^2) / sigma2
  lp_beta <- -0.5 * sum(beta^2) / 1000 # Prior
  return(as.numeric(ll + lp_beta))
}

g_linear_posterior <- function(theta, ...) {
  dots <- list(...)
  if (!is.null(dots$param)) { y <- dots$param$y; X <- dots$param$X }
  p <- length(theta) - 1
  beta <- theta[1:p]
  log_sigma_sq <- theta[p + 1]
  sigma2 <- exp(log_sigma_sq)
  resid <- y - X %*% beta
  n <- length(y)
  g_beta <- as.vector(t(X) %*% resid / sigma2 - beta / 1000)
  g_log_sigma_sq <- -0.5 * n + 0.5 * sum(resid^2) / sigma2
  return(c(g_beta, as.numeric(g_log_sigma_sq)))
}

theta_init_hmc <- c(theta_init, 0) 
res_hmc <- hmclearn::hmc(
  N = n_iter, theta.init = theta_init_hmc, epsilon = 0.005, L = 10,
  logPOSTERIOR = linear_posterior, glogPOSTERIOR = g_linear_posterior,
  param = list(y = y, X = X), parallel=FALSE, chains=1
)
time_hmc <- as.numeric(Sys.time() - t0)
samples_hmc <- do.call(rbind, res_hmc$thetaCombined)[, 1:d, drop = FALSE]


# (2) SGHMC
res_basic <- sghmc_basic_cpp(X, y, theta_init, n_iter, 
                             epsilon = 8e-5, 
                             C = 50.0, M = 1.0, 
                             lambda = 0.001,  # [수정] 1.0 -> 0.001
                             batch_size = batch_size)

# (3) Splitting SGHMC
res_split <- sghmc_splitting_cpp(X, y, theta_init, n_iter, 
                                 epsilon = 1e-4, 
                                 C = 50.0, 
                                 lambda = 0.001,  # [수정] 1.0 -> 0.001
                                 batch_size = batch_size, M = 1.0)

# (4) BOHAMIANN
# (Adaptive는 epsilon을 조금 더 작게 해야 안전할 수 있습니다)
res_adapt <- sghmc_adaptive_cpp(X, y, theta_init, n_iter, 
                                epsilon = 5e-9, 
                                C = 50.0, 
                                lambda = 0.001,  # [수정] 1.0 -> 0.001
                                batch_size = batch_size, mdecay = 0.01)

# (5) cSGHMC
# Max epsilon을 줄여야 함
res_cycle <- sghmc_cyclical_cpp(X, y, theta_init, n_iter, 
                                epsilon_max = 5e-5, 
                                cycle_length = 600, 
                                batch_size = batch_size, 
                                lambda = 0.001, # [수정] 0.001 전달
                                beta = 0.9)

# True Values
true_values <- c(2, -1, 0.5, -0.5, 1)
param_names <- paste0("Theta", 1:5)

# 함수: 1~5번째 컬럼을 모두 추출하여 Long Format으로 변환
extract_all_params <- function(algo_name, samples) {
  mat <- as.matrix(samples)
  
  # 데이터프레임 변환 (Theta1 ~ Theta5)
  df <- as.data.frame(mat[, 1:5])
  colnames(df) <- param_names
  
  df %>%
    mutate(Iteration = 1:n(), Algorithm = algo_name) %>%
    pivot_longer(cols = all_of(param_names), 
                 names_to = "Parameter", 
                 values_to = "Value")
}

# 데이터 추출
df_hmc_all   <- extract_all_params("HMC", samples_hmc)
df_basic_all <- extract_all_params("SGHMC", res_basic$Theta)
df_split_all <- extract_all_params("Splitting SGHMC", res_split$Theta)
df_adapt_all <- extract_all_params("BOHAMIANN", res_adapt$Theta)
df_cycle_all <- extract_all_params("cSGHMC", res_cycle$Theta)

# 통합
df_all_params <- rbind(df_hmc_all, df_basic_all, df_split_all, df_adapt_all, df_cycle_all)

# Factor 순서 지정
algo_levels <- c("HMC", "SGHMC", "Splitting SGHMC", "BOHAMIANN", "cSGHMC")
df_all_params$Algorithm <- factor(df_all_params$Algorithm, levels = algo_levels)

# ------------------------------------------------------------------------------
# 2. Loop를 통한 시각화 (Theta 1 ~ 5 각각 출력)
# ------------------------------------------------------------------------------

for (j in 1:5) {
  p_name <- param_names[j]
  t_val  <- true_values[j]
  
  # 해당 파라미터 데이터만 필터링
  df_sub <- df_all_params %>% filter(Parameter == p_name)
  
  # -------------------------
  # A. Trace Plot
  # -------------------------
  p_trace <- ggplot(df_sub %>% filter(Iteration > 100), 
                    aes(x=Iteration, y=Value, color=Algorithm)) +
    geom_line(alpha=0.8, linewidth=0.3) +
    geom_hline(yintercept = t_val, linetype="dashed", color="red", linewidth=0.8) +
    facet_wrap(~Algorithm, ncol=1, scales="free_y") + 
    theme_bw() +
    labs(#title = paste0("Trace Plot: ", p_name),
         #subtitle = paste0("True Value = ", t_val),
         y = "Parameter Value") +
    theme(legend.position="none", 
          strip.text = element_text(face="bold"),
          strip.background = element_blank(),
          panel.grid = element_blank())
  
  # -------------------------
  # B. Density Plot (Burn-in 500 이후)
  # -------------------------
  # 동적 X축 범위 설정: True Value 기준으로 ±1.5 범위
  x_lims <- c(t_val - 1.0, t_val + 1.0)
  
  p_dens <- ggplot(df_sub %>% filter(Iteration > 500), 
                   aes(x=Value, fill=Algorithm)) +
    geom_density(alpha=0.4, color=NA) +
    geom_vline(xintercept = t_val, linetype="dashed", color="black", linewidth=0.8) +
    scale_fill_brewer(palette = "Set1") + # 색상 팔레트
    theme_classic() +
    labs(#title = paste0("Posterior Density: ", p_name),
         #subtitle = "After Burn-in (500)",
         x = "Parameter Value") +
    coord_cartesian(xlim = x_lims) + # 튀는 값 제외하고 관심 영역 확대
    theme(legend.position="bottom")

  print(p_trace)
  print(p_dens) 
}




# ------------------------------------------------------------------------------
# 3. 요약 통계량 (Summary Stats)
# ------------------------------------------------------------------------------
# 1. True Value 매핑 데이터 (이미 정의되어 있다면 생략 가능)
true_vals <- data.frame(
  Parameter = c("Theta1", "Theta2", "Theta3", "Theta4", "Theta5"),
  True_Value = c(2.0, -1.0, 0.5, -0.5, 1.0)
)

# 2. MSE를 포함한 최종 요약표 생성
final_summary <- df_all_params %>%
  filter(Iteration > 500) %>%                 
  left_join(true_vals, by = "Parameter") %>%
  group_by(Algorithm, Parameter) %>%
  summarise(
    Mean   = mean(Value, na.rm = TRUE),       
    SD     = sd(Value, na.rm = TRUE),         
    True_V = first(True_Value),               
    Bias   = mean(Value, na.rm = TRUE) - first(True_Value), 
    MSE    = mean((Value - True_Value)^2, na.rm = TRUE),    
    .groups = 'drop'
  ) %>%
  arrange(Parameter, Algorithm) # 파라미터별로 정렬하여 비교 용이하게

# 3. 결과 출력 (전체 행)
print(final_summary, n = 30)

burn_in <- 500
calc_mse <- function(samp, true_th, burn) {
  if(is.null(samp)) return(NA)
  valid <- as.matrix(samp)[(burn+1):nrow(samp), , drop=FALSE]
  mean((colMeans(valid) - true_th)^2)
}

results <- data.frame(
  Algorithm = c("HMC", "SGHMC", "Splitting", "BOHAMIANN", "cSGHMC"),
  Time_Sec = c(time_hmc, max(res_basic$Time), max(res_split$Time), max(res_adapt$Time), max(res_cycle$Time)),
  MSE = c(
    calc_mse(samples_hmc, true_theta, burn_in),
    calc_mse(res_basic$Theta, true_theta, burn_in),
    calc_mse(res_split$Theta, true_theta, burn_in),
    calc_mse(res_adapt$Theta, true_theta, burn_in),
    calc_mse(res_cycle$Theta, true_theta, burn_in)
  )
)
print(results)
