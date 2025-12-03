# ==============================================================================
# HMC vs SGHMC (Pure R & Rcpp)
# ==============================================================================

library(hmclearn)
library(Rcpp)
library(ggplot2)
library(gridExtra)

# ------------------------------------------------------------------------------
# 1. functions
# ------------------------------------------------------------------------------

# (1) Basic SGHMC (Rcpp Version)
cppFunction('
  #include <Rcpp.h>
  #include <chrono> 
  using namespace Rcpp;

  // [[Rcpp::export]]
  List sghmc_cpp_timed(NumericMatrix X, NumericVector y, 
                       NumericVector theta_init, 
                       int n_iter, double epsilon, double C, 
                       double M, double lambda, int batch_size) {
    
    int N = X.nrow();
    int d = X.ncol();
    NumericMatrix samples(n_iter, d);
    NumericVector times(n_iter); 
    NumericVector theta = clone(theta_init);
    NumericVector r(d);
    
    // Init Momentum
    for(int i=0; i<d; i++) r[i] = R::rnorm(0, sqrt(M));
    
    double noise_std = sqrt(2.0 * C * epsilon);
    NumericVector grad(d);
    double scale = (double)N / batch_size;
    IntegerVector all_indices = seq(0, N - 1);
    
    auto start_time = std::chrono::high_resolution_clock::now();

    for(int t = 0; t < n_iter; t++) {
      // A: Theta Update
      for(int j = 0; j < d; j++) theta[j] += epsilon * (r[j] / M);
      
      // B: Gradient Update
      for(int j = 0; j < d; j++) grad[j] = lambda * theta[j];
      IntegerVector indices = sample(all_indices, batch_size);
      for(int i = 0; i < batch_size; i++) {
        int idx = indices[i]; 
        double pred = 0;
        for(int j = 0; j < d; j++) pred += X(idx, j) * theta[j];
        double residual = y[idx] - pred;
        for(int j = 0; j < d; j++) grad[j] += scale * (-residual * X(idx, j));
      }
      
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
')

# (2) Splitting SGHMC (Pure R Version)
sghmc_splitting_r <- function(X, y, theta_init, n_iter, epsilon, C, lambda, batch_size, M = 1) {
  d <- length(theta_init)
  N <- nrow(X)
  samples <- matrix(0, nrow = n_iter, ncol = d)
  times <- numeric(n_iter)
  theta <- theta_init
  r <- rnorm(d, 0, sqrt(M))
  
  decay <- exp(-C * epsilon)
  noise_std <- sqrt(1 - exp(-2 * C * epsilon))
  
  # Initial Gradient
  indices <- sample(1:N, batch_size)
  X_batch <- X[indices, , drop=FALSE]
  y_batch <- y[indices]
  residual <- y_batch - X_batch %*% theta
  grad <- as.vector((N / batch_size) * (-t(X_batch) %*% residual) + lambda * theta)
  
  start_time <- Sys.time()
  for (t in 1:n_iter) {
    r <- r - 0.5 * epsilon * grad
    theta <- theta + 0.5 * epsilon * (r / M)
    
    noise <- rnorm(d, 0, 1)
    r <- r * decay + (sqrt(M) * noise_std) * noise
    
    theta <- theta + 0.5 * epsilon * (r / M)
    
    indices <- sample(1:N, batch_size)
    X_batch <- X[indices, , drop=FALSE]
    y_batch <- y[indices]
    residual <- y_batch - X_batch %*% theta
    grad <- as.vector((N / batch_size) * (-t(X_batch) %*% residual) + lambda * theta)
    
    r <- r - 0.5 * epsilon * grad
    
    samples[t, ] <- theta
    times[t] <- as.numeric(Sys.time() - start_time)
  }
  return(list(Time = times, Theta = samples))
}

# (3) Adaptive SGHMC (Pure R Version)
sghmc_adaptive_r <- function(X, y, theta_init, n_iter, epsilon, C, lambda, batch_size, mdecay = 0.05) {
  N <- nrow(X)
  d <- length(theta_init)
  samples <- matrix(0, nrow = n_iter, ncol = d)
  times <- numeric(n_iter)
  theta <- theta_init
  v <- rnorm(d, 0, sqrt(epsilon)) 
  g_sq <- rep(1e-8, d)
  scale <- N / batch_size
  
  start_time <- Sys.time()
  for (t in 1:n_iter) {
    indices <- sample(1:N, batch_size)
    X_batch <- X[indices, , drop=FALSE]
    y_batch <- y[indices]
    pred <- X_batch %*% theta
    residual <- y_batch - pred
    grad <- as.vector(scale * (-t(X_batch) %*% residual) + lambda * theta)
    
    g_sq <- (1 - mdecay) * g_sq + mdecay * (grad^2)
    sigma_inv <- 1 / sqrt(g_sq)
    noise_std <- sqrt(2 * epsilon * C) * sqrt(sigma_inv)
    noise <- rnorm(d, 0, 1) * noise_std
    
    v <- v - (epsilon * grad * sigma_inv) - (epsilon * C * v * sigma_inv) + noise
    theta <- theta + v
    
    samples[t, ] <- theta
    times[t] <- as.numeric(Sys.time() - start_time)
  }
  return(list(Time = times, Theta = samples))
}

# (4) Cyclical SGHMC (Pure R Version)
sghmc_cyclical_r <- function(X, y, theta_init, n_iter, epsilon_max, cycle_length, batch_size, beta = 0.9) {
  N <- nrow(X)
  d <- length(theta_init)
  samples <- matrix(0, nrow = n_iter, ncol = d)
  times <- numeric(n_iter)
  theta <- theta_init
  m <- rep(0, d)
  scale <- N / batch_size
  
  start_time <- Sys.time()
  for (t in 1:n_iter) {
    r_cyc <- (t %% cycle_length) / cycle_length
    epsilon_t <- epsilon_max * (cos(pi * r_cyc) + 1) / 2
    
    indices <- sample(1:N, batch_size)
    X_batch <- X[indices, , drop=FALSE]
    y_batch <- y[indices]
    grad_U <- as.vector(scale * (-t(X_batch) %*% (y_batch - X_batch %*% theta)) + theta)
    
    noise_scale <- sqrt(1.0 * (1 - beta) * epsilon_t)
    noise <- rnorm(d, 0, 1) * noise_scale
    m <- beta * m - (epsilon_t / 2) * grad_U + noise
    theta <- theta + m
    
    samples[t, ] <- theta
    times[t] <- as.numeric(Sys.time() - start_time)
  }
  return(list(Time = times, Theta = samples))
}

# ------------------------------------------------------------------------------
# 2. setting
# ------------------------------------------------------------------------------
set.seed(42)
N <- 2000; d <- 5
true_theta <- c(2, -1, 0.5, -0.5, 1)
X <- matrix(rnorm(N * d), nrow = N, ncol = d)
y <- as.vector(X %*% true_theta + rnorm(N, 0, 1.5))

n_iter <- 3000
batch_size <- 200
theta_init <- rep(0, d)


# ------------------------------------------------------------------------------
# 1. Baseline: HMC
# ------------------------------------------------------------------------------
t0 <- Sys.time()
linear_posterior <- function(theta, ...) {
  dots <- list(...)
  if (!is.null(dots$param)) {
    y <- dots$param$y; X <- dots$param$X
  } else (!is.null(dots$y) && !is.null(dots$X)) {
    y <- dots$y; X <- dots$X
  }
  
  p <- length(theta) - 1
  beta <- theta[1:p]
  log_sigma_sq <- theta[p + 1]
  sigma2 <- exp(log_sigma_sq)
  
  resid <- y - X %*% beta
  n <- length(y)
  
  ll <- -0.5 * n * log(2 * pi * sigma2) - 0.5 * sum(resid^2) / sigma2
  lp_beta <- -0.5 * sum(beta^2) / 1000
  
  return(as.numeric(ll + lp_beta))
}

g_linear_posterior <- function(theta, ...) {
  dots <- list(...)
  if (!is.null(dots$param)) {
    y <- dots$param$y; X <- dots$param$X
  } else (!is.null(dots$y) && !is.null(dots$X)) {
    y <- dots$y; X <- dots$X
  }
  
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
  N = n_iter,          
  theta.init = theta_init_hmc, # d+1 차원
  epsilon = 0.001,          
  L = 10,                  
  logPOSTERIOR = linear_posterior,    
  glogPOSTERIOR = g_linear_posterior, 
  param = list(y = y, X = X)
  , parallel=FALSE, chains=1
)

time_hmc <- as.numeric(Sys.time() - t0)


theta_mat <- do.call(rbind, res_hmc$thetaCombined)
samples_hmc <- theta_mat[, 1:d, drop = FALSE]


# --- (2) Basic SGHMC (Rcpp) ---
res_basic <- sghmc_cpp_timed(X, y, theta_init, n_iter, epsilon=0.01, C=10.0, M=1.0, lambda=1.0, batch_size=batch_size)

# --- (3) Splitting SGHMC (R) ---
res_split <- sghmc_splitting_r(X, y, theta_init, n_iter, epsilon=0.01, C=10.0, lambda=1.0, batch_size=batch_size)

# --- (4) Adaptive SGHMC (R) ---
res_adapt <- sghmc_adaptive_r(X, y, theta_init, n_iter, epsilon=0.005, C=5.0, lambda=1.0, batch_size=batch_size, mdecay=0.1)

# --- (5) Cyclical SGHMC (R) ---
res_cycle <- sghmc_cyclical_r(X, y, theta_init, n_iter, epsilon_max=0.001, cycle_length=600, batch_size=batch_size)

# ------------------------------------------------------------------------------
# 3. Result
# ------------------------------------------------------------------------------
burn_in <- 200
calc_mse <- function(samp, true_th, burn) {
  if(is.null(samp)) return(NA)
  valid <- as.matrix(samp)[(burn+1):nrow(samp), , drop=FALSE]
  mean((colMeans(valid) - true_th)^2)
}

results <- data.frame(
  Algorithm = c("HMC (Base)", "Basic (Rcpp)", "Splitting (R)", "Adaptive (R)", "Cyclical (R)"),
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


