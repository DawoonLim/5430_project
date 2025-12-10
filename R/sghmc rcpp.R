library(ggplot2)
library(gridExtra)
library(MASS)
library(Rcpp)

# -------------------------------------------------------------------------
# 1. Rcpp SGHMC
# -------------------------------------------------------------------------
cppFunction('
  #include <Rcpp.h>
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
    for(int i=0; i<d; i++) r[i] = rnorm(1, 0, sqrt(M))[0];
    
    double noise_std = sqrt(2.0 * C * epsilon);
    NumericVector grad(d);
    double scale = (double)N / batch_size;
    
    IntegerVector all_indices = seq(0, N - 1);
    
    Function sys_time("Sys.time");
    double start_time = as<double>(sys_time());

    for(int t = 0; t < n_iter; t++) {
      
      // (A) Theta Update
      for(int j = 0; j < d; j++) theta[j] += epsilon * (r[j] / M);
      
      // (B) Gradient Update
      for(int j = 0; j < d; j++) grad[j] = lambda * theta[j];
      
      IntegerVector indices = sample(all_indices, batch_size);
      
      for(int i = 0; i < batch_size; i++) {
        int idx = indices[i]; 
        
        double pred = 0;
        for(int j = 0; j < d; j++) pred += X(idx, j) * theta[j];
        
        double residual = y[idx] - pred;
        for(int j = 0; j < d; j++) grad[j] += scale * (-residual * X(idx, j));
      }
      
      // (C) Momentum Update
      for(int j = 0; j < d; j++) {
        double noise = rnorm(1, 0, noise_std)[0];
        r[j] = r[j] - epsilon * grad[j] - epsilon * C * (r[j] / M) + noise;
      }
      
      samples(t, _) = theta;
      
      double current_time = as<double>(sys_time());
      times[t] = current_time - start_time;
    }
    
    // [FIX] DataFrame 대신 List::create 사용
    return List::create(Named("Time") = times, Named("Theta") = samples);
  }
')





########################################################################
cppFunction('
#include <Rcpp.h>
#include <chrono> // [FIX] C++

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
  
  // Momentum
  for(int i=0; i<d; i++) r[i] = R::rnorm(0, sqrt(M));
  
  double noise_std = sqrt(2.0 * C * epsilon);
  NumericVector grad(d);
  double scale = (double)N / batch_size;
  
  IntegerVector all_indices = seq(0, N - 1);
  
  // [FIX] C++ time
  auto start_time = std::chrono::high_resolution_clock::now();
  
  for(int t = 0; t < n_iter; t++) {
    
    // (A) Theta Update
    for(int j = 0; j < d; j++) theta[j] += epsilon * (r[j] / M);
    
    // (B) Gradient Update (Mini-batch)
    // 1. Prior Gradient (Ridge)
    for(int j = 0; j < d; j++) grad[j] = lambda * theta[j];
    
    // 2. Likelihood Gradient (Data)
    IntegerVector indices = sample(all_indices, batch_size);
    
    for(int i = 0; i < batch_size; i++) {
      int idx = indices[i]; 
      
      double pred = 0;
      for(int j = 0; j < d; j++) pred += X(idx, j) * theta[j];
      
      double residual = y[idx] - pred;
      for(int j = 0; j < d; j++) grad[j] += scale * (-residual * X(idx, j));
    }
    
    // (C) Momentum Update
    for(int j = 0; j < d; j++) {
      // [FIX] scalar
      double noise = R::rnorm(0, noise_std);
      r[j] = r[j] - epsilon * grad[j] - epsilon * C * (r[j] / M) + noise;
    }
    
    samples(t, _) = theta;
    
    // [FIX] (seconds)
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = current_time - start_time;
    times[t] = elapsed.count();
  }
  
  return List::create(Named("Time") = times, Named("Theta") = samples);
}
')







# -------------------------------------------------------------------------
# 2. R SGHMC
# -------------------------------------------------------------------------
sghmc_r_timed <- function(X, y, theta_init, n_iter, epsilon, C, lambda, batch_size, M = 1) {
  d <- length(theta_init)
  N <- nrow(X)
  
  samples <- matrix(0, nrow = n_iter, ncol = d)
  times <- numeric(n_iter)
  
  theta <- theta_init
  r <- rnorm(d, 0, sqrt(M))
  noise_std <- sqrt(2 * C * epsilon)
  
  start_time <- Sys.time()
  
  for (t in 1:n_iter) {
    theta <- theta + epsilon * (r / M)
    
    indices <- sample(1:N, batch_size)
    X_batch <- X[indices, ]
    y_batch <- y[indices]
    residual <- y_batch - X_batch %*% theta
    grad <- as.vector((N / batch_size) * (-t(X_batch) %*% residual) + lambda * theta)
    
    noise <- rnorm(d, 0, noise_std)
    r <- r - epsilon * grad - epsilon * C * (r / M) + noise
    
    samples[t, ] <- theta
    times[t] <- as.numeric(Sys.time() - start_time)
  }
  
  return(list(Time = times, Theta = samples))
}

# -------------------------------------------------------------------------
# 3. (d > N High-Dim Setting)
# -------------------------------------------------------------------------
#N <- 500      
#d <- 1000
N <- 4000      
d <- 7000
batch_size <- 50 

set.seed(42)
true_beta <- runif(d, -0.2, 0.2) 
X <- matrix(rnorm(N * d), nrow = N, ncol = d)
y <- as.vector(X %*% true_beta + rnorm(N, 0, 1))
lambda <- 2.0  

cat("Calculating Ground Truth...\n")
A_matrix <- diag(lambda, d) + t(X) %*% X
Sigma_post <- solve(A_matrix)
mu_post <- as.vector(Sigma_post %*% t(X) %*% y)

# -------------------------------------------------------------------------
# 4. simulation
# -------------------------------------------------------------------------
n_iter <- 2000 
epsilon <- 1e-6
C <- 50
theta_init <- rep(0, d)

cat("Running R Version...\n")
res_r <- sghmc_r_timed(X, y, theta_init, n_iter, epsilon, C, lambda, batch_size)

cat("Running Rcpp Version...\n")
# M=1.0 
res_cpp <- sghmc_cpp_timed(X, y, theta_init, n_iter, epsilon, C, 1.0, lambda, batch_size)

get_error_df <- function(res_obj, method_name, true_mean) {
  samples <- res_obj$Theta
  times <- res_obj$Time
  
  check_idx <- seq(10, nrow(samples), by = 10)
  
  errors <- numeric(length(check_idx))
  time_points <- numeric(length(check_idx))
  
  for(i in seq_along(check_idx)) {
    idx <- check_idx[i]
    est_mean <- colMeans(samples[1:idx, , drop=FALSE])
    errors[i] <- sum((est_mean - true_mean)^2)
    time_points[i] <- times[idx]
  }
  
  return(data.frame(Iter = check_idx, Time = time_points, Error = errors, Method = method_name))
}

theta_cpp_mat <- as.matrix(res_cpp$Theta)
time_cpp_vec <- res_cpp$Time

res_cpp_clean <- list(Theta = theta_cpp_mat, Time = time_cpp_vec)

df_r <- get_error_df(res_r, "R Version", mu_post)
df_cpp <- get_error_df(res_cpp_clean, "Rcpp Version", mu_post)

df_total <- rbind(df_r, df_cpp)

# -------------------------------------------------------------------------
# 5. visualization
# -------------------------------------------------------------------------
p1 <- ggplot(df_total, aes(x = Iter, y = Error, color = Method)) +
  geom_line(linewidth = 1) +
  labs(title = "1. Accuracy per Iteration (Fair Comparison)",
       subtitle = "With 'sample()' (no-replacement) in both, curves are similar",
       y = "Mean Squared Error") +
  theme_minimal() + theme(legend.position = "none")

p2 <- ggplot(df_total, aes(x = Time, y = Error, color = Method)) +
  geom_line(linewidth = 1) +
  scale_x_log10() + 
  labs(title = "2. Accuracy per Wall-clock Time (The Real Winner)",
       subtitle = "Rcpp reaches the target error MUCH faster",
       x = "Time (Seconds, Log Scale)",
       y = "Mean Squared Error") +
  theme_minimal() + theme(legend.position = "bottom")

#grid.arrange(p1, p2, ncol = 1)
print(p1)
print(p2)
