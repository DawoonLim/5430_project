# p= (50×10)+10+(10×1)+1=521
# n = 400
library(hmclearn)
library(MASS)
library(ggplot2)
library(RcppArmadillo)
library(Rcpp)

# Rcpp 
sourceCpp(code = '
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// =============================================================================
// [Helper] Gradient Calculation (Backpropagation)
// =============================================================================
vec get_bnn_grad_cpp(vec theta, mat X_batch, vec y_batch, double lambda, 
                     double scale, int d_in, int d_hidden, int d_out) {
    
    // 1. Unpack Parameters
    int idx1 = d_in * d_hidden;
    int idx2 = idx1 + d_hidden;
    int idx3 = idx2 + (d_hidden * d_out);
    
    mat W1(theta.begin(), d_in, d_hidden, false); 
    vec b1 = theta.subvec(idx1, idx2 - 1);
    mat W2(theta.begin() + idx2, d_hidden, d_out, false);
    vec b2 = theta.subvec(idx3, theta.n_elem - 1);
    
    // 2. Forward
    mat Z1 = X_batch * W1;
    Z1.each_row() += b1.t(); 
    mat A1 = tanh(Z1);
    
    mat Z2 = A1 * W2;
    Z2.each_row() += b2.t();
    vec y_pred = Z2; 
    
    // 3. Backward
    vec residual = y_batch - y_pred;
    
    mat grad_W2 = A1.t() * (-residual);
    double grad_b2 = sum(-residual);
    mat d_A1 = (-residual) * W2.t();
    mat d_Z1 = d_A1 % (1 - square(A1));
    mat grad_W1 = X_batch.t() * d_Z1;
    vec grad_b1 = sum(d_Z1, 0).t(); 
    
    // 4. Prior & Scaling
    vec g_W1 = scale * vectorise(grad_W1) + lambda * vectorise(W1);
    vec g_b1 = scale * grad_b1 + lambda * b1;
    vec g_W2 = scale * vectorise(grad_W2) + lambda * vectorise(W2);
    vec g_b2 = scale * grad_b2 + lambda * b2; 
    
    vec total_grad = join_cols(g_W1, g_b1);
    total_grad = join_cols(total_grad, g_W2);
    total_grad = join_cols(total_grad, g_b2);
    
    return total_grad;
}

// =============================================================================
// [Algorithm 1] Basic SGHMC
// =============================================================================
// [[Rcpp::export]]
mat sghmc_basic_cpp(mat X, vec y, vec theta_init, int n_iter, 
                    double epsilon, double C, double lambda, int batch_size,
                    int d_in, int d_hidden, int d_out) {
                        
    int N = X.n_rows;
    int d = theta_init.n_elem;
    mat samples(n_iter, d);
    vec theta = theta_init;
    vec r = randn(d) * sqrt(epsilon);
    
    double noise_std = sqrt(2.0 * epsilon * C);
    double scale = (double)N / batch_size;
    
    for(int t = 0; t < n_iter; t++) {
        uvec indices = randi<uvec>(batch_size, distr_param(0, N-1));
        vec grad = get_bnn_grad_cpp(theta, X.rows(indices), y.elem(indices), lambda, scale, d_in, d_hidden, d_out);
        
        vec noise = randn(d) * noise_std;
        r = r - epsilon * grad - epsilon * C * r + noise;
        theta = theta + r;
        samples.row(t) = theta.t();
    }
    return samples;
}

// =============================================================================
// [Algorithm 2] Splitting SGHMC
// =============================================================================
// [[Rcpp::export]]
mat sghmc_splitting_cpp(mat X, vec y, vec theta_init, int n_iter, 
                        double epsilon, double C, double lambda, int batch_size,
                        int d_in, int d_hidden, int d_out) {

    int N = X.n_rows;
    int d = theta_init.n_elem;
    mat samples(n_iter, d);
    
    vec theta = theta_init;
    vec r = randn(d); // Standard Normal initialization for Splitting
    
    double decay = exp(-C * epsilon);
    double noise_std = sqrt(1.0 - exp(-2.0 * C * epsilon));
    double scale = (double)N / batch_size;
    
    // Initial Gradient
    uvec indices = randi<uvec>(batch_size, distr_param(0, N-1));
    vec grad = get_bnn_grad_cpp(theta, X.rows(indices), y.elem(indices), lambda, scale, d_in, d_hidden, d_out);
    
    for(int t = 0; t < n_iter; t++) {
        // 1. Half step for r
        r = r - 0.5 * epsilon * grad;
        
        // 2. Half step for theta
        theta = theta + 0.5 * epsilon * r;
        
        // 3. Friction & Noise (Exact Ornstein-Uhlenbeck solution)
        vec noise = randn(d);
        r = r * decay + noise_std * noise;
        
        // 4. Half step for theta
        theta = theta + 0.5 * epsilon * r;
        
        // 5. Re-evaluate Gradient (New Batch)
        // R implementation samples a new batch here
        indices = randi<uvec>(batch_size, distr_param(0, N-1));
        grad = get_bnn_grad_cpp(theta, X.rows(indices), y.elem(indices), lambda, scale, d_in, d_hidden, d_out);
        
        // 6. Half step for r
        r = r - 0.5 * epsilon * grad;
        
        samples.row(t) = theta.t();
    }
    return samples;
}

// =============================================================================
// [Algorithm 3] Adaptive SGHMC (RMSprop Preconditioning)
// =============================================================================
// [[Rcpp::export]]
mat sghmc_adaptive_cpp(mat X, vec y, vec theta_init, int n_iter, 
                       double epsilon, double C, double lambda, int batch_size,
                       double mdecay, int d_in, int d_hidden, int d_out) {
                       
    int N = X.n_rows;
    int d = theta_init.n_elem;
    mat samples(n_iter, d);
    
    vec theta = theta_init;
    vec v = randn(d) * sqrt(epsilon);
    vec g_sq = ones(d) * 1e-5; // Initialize
    double scale = (double)N / batch_size;
    
    for(int t = 0; t < n_iter; t++) {
        uvec indices = randi<uvec>(batch_size, distr_param(0, N-1));
        vec grad = get_bnn_grad_cpp(theta, X.rows(indices), y.elem(indices), lambda, scale, d_in, d_hidden, d_out);
        
        // Update preconditioner (Element-wise)
        g_sq = (1 - mdecay) * g_sq + mdecay * square(grad);
        vec sigma_inv = 1.0 / sqrt(g_sq + 1e-10);
        
        // Noise scaling with preconditioner
        double noise_base = sqrt(2.0 * epsilon * C);
        vec noise = randn(d) % (noise_base * sqrt(sigma_inv));
        
        // Update v (Momentum)
        // v = v - eps * grad * sigma_inv - eps * C * v * sigma_inv + noise
        v = v - (epsilon * grad % sigma_inv) - (epsilon * C * v % sigma_inv) + noise;
        
        theta = theta + v;
        samples.row(t) = theta.t();
    }
    return samples;
}

// =============================================================================
// [Algorithm 4] Cyclical SGHMC
// =============================================================================
// [[Rcpp::export]]
mat sghmc_cyclical_cpp(mat X, vec y, vec theta_init, int n_iter, 
                       double epsilon_max, int cycle_length, int batch_size,
                       double beta, double lambda, int d_in, int d_hidden, int d_out) {
    
    int N = X.n_rows;
    int d = theta_init.n_elem;
    mat samples(n_iter, d);
    
    vec theta = theta_init;
    vec m = zeros(d);
    double scale = (double)N / batch_size;
    
    for(int t = 0; t < n_iter; t++) {
        // Cyclical Learning Rate Schedule
        double r_cyc = (double)((t+1) % cycle_length) / cycle_length;
        double epsilon_t = epsilon_max * (cos(M_PI * r_cyc) + 1.0) / 2.0;
        
        uvec indices = randi<uvec>(batch_size, distr_param(0, N-1));
        vec grad = get_bnn_grad_cpp(theta, X.rows(indices), y.elem(indices), lambda, scale, d_in, d_hidden, d_out);
        
        double noise_scale = sqrt(2.0 * epsilon_t * (1.0 - beta));
        vec noise = randn(d) * noise_scale;
        
        // Momentum update with friction built into beta
        m = beta * m - epsilon_t * grad + noise;
        theta = theta + m;
        
        samples.row(t) = theta.t();
    }
    return samples;
}
')

# ==============================================================================
# 3. Generating the data
# ==============================================================================
d_in <- 50; d_hidden <- 10; d_out <- 1
total_params <- (d_in * d_hidden) + d_hidden + (d_hidden * d_out) + d_out

set.seed(10)
N <- 400 
X <- matrix(rnorm(N * d_in), nrow = N, ncol = d_in)
true_theta <- rnorm(total_params, 0, 1) / sqrt(d_in)

unpack_theta <- function(theta, d_in, d_hidden, d_out) {
  idx1 <- d_in * d_hidden
  W1 <- matrix(theta[1:idx1], nrow = d_in, ncol = d_hidden)
  idx2 <- idx1 + d_hidden; b1 <- theta[(idx1 + 1):idx2]
  idx3 <- idx2 + (d_hidden * d_out); W2 <- matrix(theta[(idx2 + 1):idx3], nrow = d_hidden, ncol = d_out)
  idx4 <- idx3 + d_out; b2 <- theta[(idx3 + 1):idx4]
  list(W1=W1, b1=b1, W2=W2, b2=b2)
}

p_true <- unpack_theta(true_theta, d_in, d_hidden, d_out)
y <- as.vector(sweep(tanh(sweep(X %*% p_true$W1, 2, p_true$b1, "+")) %*% p_true$W2, 2, p_true$b2, "+") + rnorm(N, 0, 0.5))

theta_init <- rnorm(total_params, 0, 0.1)
n_iter <- 10000
batch_size <- 40 

# (1) Gradient (Backpropagation)
get_bnn_grad <- function(theta, X_batch, y_batch, lambda, N_total, batch_size) {
  params <- unpack_theta(theta, d_in, d_hidden, d_out)
  W1 <- params$W1; b1 <- params$b1; W2 <- params$W2; b2 <- params$b2
  
  # Forward
  Z1 <- sweep(X_batch %*% W1, 2, b1, "+")
  A1 <- tanh(Z1)
  y_pred <- sweep(A1 %*% W2, 2, b2, "+")
  
  # Backward
  scale <- N_total / batch_size
  residual <- y_batch - y_pred
  
  grad_W2 <- t(A1) %*% (-residual)
  grad_b2 <- sum(-residual)
  d_A1 <- (-residual) %*% t(W2)
  d_Z1 <- d_A1 * (1 - A1^2)
  grad_W1 <- t(X_batch) %*% d_Z1
  grad_b1 <- colSums(d_Z1)
  
  # Gradient of U (Potential Energy) = -LogPosterior Gradient
  # We want gradient of -logP, so signs are flipped compared to ascent
  g_W1 <- scale * grad_W1 + lambda * W1
  g_b1 <- scale * grad_b1 + lambda * b1
  g_W2 <- scale * grad_W2 + lambda * W2
  g_b2 <- scale * grad_b2 + lambda * b2
  
  c(as.vector(g_W1), as.vector(g_b1), as.vector(g_W2), as.vector(g_b2))
}

# (2) Potential Energy U(theta)
get_potential_energy <- function(theta, X, y, lambda) {
  params <- unpack_theta(theta, d_in, d_hidden, d_out)
  Z1 <- tanh(sweep(X %*% params$W1, 2, params$b1, "+"))
  y_pred <- sweep(Z1 %*% params$W2, 2, params$b2, "+")
  
  # Negative Log Likelihood (Gaussian errors)
  nll <- 0.5 * sum((y - y_pred)^2)
  # Negative Log Prior (Gaussian prior)
  nlp <- 0.5 * lambda * sum(theta^2)
  
  return(nll + nlp)
}


# ==============================================================================
# 4. Simulation
# ==============================================================================
# logPOSTERIOR 
# get_potential_energy (-)
# (1) logPOSTERIOR: 
bnn_log_posterior <- function(theta, X, y, lambda, ...) {
  # param 리스트를 푸는 과정 삭제 -> 인자로 바로 들어옴
  
  # Forward Pass
  # 
  params <- unpack_theta(theta, d_in, d_hidden, d_out)
  Z1 <- tanh(sweep(X %*% params$W1, 2, params$b1, "+"))
  y_pred <- sweep(Z1 %*% params$W2, 2, params$b2, "+")
  
  # Log Likelihood + Log Prior
  log_lik <- -0.5 * sum((y - y_pred)^2)
  log_prior <- -0.5 * lambda * sum(theta^2)
  
  return(log_lik + log_prior)
}

# (2) glogPOSTERIOR: 
bnn_g_log_posterior <- function(theta, X, y, lambda, ...) {
  # HMC는 Full Batch이므로 N_total과 batch_size는 전체 데이터 개수
  N_total <- nrow(X)
  
  

  grad_U <- get_bnn_grad(theta, X, y, lambda, N_total, batch_size=N_total)
  
  return(-grad_U) 
}

t0 <- Sys.time()
res_hmclearn <- hmclearn::hmc(
  N = n_iter,
  theta.init = theta_init,
  epsilon = 0.002,   
  L = 10,            
  logPOSTERIOR = bnn_log_posterior,
  glogPOSTERIOR = bnn_g_log_posterior,
  
  # 
  X = X, 
  y = y, 
  lambda = 1,
  
  parallel = FALSE,
  chains = 1
)

time_hmc <- Sys.time() - t0
res_hmc <- do.call(rbind, res_hmclearn$thetaCombined) 


# burn-in 
res_hmc_samples <- do.call(rbind, res_hmclearn$thetaCombined)

t1 <- Sys.time()
res_basic <- sghmc_basic_cpp(X, y, theta_init, n_iter, epsilon=0.0005, C=50, lambda=1, batch_size=batch_size, d_in, d_hidden, d_out)
time_basic <- Sys.time() - t1

t2 <- Sys.time()
res_split <- sghmc_splitting_cpp(X, y, theta_init, n_iter, epsilon=0.0005, C=50, lambda=1, batch_size=batch_size, d_in, d_hidden, d_out)
time_split <- Sys.time() - t2

t3 <- Sys.time()
res_adapt <- sghmc_adaptive_cpp(X, y, theta_init, n_iter, epsilon=0.005, C=5, lambda=1, batch_size=batch_size, mdecay=0.05, d_in, d_hidden, d_out)
time_adapt <- Sys.time() - t3

t4 <- Sys.time()
res_cyc <- sghmc_cyclical_cpp(X, y, theta_init, n_iter, epsilon_max=0.0001, cycle_length=200, batch_size=batch_size, beta=0.9, lambda=1, d_in, d_hidden, d_out)
time_cyc <- Sys.time() - t4

# ==============================================================================
# 5. Result
# ==============================================================================
predict_bnn <- function(samp, X_new, burn=2000) {
  preds <- matrix(0, nrow=nrow(samp)-burn, ncol=nrow(X_new))
  idx <- 1
  for(i in (burn+1):nrow(samp)) {
    p <- unpack_theta(samp[i,], d_in, d_hidden, d_out)
    preds[idx,] <- sweep(tanh(sweep(X_new %*% p$W1, 2, p$b1, "+")) %*% p$W2, 2, p$b2, "+")
    idx <- idx+1
  }
  colMeans(preds)
}


res_list <- list(HMC=res_hmc, sghmc=res_basic, Split=res_split, Adapt=res_adapt, Cycle=res_cyc)
times <- c(time_hmc, time_basic, time_split, time_adapt, time_cyc)
algo_names <- c("HMC", "SGHMC", "Splitting SGHMC", "BOHAMIANN", "cSGHMC")

for(i in 1:5) {
  mse <- mean((y - predict_bnn(res_list[[i]], X))^2)
  cat(sprintf("%-15s | Time: %6.2f s | MSE: %.5f\n", algo_names[i], as.numeric(times[i]), mse))
}

# Traceplot of 1st to 5th Parameter
par(mfrow=c(5,1), mar=c(2,4,2,1))
for(i in 1:5) {
  plot(res_list[[i]][,1], type='l', main=algo_names[i], ylab="Theta[1]", col=i)
  abline(h=true_theta[1], col='red', lty=2)
} 

par(mfrow=c(5,1), mar=c(2,4,2,1))
for(i in 1:5) {
  plot(res_list[[i]][,2], type='l', main=algo_names[i], ylab="Theta[2]", col=i)
  abline(h=true_theta[2], col='red', lty=2)
} 

par(mfrow=c(5,1), mar=c(2,4,2,1))
for(i in 1:5) {
  plot(res_list[[i]][,3], type='l', main=algo_names[i], ylab="Theta[3]", col=i)
  abline(h=true_theta[3], col='red', lty=2)
} 

par(mfrow=c(5,1), mar=c(2,4,2,1))
for(i in 1:5) {
  plot(res_list[[i]][,4], type='l', main=algo_names[i], ylab="Theta[4]", col=i)
  abline(h=true_theta[4], col='red', lty=2)
} 

par(mfrow=c(5,1), mar=c(2,4,2,1))
for(i in 1:5) {
  plot(res_list[[i]][,5], type='l', main=algo_names[i], ylab="Theta[5]", col=i)
  abline(h=true_theta[5], col='red', lty=2)
} 




# 1. prediction
get_pred_interval <- function(samp, X_data, burn=2000) {
  # 메모리 절약을 위해 샘플링 (예: 200개만 사용)
  use_idx <- seq(burn+1, nrow(samp), length.out=200)
  preds <- matrix(0, nrow=length(use_idx), ncol=nrow(X_data))
  
  cnt <- 1
  for(i in use_idx) {
    p <- unpack_theta(samp[i,], d_in, d_hidden, d_out)
    preds[cnt,] <- sweep(tanh(sweep(X_data %*% p$W1, 2, p$b1, "+")) %*% p$W2, 2, p$b2, "+")
    cnt <- cnt + 1
  }
  
  pred_mean <- colMeans(preds)
  pred_lower <- apply(preds, 2, quantile, probs=0.025)
  pred_upper <- apply(preds, 2, quantile, probs=0.975)
  
  return(data.frame(Observed=y, Predicted=pred_mean, Lower=pred_lower, Upper=pred_upper))
}

df_pred_hmc <- get_pred_interval(res_hmc, X, burn=2000)
df_pred_sghmc <- get_pred_interval(res_basic, X, burn=2000)
df_pred_split <- get_pred_interval(res_split, X, burn=2000)
df_pred_adaptive <- get_pred_interval(res_adapt, X, burn=2000)
df_pred_csghmc <- get_pred_interval(res_cyc, X, burn=2000)

set.seed(10)
sample_idx <- sample(1:N, 50)
df_subset_hmc <- df_pred_hmc[sample_idx, ]
df_subset_sghmc <- df_pred_sghmc[sample_idx, ]
df_subset_split <- df_pred_split[sample_idx, ]
df_subset_adaptive <- df_pred_adaptive[sample_idx, ]
df_subset_csghmc <- df_pred_csghmc[sample_idx, ]

ggplot(df_subset_hmc, aes(x=Observed, y=Predicted)) +
  geom_point(color="blue") +
  geom_errorbar(aes(ymin=Lower, ymax=Upper), width=0.1, color="gray") +
  geom_abline(intercept=0, slope=1, color="red", linetype="dashed") +
  #labs(title="Observed vs Predicted (with 95% CI) - HMC",
  #     subtitle="Gray bars represent model uncertainty") +
  theme_bw()

ggplot(df_subset_sghmc, aes(x=Observed, y=Predicted)) +
  geom_point(color="blue") +
  geom_errorbar(aes(ymin=Lower, ymax=Upper), width=0.1, color="gray") +
  geom_abline(intercept=0, slope=1, color="red", linetype="dashed") +
  #labs(title="Observed vs Predicted (with 95% CI) - SGHMC",
  #     subtitle="Gray bars represent model uncertainty") +
  theme_bw()

ggplot(df_subset_split, aes(x=Observed, y=Predicted)) +
  geom_point(color="blue") +
  geom_errorbar(aes(ymin=Lower, ymax=Upper), width=0.1, color="gray") +
  geom_abline(intercept=0, slope=1, color="red", linetype="dashed") +
  #labs(title="Observed vs Predicted (with 95% CI) - Split SGHMC",
  #     subtitle="Gray bars represent model uncertainty") +
  theme_bw()

ggplot(df_pred_adaptive, aes(x=Observed, y=Predicted)) +
  geom_point(color="blue") +
  geom_errorbar(aes(ymin=Lower, ymax=Upper), width=0.1, color="gray") +
  geom_abline(intercept=0, slope=1, color="red", linetype="dashed") +
  #labs(title="Observed vs Predicted (with 95% CI) - adaptive SGHMC",
  #     subtitle="Gray bars represent model uncertainty") +
  theme_bw()

ggplot(df_pred_csghmc, aes(x=Observed, y=Predicted)) +
  geom_point(color="blue") +
  geom_errorbar(aes(ymin=Lower, ymax=Upper), width=0.1, color="gray") +
  geom_abline(intercept=0, slope=1, color="red", linetype="dashed") +
  #labs(title="Observed vs Predicted (with 95% CI) - cSGHMC",
  #     subtitle="Gray bars represent model uncertainty") +
  theme_bw()
