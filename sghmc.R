# SGHMC implementation in R (diagonal M, C, B_hat supported)
# sghmc with safe minibatch sampling (automatically handles batch_size > N)
sghmc <- function(grad_log_post, data, theta0,
                       n_iter = 5000, eps = 1e-3, m = 1,
                       batch_size = NULL, M = 1, C = 0.01, B_hat = 0,
                       resample_r = TRUE, burnin = 0, thin = 1,
                       store_momentum = FALSE, verbose = TRUE, ...) {
  
  # helper: detect number of observations N in 'data'
  detect_N <- function(data) {
    if (is.data.frame(data) || is.matrix(data)) {
      return(nrow(data))
    } else if (is.list(data)) {
      # common pattern: data = list(X = X, y = y)
      if (!is.null(data$X)) return(nrow(data$X))
      if (!is.null(data$y)) return(length(data$y))
      # fallback to length
      return(length(data))
    } else {
      return(length(data))
    }
  }
  
  N <- detect_N(data)
  
  tovec <- function(x, p) {
    if (length(x) == 1) rep(as.numeric(x), p) else as.numeric(x)
  }
  
  theta <- as.numeric(theta0)
  p <- length(theta)
  Mv <- tovec(M, p)
  Cv <- tovec(C, p)
  Bhv <- tovec(B_hat, p)
  
  if (any(Cv - Bhv < 0)) {
    warning("Some entries of C - B_hat are negative. Using pmax(C - B_hat, 0).")
  }
  noise_var_vec <- pmax(Cv - Bhv, 0) * 2 * eps
  
  nstore <- floor((n_iter - burnin) / thin)
  samples <- matrix(NA_real_, nrow = nstore, ncol = p)
  colnames(samples) <- paste0("theta", seq_len(p))
  if (store_momentum) moms <- matrix(NA_real_, nrow = nstore, ncol = p)
  
  # init momentum
  r <- rnorm(p, mean = 0, sd = sqrt(Mv))
  store_idx <- 0
  
  for (t in seq_len(n_iter)) {
    if (resample_r) r <- rnorm(p, 0, sqrt(Mv))
    
    # inner steps
    for (i in seq_len(m)) {
      theta <- theta + eps * (r / Mv)
      
      # pick minibatch indices safely
      if (!is.null(batch_size)) {
        if (N <= 0) stop("Cannot determine number of observations in 'data'.")
        if (batch_size > N) {
          # strategy: use full data (batch_size <- N) and warn
          warning(sprintf("Requested batch_size = %d > N = %d. Using full data (batch_size <- %d).",
                          batch_size, N, N))
          idx_batch <- seq_len(N)
        } else {
          idx_batch <- sample(seq_len(N), size = batch_size, replace = FALSE)
        }
        grad_logp <- grad_log_post(theta, data, idx = idx_batch, ...)
      } else {
        grad_logp <- grad_log_post(theta, data, idx = NULL, ...)
      }
      
      grad_U <- - as.numeric(grad_logp)
      noise_sd <- sqrt(noise_var_vec)
      noise <- rnorm(p, 0, noise_sd)
      r <- r - eps * grad_U - eps * (Cv * (r / Mv)) + noise
    }
    
    if (t > burnin && ((t - burnin) %% thin == 0)) {
      store_idx <- store_idx + 1
      samples[store_idx, ] <- theta
      if (store_momentum) moms[store_idx, ] <- r
    }
    
    if (verbose && (t %% max(1, floor(n_iter/10)) == 0)) {
      cat(sprintf("Iter %d / %d (stored %d)\n", t, n_iter, store_idx))
    }
  }
  
  res <- list(samples = samples, final_theta = theta, final_r = r, M = Mv, C = Cv, B_hat = Bhv)
  if (store_momentum) res$moments <- moms
  return(res)
}


# Simulate data
set.seed(1)
n <- 2000
p <- 5
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(0.5, -1, 0.3, 0, 0)
eta <- X %*% beta_true
y <- rbinom(n, size = 1, prob = 1 / (1 + exp(-eta)))
data_list <- list(X = X, y = y)

# gradient of log posterior (log-likelihood + log-prior)
grad_log_post_logistic <- function(theta, data, idx = NULL, sigma_prior = 10) {
  X <- data$X
  y <- data$y
  if (!is.null(idx)) {
    Xb <- X[idx, , drop = FALSE]
    yb <- y[idx]
  } else {
    Xb <- X
    yb <- y
  }
  eta <- Xb %*% theta
  pvec <- 1 / (1 + exp(-eta))
  grad_ll <- t(Xb) %*% (yb - pvec)  # gradient of log-likelihood
  # Gaussian prior N(0, sigma_prior^2 I) -> grad log prior = - theta / sigma^2
  grad_prior <- - theta / (sigma_prior^2)
  as.numeric(grad_ll + grad_prior)
}

# Run SGHMC
res <- sghmc(grad_log_post = grad_log_post_logistic,
             data = data_list,
             theta0 = rep(0, p),
             n_iter = 2000,
             eps = 5e-4,
             m = 1,
             batch_size = 200,    # mini-batch
             M = 1,
             C = 0.01,
             B_hat = 0,           # if you can estimate gradient noise, put it here
             burnin = 500,
             thin = 5,
             verbose = TRUE)

samples <- res$samples
apply(samples, 2, mean)    # posterior mean estimates (rough)
apply(samples, 2, sd)
