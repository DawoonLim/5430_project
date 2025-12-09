# -------------------------------------------------------------------
# BOHAMIANN: Bayesian Optimization with Robust Bayesian Neural Networks
# Scale-Adapted SGHMC 
# -------------------------------------------------------------------

tanh_act <- function(x) {
  return(tanh(x))
}

dtanh_act <- function(x) {
  return(1 - tanh(x)^2)
}

init_network <- function(n_in, n_hidden, scale = 0.1) {
  W1 <- matrix(rnorm(n_in * n_hidden, sd = 1/sqrt(n_in)), nrow = n_in, ncol = n_hidden)
  b1 <- runif(n_hidden, -0.1, 0.1) 
  
  W2 <- matrix(rnorm(n_hidden * 1, sd = 1/sqrt(n_hidden)), nrow = n_hidden, ncol = 1)
  b2 <- runif(1, -0.1, 0.1)
  
  params <- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)
  return(params)
}

# 3. Forward Pass
forward <- function(X, params) {
  z1 <- X %*% params$W1 + matrix(params$b1, nrow = nrow(X), ncol = length(params$b1), byrow = TRUE)
  a1 <- tanh_act(z1)
  
  # Layer 2 (Output)
  z2 <- a1 %*% params$W2 + params$b2
  
  return(list(output = z2, a1 = a1)) 
}

# 4. Gradients 
# Negative Log Posterior
get_gradients <- function(X, y, params, lambda_prior = 1.0, noise_prec = 10.0) {
  N <- nrow(X)
  fwd <- forward(X, params)
  y_pred <- fwd$output
  a1 <- fwd$a1
  
  # --- Likelihood Gradient ---
  # Gaussian Likelihood 가정: -log P(D|w) ~ (precision/2) * sum((y - pred)^2)
  # d(Loss)/d(pred) = -precision * (y - pred)
  grad_loss_output <- -noise_prec * (y - y_pred) 
  
  # --- Backpropagation ---
  # Output Layer Gradients
  # d(Loss)/dW2 = a1.T * grad_loss_output
  dW2_lik <- t(a1) %*% grad_loss_output
  db2_lik <- sum(grad_loss_output)
  
  # Hidden Layer Gradients
  # d(Loss)/da1 = grad_loss_output * W2.T
  grad_loss_a1 <- grad_loss_output %*% t(params$W2)
  # d(Loss)/dz1 = grad_loss_a1 * dtanh(z1) ... 여기서 a1이 이미 tanh(z1)임
  grad_loss_z1 <- grad_loss_a1 * (1 - a1^2)
  
  # d(Loss)/dW1 = X.T * grad_loss_z1
  dW1_lik <- t(X) %*% grad_loss_z1
  db1_lik <- colSums(grad_loss_z1)
  
  # --- Prior Gradient (Weight Decay) ---
  # Gaussian Prior: -log P(w) ~ (lambda/2) * w^2
  # grad = lambda * w
  
  
  grads <- list(
    W1 = -dW1_lik + lambda_prior * params$W1, 
    b1 = -db1_lik + lambda_prior * params$b1,
    W2 = -dW2_lik + lambda_prior * params$W2,
    b2 = -db2_lik + params$b2 
  )
  
  return(grads)
}

# 5. BOHAMIANN (Scale-Adapted SGHMC )
bohamiann_train <- function(X, y, n_hidden = 50, num_iters = 2000, 
                            keep_every = 50, epsilon = 0.01, mdecay = 0.05) {
  
  n_in <- ncol(X)
  params <- init_network(n_in, n_hidden)
  
  # SGHMC
  v <- list(W1 = params$W1 * 0, b1 = params$b1 * 0, W2 = params$W2 * 0, b2 = params$b2 * 0)
  g_sq <- list(W1 = params$W1 * 0 + 1e-8, b1 = params$b1 * 0 + 1e-8, 
               W2 = params$W2 * 0 + 1e-8, b2 = params$b2 * 0 + 1e-8)
  
  samples <- list() 
  
  
  for (i in 1:num_iters) {
    grads <- get_gradients(X, y, params)
    
  
    for (p_name in names(params)) {
      g <- grads[[p_name]]
      
     
      g_sq[[p_name]] <- (1 - mdecay) * g_sq[[p_name]] + mdecay * g^2
      
     
      inv_M_sqrt <- 1 / sqrt(g_sq[[p_name]])
      
      noise_std <- sqrt(2 * epsilon * mdecay) * inv_M_sqrt 
      noise <- matrix(rnorm(length(g), 0, 1), nrow=nrow(g), ncol=ncol(g)) * noise_std
      
      # Update Velocity
      # v_{t+1} = (1 - mdecay) * v_t - epsilon * inv_M_sqrt * g + noise
      v[[p_name]] <- (1 - mdecay) * v[[p_name]] - (epsilon * inv_M_sqrt * g) + noise
      
      # (C) Parameter 
      params[[p_name]] <- params[[p_name]] + v[[p_name]]
    }
    
    # Sampling: Burn-in 
    if (i > (num_iters / 2) && i %% keep_every == 0) {
      samples[[length(samples) + 1]] <- params
    }
    
    if (i %% 200 == 0) {
      cat(sprintf("Iteration %d/%d finish\n", i, num_iters))
    }
  }
  
  return(samples)
}

# 6. (Bayesian Model Averaging)
predict_bohamiann <- function(X_test, samples) {
  n_samples <- length(samples)
  preds <- matrix(0, nrow = nrow(X_test), ncol = n_samples)
  
  for (i in 1:n_samples) {
    out <- forward(X_test, samples[[i]])$output
    preds[, i] <- out
  }
  
  mean_pred <- rowMeans(preds)
  var_pred <- apply(preds, 1, var)
  

  total_var <- var_pred + (1/10.0) 
  
  return(list(mean = mean_pred, var = total_var, all_preds = preds))
}

# -------------------------------------------------------------------
# example
# -------------------------------------------------------------------

set.seed(42)
X_train <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
y_train <- sin(X_train) + rnorm(20, 0, 0.1)
X_test <- matrix(seq(-4, 4, length.out = 100), ncol = 1)

samples <- bohamiann_train(X_train, y_train, n_hidden = 20, num_iters = 3000, epsilon = 0.005)

result <- predict_bohamiann(X_test, samples)

plot(X_test, result$mean, type = "l", col = "blue", lwd = 2, ylim = c(-2, 2),
     main = "BOHAMIANN (R Implementation)", xlab = "X", ylab = "y")

lines(X_test, result$mean + 2*sqrt(result$var), col = "lightblue", lty = 2)
lines(X_test, result$mean - 2*sqrt(result$var), col = "lightblue", lty = 2)

points(X_train, y_train, col = "red", pch = 19)
legend("topright", legend = c("Mean Pred", "Uncertainty", "Data"),
       col = c("blue", "lightblue", "red"), lty = c(1, 2, 0), pch = c(NA, NA, 19))




















# -------------------------------------------------------------------
# revised for linear regression 
# -------------------------------------------------------------------
sghmc_adaptive_r <- function(X, y, theta_init, n_iter, epsilon, C, lambda, batch_size, mdecay = 0.05) {
  N <- nrow(X)
  d <- length(theta_init)
  
  samples <- matrix(0, nrow = n_iter, ncol = d)
  times <- numeric(n_iter)
  
  theta <- theta_init
  
  # Momentum (v) 
  v <- rnorm(d, 0, sqrt(epsilon)) 
  
  # (Preconditioner)
  g_sq <- rep(1e-8, d)
  
  scale <- N / batch_size
  
  start_time <- Sys.time()
  
  for (t in 1:n_iter) {
    
    # 1. Stochastic Gradient Calculation
    indices <- sample(1:N, batch_size)
    X_batch <- X[indices, , drop=FALSE]
    y_batch <- y[indices]
    
    # Prediction & Residual
    pred <- X_batch %*% theta
    residual <- y_batch - pred
    
    # Gradient: -Likelihood_grad + Prior_grad
    grad <- as.vector(scale * (-t(X_batch) %*% residual) + lambda * theta)
    
    # 2. Adaptive Mass Update (RMSProp style)

    g_sq <- (1 - mdecay) * g_sq + mdecay * (grad^2)
    
    # Preconditioner: M^(-1/2) corresponds to 1 / sqrt(V_hat)
    sigma_inv <- 1 / sqrt(g_sq)
    
    # 3. Momentum & Position Update (Scale-Adapted)

    
    noise_std <- sqrt(2 * epsilon * C) * sqrt(sigma_inv) # M^(-1/2) scaling roughly
    noise <- rnorm(d, 0, 1) * noise_std
    
   
    v <- v - (epsilon * grad * sigma_inv) - (epsilon * C * v * sigma_inv) + noise
    
    # theta update
    theta <- theta + v
    
    samples[t, ] <- theta
    times[t] <- as.numeric(Sys.time() - start_time)
  }
  
  return(list(Time = times, Theta = samples))
}
