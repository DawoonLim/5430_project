sghmc_splitting_r <- function(X, y, theta_init, n_iter, epsilon, C, lambda, batch_size, M = 1) {
  d <- length(theta_init)
  N <- nrow(X)
  
  samples <- matrix(0, nrow = n_iter, ncol = d)
  times <- numeric(n_iter)
  
  theta <- theta_init
  r <- rnorm(d, 0, sqrt(M))
  
  # --- [사전 계산] ---
  # Exact integration parameters
  decay <- exp(-C * epsilon)
  noise_std <- sqrt(1 - exp(-2 * C * epsilon))
  
  # 초기 Gradient 계산
  indices <- sample(1:N, batch_size)
  X_batch <- X[indices, , drop=FALSE]
  y_batch <- y[indices]
  residual <- y_batch - X_batch %*% theta
  # Gradient: Negative Log Posterior (Likelihood + Prior)
  grad <- as.vector((N / batch_size) * (-t(X_batch) %*% residual) + lambda * theta)
  
  start_time <- Sys.time()
  
  for (t in 1:n_iter) {
    # 1. B-step (Half): Momentum update
    r <- r - 0.5 * epsilon * grad
    
    # 2. A-step (Half): Position update
    theta <- theta + 0.5 * epsilon * (r / M)
    
    # 3. O-step (Full): Friction & Noise (Exact Solution)
    noise <- rnorm(d, 0, 1)
    r <- r * decay + (sqrt(M) * noise_std) * noise
    
    # 4. A-step (Half): Position update
    theta <- theta + 0.5 * epsilon * (r / M)
    
    # 5. Gradient Update (New Position)
    indices <- sample(1:N, batch_size)
    X_batch <- X[indices, , drop=FALSE]
    y_batch <- y[indices]
    residual <- y_batch - X_batch %*% theta
    grad <- as.vector((N / batch_size) * (-t(X_batch) %*% residual) + lambda * theta)
    
    # 6. B-step (Half): Momentum update
    r <- r - 0.5 * epsilon * grad
    
    samples[t, ] <- theta
    times[t] <- as.numeric(Sys.time() - start_time)
  }
  
  return(list(Time = times, Theta = samples))
}