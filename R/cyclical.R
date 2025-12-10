library(torch)


get_mlp <- function(dim, projection_size, hidden_size = 4096) {
  nn_sequential(
    nn_linear(dim, hidden_size),
    nn_batch_norm1d(hidden_size),
    nn_relu(),
    nn_linear(hidden_size, projection_size)
  )
}

BYOL_Module <- nn_module(
  "BYOL_Module",
  initialize = function(base_encoder, encoder_out_dim = 512, project_dim = 256) {
    # Online Network
    self$online_encoder <- base_encoder
    self$online_projector <- get_mlp(encoder_out_dim, project_dim)
    self$online_predictor <- get_mlp(project_dim, project_dim)
    
    # Target Network
    self$target_encoder <- base_encoder$clone()
    self$target_projector <- get_mlp(encoder_out_dim, project_dim)
    
    for (p in self$target_encoder$parameters) p$requires_grad_(FALSE)
    for (p in self$target_projector$parameters) p$requires_grad_(FALSE)
  },
  
  forward = function(x1, x2) {
    
    # Online Network Forward
    z1_online <- self$online_projector(self$online_encoder(x1))
    z2_online <- self$online_projector(self$online_encoder(x2))
    
    p1 <- self$online_predictor(z1_online)
    p2 <- self$online_predictor(z2_online)
    
    # Target Network Forward (no grad)
    with_no_grad({
      z1_target <- self$target_projector(self$target_encoder(x1))
      z2_target <- self$target_projector(self$target_encoder(x2))
    })
    
    list(p1 = p1, p2 = p2, z1_target = z1_target, z2_target = z2_target)
  }
)

# 2. (Normalized MSE / Cosine Similarity)
# (1): normalized predictions와 target projections MSE
byol_loss_fn <- function(p, z) {
  p_norm <- nnf_normalize(p, dim = 2)
  z_norm <- nnf_normalize(z, dim = 2)
  return(2 - 2 * (p_norm * z_norm)$sum(dim = -1)$mean())
}

# 3. cSGHMC 
csghmc_update <- function(params, momentum_buffer, lr, n_train, batch_size, beta = 0.9, temperature = 1.0) {
  
  for (i in seq_along(params)) {
    p <- params[[i]]
    if (is.null(p$grad)) next
    
    
    grad_likelihood <- p$grad 
    grad_prior <- p # Gaussian Prior N(0, I) -> grad(-log P) = theta
    
    grad_U <- (n_train / batch_size) * grad_likelihood + grad_prior
    
    # Noise Sampling epsilon ~ N(0, I)
    epsilon <- torch_randn_like(p)
    
    # Momentum Update
    # m = beta * m - (lr / 2) * grad_U + sqrt(T * (1 - beta) * lr) * epsilon
    m_prev <- momentum_buffer[[i]]
    noise_term <- sqrt(temperature * (1 - beta) * lr) * epsilon
    gradient_term <- (lr / 2) * grad_U
    
    m_new <- beta * m_prev - gradient_term + noise_term
    momentum_buffer[[i]] <- m_new
    
    # Parameter Update
    # theta = theta + m
    p$add_(m_new)
  }
}

# 4. Target Network EMA 
update_target_network <- function(online_net, target_net, tau) {
  online_params <- online_net$parameters
  target_params <- target_net$parameters
  
  for (i in seq_along(online_params)) {
    target_params[[i]]$data()$mul_(tau)$add_(online_params[[i]]$data(), alpha = 1 - tau)
  }
}

train_bayesian_byol <- function(dataloader, model, n_epochs, n_train_samples, T_cycle) {
  
  # Momentum 
  momentum_buffer <- lapply(model$online_encoder$parameters, function(p) torch_zeros_like(p))

  
  iter_count <- 0
  
  for (epoch in 1:n_epochs) {
    
    # Cyclical Learning Rate Schedule 
    
    coro::loop(batch in dataloader, {
      iter_count <- iter_count + 1
      
      # Cyclic Learning Rate 
      # Cosine Schedule
      lr <- 0.1 * (cos(pi * (iter_count %% T_cycle) / T_cycle) + 1) / 2
      
      aug1 <- batch[[1]]$to(device = "cuda")
      aug2 <- batch[[2]]$to(device = "cuda")
      batch_size <- aug1$size(1)
      
      # 1. Forward Pass
      out <- model(aug1, aug2)
      
      # 2.(Symmetrized Loss)
      loss <- byol_loss_fn(out$p1, out$z2_target) + byol_loss_fn(out$p2, out$z1_target)
      
      # 3. Backward Pass 
      model$zero_grad()
      loss$backward()
      
      # 4. cSGHMC Update (Posterior Sampling)
      csghmc_update(
        params = model$online_encoder$parameters,
        momentum_buffer = momentum_buffer,
        lr = lr,
        n_train = n_train_samples,
        batch_size = batch_size,
        beta = 0.9
      )
      
      # 5. Target Network EMA Update
      update_target_network(model$online_encoder, model$target_encoder, tau = 0.99)
      update_target_network(model$online_projector, model$target_projector, tau = 0.99)
      
      if (iter_count %% 100 == 0) {
        cat(sprintf("Epoch: %d, Iter: %d, Loss: %.4f, LR: %.5f\n", epoch, iter_count, loss$item(), lr))
      }
    })
  }
}























# -------------------------------------------------------------------
# revised for linear regression 
# -------------------------------------------------------------------
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
    
    r <- (t %% cycle_length) / cycle_length
    epsilon_t <- epsilon_max * (cos(pi * r) + 1) / 2
    
    # --- 2. Gradient Calculation ---
    indices <- sample(1:N, batch_size)
    X_batch <- X[indices, , drop=FALSE]
    y_batch <- y[indices]
    
    pred <- X_batch %*% theta
    residual <- y_batch - pred
    
    # grad_U (Negative Log Posterior)
    grad_U <- as.vector(scale * (-t(X_batch) %*% residual) + theta)
    
    # --- 3. Momentum Update ---
    # m_new = beta * m_prev - (epsilon/2) * grad_U + Noise
    # Noise term: sqrt(T * (1 - beta) * epsilon) * N(0,I)
    
    noise_scale <- sqrt(1.0 * (1 - beta) * epsilon_t)
    noise <- rnorm(d, 0, 1) * noise_scale
    
    gradient_term <- (epsilon_t / 2) * grad_U
    
    m <- beta * m - gradient_term + noise
    
    # --- 4. Theta Update ---
    theta <- theta + m
    
    samples[t, ] <- theta
    times[t] <- as.numeric(Sys.time() - start_time)
  }
  
  return(list(Time = times, Theta = samples))
}
