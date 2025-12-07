library(torch)

# 1. 모델 아키텍처 정의 (ResNet Encoder + MLP Heads)
# 논문에서는 ResNet-18과 2-layer MLP를 사용했습니다.

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
    
    # Target Network (Online과 동일 구조, 초기화 후 복사)
    self$target_encoder <- base_encoder$clone()
    self$target_projector <- get_mlp(encoder_out_dim, project_dim)
    
    # Target Network는 학습되지 않으므로 gradient 계산 중지
    for (p in self$target_encoder$parameters) p$requires_grad_(FALSE)
    for (p in self$target_projector$parameters) p$requires_grad_(FALSE)
  },
  
  forward = function(x1, x2) {
    # x1, x2는 각각 다르게 증강(Augmentation)된 이미지 배치
    
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

# 2. 손실 함수 (Normalized MSE / Cosine Similarity)
# 논문 식 (1): normalized predictions와 target projections 사이의 MSE
byol_loss_fn <- function(p, z) {
  p_norm <- nnf_normalize(p, dim = 2)
  z_norm <- nnf_normalize(z, dim = 2)
  return(2 - 2 * (p_norm * z_norm)$sum(dim = -1)$mean())
}

# 3. cSGHMC 업데이트 함수 (핵심 알고리즘)
# 논문 Algorithm 1 및 식 (7) 구현
# params: 업데이트할 파라미터 리스트
# momentum_buffer: 모멘텀 상태 저장 (m)
# lr: 현재 learning rate
# n_train: 전체 데이터셋 크기 (Training set size)
# batch_size: 현재 배치 크기
# beta: Momentum term
# temperature: Temperature scaling (T)
csghmc_update <- function(params, momentum_buffer, lr, n_train, batch_size, beta = 0.9, temperature = 1.0) {
  
  for (i in seq_along(params)) {
    p <- params[[i]]
    if (is.null(p$grad)) next
    
    # Gradient of Log Posterior approximation
    # grad_U = n/N * grad_likelihood + grad_prior
    # 논문에서는 Prior p(theta) ~ N(0, I)를 가정하므로 grad_prior는 p (weight decay와 유사)
    
    grad_likelihood <- p$grad 
    grad_prior <- p # Gaussian Prior N(0, I) -> grad(-log P) = theta
    
    # Unbiased estimate of Gradient U (논문 식 5, 6 관련)
    # n_train/batch_size 로 Likelihood 스케일링
    grad_U <- (n_train / batch_size) * grad_likelihood + grad_prior
    
    # Noise Sampling epsilon ~ N(0, I)
    epsilon <- torch_randn_like(p)
    
    # Momentum Update (논문 식 7)
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

# 4. Target Network EMA 업데이트 (식 4)
update_target_network <- function(online_net, target_net, tau) {
  online_params <- online_net$parameters
  target_params <- target_net$parameters
  
  for (i in seq_along(online_params)) {
    target_params[[i]]$data()$mul_(tau)$add_(online_params[[i]]$data(), alpha = 1 - tau)
  }
}

# 5. 전체 학습 루프 예시 함수
train_bayesian_byol <- function(dataloader, model, n_epochs, n_train_samples, T_cycle) {
  
  # Momentum 초기화 (0으로 시작)
  momentum_buffer <- lapply(model$online_encoder$parameters, function(p) torch_zeros_like(p))
  # Note: Projection/Prediction head도 베이지안 업데이트를 할지 논문에 명시되지 않았으나,
  # 보통 Encoder(Representation)에 집중하므로 여기선 Encoder만 예시로 듭니다.
  # 전체 파라미터에 적용하려면 리스트를 합치면 됩니다.
  
  iter_count <- 0
  
  for (epoch in 1:n_epochs) {
    
    # Cyclical Learning Rate Schedule (Cosine Annealing 등)
    # 여기서는 단순화를 위해 매 epoch마다 계산한다고 가정
    
    coro::loop(batch in dataloader, {
      iter_count <- iter_count + 1
      
      # Cyclic Learning Rate 계산 (논문의 C(k) 부분)
      # 예: Cosine Schedule
      lr <- 0.1 * (cos(pi * (iter_count %% T_cycle) / T_cycle) + 1) / 2
      
      # 데이터 로드 (aug1, aug2는 증강된 이미지)
      # torch dataloader 구조에 따라 다를 수 있음
      aug1 <- batch[[1]]$to(device = "cuda")
      aug2 <- batch[[2]]$to(device = "cuda")
      batch_size <- aug1$size(1)
      
      # 1. Forward Pass
      out <- model(aug1, aug2)
      
      # 2. Loss 계산 (Symmetrized Loss)
      loss <- byol_loss_fn(out$p1, out$z2_target) + byol_loss_fn(out$p2, out$z1_target)
      
      # 3. Backward Pass (Gradient 계산)
      model$zero_grad()
      loss$backward()
      
      # 4. cSGHMC Update (Posterior Sampling)
      # 인코더 파라미터에 대해 베이지안 업데이트 적용
      csghmc_update(
        params = model$online_encoder$parameters,
        momentum_buffer = momentum_buffer,
        lr = lr,
        n_train = n_train_samples,
        batch_size = batch_size,
        beta = 0.9
      )
      
      # 나머지 Head 부분은 SGD로 할지 cSGHMC로 할지 선택 필요 (보통 SGD 혹은 동일하게 적용)
      # 여기서는 편의상 동일하게 적용한다고 가정하거나, 별도 옵티마이저 사용 가능
      
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
  # epsilon_max: 초기(최대) 학습률
  # cycle_length: 한 사이클의 반복 횟수
  
  N <- nrow(X)
  d <- length(theta_init)
  
  samples <- matrix(0, nrow = n_iter, ncol = d)
  times <- numeric(n_iter)
  
  theta <- theta_init
  
  # Momentum 초기화 (보통 0으로 시작)
  m <- rep(0, d)
  
  scale <- N / batch_size
  start_time <- Sys.time()
  
  for (t in 1:n_iter) {
    
    # --- 1. Cyclical Step Size Schedule (Cosine Annealing) ---
    # 논문의 C(k) 구현: epsilon을 주기적으로 줄였다 키웠다 함
    # t가 cycle_length의 배수가 될 때마다 reset
    r <- (t %% cycle_length) / cycle_length
    epsilon_t <- epsilon_max * (cos(pi * r) + 1) / 2
    
    # --- 2. Gradient Calculation ---
    indices <- sample(1:N, batch_size)
    X_batch <- X[indices, , drop=FALSE]
    y_batch <- y[indices]
    
    pred <- X_batch %*% theta
    residual <- y_batch - pred
    
    # grad_U (Negative Log Posterior)
    # Prior는 Gaussian(0, I) 가정 -> lambda=1 로 가정 (코드 단순화)
    grad_U <- as.vector(scale * (-t(X_batch) %*% residual) + theta)
    
    # --- 3. Momentum Update (사용자 코드 식 7 기반) ---
    # m_new = beta * m_prev - (epsilon/2) * grad_U + Noise
    # Noise term: sqrt(T * (1 - beta) * epsilon) * N(0,I)
    # T(Temperature)는 1로 가정
    
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