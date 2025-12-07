# -------------------------------------------------------------------
# BOHAMIANN: Bayesian Optimization with Robust Bayesian Neural Networks
# Scale-Adapted SGHMC 알고리즘 R 구현 (Base R)
# -------------------------------------------------------------------

# 1. 유틸리티 함수: 활성화 함수 및 미분
tanh_act <- function(x) {
  return(tanh(x))
}

dtanh_act <- function(x) {
  return(1 - tanh(x)^2)
}

# 2. 신경망 초기화 함수
# n_in: 입력 차원, n_hidden: 은닉층 노드 수
init_network <- function(n_in, n_hidden, scale = 0.1) {
  # Xavier/Glorot Initialization 스타일
  W1 <- matrix(rnorm(n_in * n_hidden, sd = 1/sqrt(n_in)), nrow = n_in, ncol = n_hidden)
  b1 <- runif(n_hidden, -0.1, 0.1) # bias는 벡터 (R에서는 recycling rule 주의, 여기선 row vector 취급)
  
  W2 <- matrix(rnorm(n_hidden * 1, sd = 1/sqrt(n_hidden)), nrow = n_hidden, ncol = 1)
  b2 <- runif(1, -0.1, 0.1)
  
  # 모든 파라미터를 하나의 리스트로 관리
  params <- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)
  return(params)
}

# 3. Forward Pass (순전파)
forward <- function(X, params) {
  # Layer 1
  # X: (N, n_in), W1: (n_in, n_hidden)
  # sweep을 사용하여 bias 더하기
  z1 <- X %*% params$W1 + matrix(params$b1, nrow = nrow(X), ncol = length(params$b1), byrow = TRUE)
  a1 <- tanh_act(z1)
  
  # Layer 2 (Output)
  z2 <- a1 %*% params$W2 + params$b2
  
  return(list(output = z2, a1 = a1)) # 역전파를 위해 a1 저장
}

# 4. Gradients 계산 (역전파 - Backpropagation)
# 논문의 U(theta)에 대한 Gradient 계산 (Negative Log Posterior)
get_gradients <- function(X, y, params, lambda_prior = 1.0, noise_prec = 10.0) {
  N <- nrow(X)
  fwd <- forward(X, params)
  y_pred <- fwd$output
  a1 <- fwd$a1
  
  # --- Likelihood Gradient (MSE 부분) ---
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
  
  # --- Prior Gradient (Weight Decay 부분) ---
  # Gaussian Prior: -log P(w) ~ (lambda/2) * w^2
  # grad = lambda * w
  # 데이터 스케일에 맞게 N으로 나눠주는 경우가 많으나 여기선 표준 SGHMC 수식 따름
  
  grads <- list(
    W1 = -dW1_lik + lambda_prior * params$W1, # Negative Log Posterior의 Gradient이므로 부호 반전 주의
    b1 = -db1_lik + lambda_prior * params$b1,
    W2 = -dW2_lik + lambda_prior * params$W2,
    b2 = -db2_lik + params$b2 # Bias에는 보통 Prior를 약하게 주거나 안 줌
  )
  
  return(grads)
}

# 5. BOHAMIANN 훈련 함수 (Scale-Adapted SGHMC 구현)
# num_burn: 버닝(초반 샘플 버림), num_samples: 수집할 샘플 수
bohamiann_train <- function(X, y, n_hidden = 50, num_iters = 2000, 
                            keep_every = 50, epsilon = 0.01, mdecay = 0.05) {
  
  n_in <- ncol(X)
  params <- init_network(n_in, n_hidden)
  
  # SGHMC 변수 초기화
  # v: 속도(Momentum), g_sq: Gradient 제곱의 이동평균 (Mass Matrix 추정용)
  v <- list(W1 = params$W1 * 0, b1 = params$b1 * 0, W2 = params$W2 * 0, b2 = params$b2 * 0)
  g_sq <- list(W1 = params$W1 * 0 + 1e-8, b1 = params$b1 * 0 + 1e-8, 
               W2 = params$W2 * 0 + 1e-8, b2 = params$b2 * 0 + 1e-8)
  
  samples <- list() # Posterior 샘플 저장소
  
  cat("BOHAMIANN 학습 시작 (SGHMC)...\n")
  
  for (i in 1:num_iters) {
    # 1. Mini-batch 선택 (여기선 데이터가 적다고 가정하여 Full batch 사용)
    # 실제 구현에선 sample(N, batch_size) 사용
    
    # 2. Gradient 계산 (Negative Log Posterior의 기울기)
    grads <- get_gradients(X, y, params)
    
    # 3. Scale-Adapted SGHMC Update (논문의 핵심)
    # 파라미터별로 루프
    for (p_name in names(params)) {
      g <- grads[[p_name]]
      
      # (A) Mass Matrix 추정 (RMSProp 스타일)
      # g_sq = (1 - mdecay) * g_sq + mdecay * g^2
      g_sq[[p_name]] <- (1 - mdecay) * g_sq[[p_name]] + mdecay * g^2
      
      # M^(-1/2) 계산: Gradient의 표준편차 역수 (Scale)
      inv_M_sqrt <- 1 / sqrt(g_sq[[p_name]])
      
      # (B) Momentum(Velocity) 업데이트
      # v = (1 - friction) * v - step_size * grad + noise
      # 논문의 Scale Adaptation: step_size가 M^(-1/2)에 의해 조정됨
      
      # Noise term: N(0, 2 * epsilon * friction * M^(-1))
      # Adaptive scale 적용 시 noise의 스케일도 조정 필요
      noise_std <- sqrt(2 * epsilon * mdecay) * inv_M_sqrt # 단순화된 형태
      noise <- matrix(rnorm(length(g), 0, 1), nrow=nrow(g), ncol=ncol(g)) * noise_std
      
      # Update Velocity
      # v_{t+1} = (1 - mdecay) * v_t - epsilon * inv_M_sqrt * g + noise
      v[[p_name]] <- (1 - mdecay) * v[[p_name]] - (epsilon * inv_M_sqrt * g) + noise
      
      # (C) Parameter 업데이트
      params[[p_name]] <- params[[p_name]] + v[[p_name]]
    }
    
    # Sampling: Burn-in 이후 일정 간격으로 파라미터 저장
    if (i > (num_iters / 2) && i %% keep_every == 0) {
      samples[[length(samples) + 1]] <- params
    }
    
    if (i %% 200 == 0) {
      cat(sprintf("Iteration %d/%d 완료\n", i, num_iters))
    }
  }
  
  return(samples)
}

# 6. 예측 함수 (Bayesian Model Averaging)
predict_bohamiann <- function(X_test, samples) {
  n_samples <- length(samples)
  preds <- matrix(0, nrow = nrow(X_test), ncol = n_samples)
  
  # 모든 수집된 신경망(Posterior Samples)에 대해 예측 수행
  for (i in 1:n_samples) {
    out <- forward(X_test, samples[[i]])$output
    preds[, i] <- out
  }
  
  # 평균과 분산 계산
  mean_pred <- rowMeans(preds)
  var_pred <- apply(preds, 1, var) # Epistemic Uncertainty (모델 불확실성)
  
  # Aleatoric Uncertainty(데이터 노이즈)를 더해줌 (여기선 고정값 가정 1/noise_prec)
  total_var <- var_pred + (1/10.0) 
  
  return(list(mean = mean_pred, var = total_var, all_preds = preds))
}

# -------------------------------------------------------------------
# [사용 예제]
# -------------------------------------------------------------------

# 1. 데이터 생성 (Sine 함수 + 노이즈)
set.seed(42)
X_train <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
y_train <- sin(X_train) + rnorm(20, 0, 0.1)
X_test <- matrix(seq(-4, 4, length.out = 100), ncol = 1)

# 2. 모델 학습
# 실제로는 iteration을 더 많이(10,000 이상) 해야 잘 수렴함
samples <- bohamiann_train(X_train, y_train, n_hidden = 20, num_iters = 3000, epsilon = 0.005)

# 3. 예측
result <- predict_bohamiann(X_test, samples)

# 4. 시각화
plot(X_test, result$mean, type = "l", col = "blue", lwd = 2, ylim = c(-2, 2),
     main = "BOHAMIANN (R Implementation)", xlab = "X", ylab = "y")
# 신뢰구간 (Confidence Interval)
lines(X_test, result$mean + 2*sqrt(result$var), col = "lightblue", lty = 2)
lines(X_test, result$mean - 2*sqrt(result$var), col = "lightblue", lty = 2)
# 학습 데이터 표시
points(X_train, y_train, col = "red", pch = 19)
legend("topright", legend = c("Mean Pred", "Uncertainty", "Data"),
       col = c("blue", "lightblue", "red"), lty = c(1, 2, 0), pch = c(NA, NA, 19))




















# -------------------------------------------------------------------
# revised for linear regression 
# -------------------------------------------------------------------
sghmc_adaptive_r <- function(X, y, theta_init, n_iter, epsilon, C, lambda, batch_size, mdecay = 0.05) {
  # --- 초기 설정 ---
  N <- nrow(X)
  d <- length(theta_init)
  
  samples <- matrix(0, nrow = n_iter, ncol = d)
  times <- numeric(n_iter)
  
  theta <- theta_init
  
  # Momentum (v) 초기화
  v <- rnorm(d, 0, sqrt(epsilon)) 
  
  # Adaptive Mass를 위한 Gradient 제곱 이동평균 (Preconditioner)
  # 0으로 나누는 것 방지 위해 작은 값(1e-8) 추가
  g_sq <- rep(1e-8, d)
  
  scale <- N / batch_size
  
  start_time <- Sys.time()
  
  for (t in 1:n_iter) {
    
    # 1. Stochastic Gradient Calculation
    # (선형 회귀: MSE Loss + Ridge Prior)
    indices <- sample(1:N, batch_size)
    X_batch <- X[indices, , drop=FALSE]
    y_batch <- y[indices]
    
    # Prediction & Residual
    pred <- X_batch %*% theta
    residual <- y_batch - pred
    
    # Gradient: -Likelihood_grad + Prior_grad
    # (주의: 사용하시는 수식 부호에 따라 조정. 여기서는 Negative Log Posterior의 Gradient)
    grad <- as.vector(scale * (-t(X_batch) %*% residual) + lambda * theta)
    
    # 2. Adaptive Mass Update (RMSProp style)
    # BOHAMIANN 논문 로직: g_sq = (1-tau)*g_sq + tau*g^2
    g_sq <- (1 - mdecay) * g_sq + mdecay * (grad^2)
    
    # Preconditioner: M^(-1/2) corresponds to 1 / sqrt(V_hat)
    sigma_inv <- 1 / sqrt(g_sq)
    
    # 3. Momentum & Position Update (Scale-Adapted)
    # 논문의 Update Rule을 따름:
    # Noise variance도 Adaptive term(sigma_inv)에 따라 스케일링됨
    
    # 노이즈 표준편차 계산: sqrt(2 * epsilon * C * M^(-1))
    # 여기서 M^(-1) 효과를 내기 위해 sigma_inv가 개입됨 (논문 변형에 따라 다름)
    # BOHAMIANN 단순화 버전: noise ~ N(0, 2 * epsilon * mdecay * sigma_inv)
    # Friction term도 C 대신 mdecay와 sigma_inv의 조합으로 표현되기도 함.
    # 여기서는 Standard SGHMC에 Mass Matrix M만 바뀐 형태로 구현:
    
    noise_std <- sqrt(2 * epsilon * C) * sqrt(sigma_inv) # M^(-1/2) scaling roughly
    noise <- rnorm(d, 0, 1) * noise_std
    
    # v update: v = v - epsilon * grad - epsilon * C * v * M^(-1) + noise
    # Adaptive에서는 grad와 friction에 Preconditioner(sigma_inv)가 곱해짐
    v <- v - (epsilon * grad * sigma_inv) - (epsilon * C * v * sigma_inv) + noise
    
    # theta update
    theta <- theta + v
    
    # 저장
    samples[t, ] <- theta
    times[t] <- as.numeric(Sys.time() - start_time)
  }
  
  return(list(Time = times, Theta = samples))
}