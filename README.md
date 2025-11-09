new# 5430_project
test change #afternoon
R & Python needed to be installed

Goal: adjust HMC with regards to gradient vector

HMC.R: HMC vs (RWM, Adaptive MH)

Literature review: https://arxiv.org/pdf/1402.4102, https://arxiv.org/pdf/1701.02434

sghmc package in r needs TensorFlow ver. 1.x which only works in Intel(x86_64).
i.e., in ARM64 environmnet, we cannot utilize the package.
Even if we contour this obstacle by Rosseta, tensorflow 1.x needs AVX(Advanced Vector Extensions) of CPU which Apple Silicon does not have.
https://www.rdocumentation.org/packages/sgmcmc/versions/0.2.5/topics/sghmc,
https://arxiv.org/pdf/1710.00578
