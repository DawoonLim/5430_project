new# 5430_project
test change #afternoon
R & Python needed to be installed

/finals   ─ Simulation code used in the final report 

/drafts   ─ Test, scratch, and development code

/R        ─ Implementations of algorithms and literature simulation scripts 

The existing R package for HMC: https://www.rdocumentation.org/packages/hmclearn/versions/0.0.5/topics/hmc


The sghmc package for R depends on TensorFlow 1.x, which is distributed only for Intel (x86_64) platforms. 
As a result, it is not compatible with ARM64 (Apple Silicon) environments. 
While Rosetta 2 may allow some x86_64 binaries to run on Apple Silicon, TensorFlow 1.x requires CPU features (e.g., AVX — Advanced Vector Extensions) that Apple Silicon lacks, preventing reliable operation. 
Thus, these functions donot work in R: https://www.rdocumentation.org/packages/sgmcmc/versions/0.2.5/topics/sghmc and https://arxiv.org/pdf/1710.00578
