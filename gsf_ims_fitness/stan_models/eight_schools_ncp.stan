data {
  int<lower=0> J;
  real y[J];
  real<lower=0> sigma[J];
}

parameters {
  real mu;
  real<lower=0> tau;
  real theta_tilde[J];
}

transformed parameters {
  real theta[J];
  for (j in 1:J)
    theta[j] = mu + tau * theta_tilde[j];
}

model {
  mu ~ normal(0, 1);
  tau ~ normal(0.5, 1);
  theta_tilde ~ normal(0, 1);
  y ~ normal(theta, sigma);
}

generated quantities {
  real log_tau;
  
  log_tau = log(tau);
}
