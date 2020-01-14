data {
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // fitness difference at each concentration
  vector[N] y_err;       // estimated error of fitness difference at each concentration
  
  real low_fitness_mu;      // fitness difference at zero gene expression
  real mid_g_mu;            // gene expression level at 1/2 max fitness difference
  real fitness_n_mu;        // cooperativity coefficient of fitness difference curve
}

transformed data {
  real x_gp[N];
  for (i in 1:N){
    x_gp[i] = log10(x[i] + 0.25);
  }
}

parameters {
  real<lower=0> sigma;                     // scale factor for standard deviation of noise in y
  
  real low_fitness;      // fitness difference at zero gene expression
  real mid_g;            // gene expression level at 1/2 max fitness difference
  real fitness_n;        // cooperativity coefficient of fitness difference curve

  // gp params
  real<lower=0> rho;
  real<lower=0> alpha;
  vector[N] eta;
}

transformed parameters {
  vector[N] mean_y;
  vector[N] g; // GP approx to gene expression

  {
    matrix[N, N] L_K;
    matrix[N, N] K = cov_exp_quad(x_gp, alpha, rho);

    // diagonal elements
    for (n in 1:N)
      K[n, n] = K[n, n] + 1e-9;

    L_K = cholesky_decompose(K);

    g = 2.5 + L_K * eta;
    for (i in 1:N){
      g[i] = 10^g[i];
    }
  }
  
  for (i in 1:N) {
    mean_y[i] = low_fitness - low_fitness*(g[i]^fitness_n)/(mid_g^fitness_n + g[i]^fitness_n);
  }

  
}

model {
  // regular stuff
  low_fitness ~ normal(low_fitness_mu, 0.05);
  mid_g ~ normal(mid_g_mu, 10);
  fitness_n ~ normal(fitness_n_mu, 0.03);

  // GP
  rho ~ inv_gamma(5, 5);
  alpha ~ normal(.25, .25);
  eta ~ std_normal();

  // observations
  y ~ normal(mean_y, sigma*y_err);

}

generated quantities {
  real rms_resid;
  
  rms_resid = distance(y, mean_y)/N;
  
}
