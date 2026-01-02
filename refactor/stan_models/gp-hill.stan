data {
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // fitness difference at each concentration
  vector[N] y_err;       // estimated error of fitness difference at each concentration
  
  real low_fitness_mu;      // fitness diference at zero gene expression
  real mid_g_mu;            // gene expression evel at 1/2 max fitness difference
  real fitness_n_mu;        // cooperativity coeficient of fitness difference curve
}

transformed data {
  //vector[N] mu_gp = rep_vector(0, N);
  real x_gp[N];
  for (i in 1:N){
    x_gp[i] = log(x[i] + 1);
  }
}

parameters {
  real<lower=1., upper=4> log_low_level;               // log10 of gene expression level at zero induction
  real<lower=1., upper=4>  log_high_level;             // log10 of gene expression level at infinite induction
  real<lower=-1, upper=4.6> log_IC_50;        // input level (x) that gives output 1/2 way between low_level and high_level
  real<lower=0> sensor_n;                    // cooperativity exponent of sensor gene expression vs. x curve
  real<lower=0> sigma;                     // scale factor for standard deviation of noise in y

  // vector[N] g;   // gene expression level at each concentration (GP + hill)
  
  real low_fitness;      // fitness diference at zero gene expression
  real mid_g;            // gene expression evel at 1/2 max fitness difference
  real fitness_n;        // cooperativity coeficient of fitness difference curve

  // gp params
  real<lower=0> rho;
  real<lower=0> alpha;
  vector[N] eta;
}

transformed parameters {
  real low_level;        
  real high_level; 
  real IC_50;
  
  vector[N] mean_y;
  // vector[N] g_mu;   // hill mean of gene expression
  vector[N] g; // GP approx to gene expression
  // vector[N] ghat; // GP approx to gene expression
  
  IC_50 = 10^log_IC_50;
  low_level = 10^log_low_level;
  high_level = 10^log_high_level;

  for (i in 1:N) {
    // g_mu[i] = low_level + (high_level - low_level)*(x[i]^sensor_n)/(IC_50^sensor_n + x[i]^sensor_n);
  }

  {
    matrix[N, N] L_K;
    matrix[N, N] K = cov_exp_quad(x_gp, alpha, rho);
    // real g_mu_mu = mean(g_mu);
    // real g_mu_std = sqrt(variance(g_mu));

    // diagonal elements
    for (n in 1:N)
      K[n, n] = K[n, n] + 1e-9;

    L_K = cholesky_decompose(K);

    // g = L_K * eta;
    g = 2.5 + L_K * eta;
    for (i in 1:N){
      g[i] = pow(g[i], 10);
    }
    //g = g_mu_std*g + g_mu;
    //g = g + g_mu;
    // ghat = g + g_mu;
  }
  
  for (i in 1:N) {
    mean_y[i] = low_fitness - low_fitness*(g[i]^fitness_n)/(mid_g^fitness_n + g[i]^fitness_n);
    // mean_y[i] = low_fitness - low_fitness*(ghat[i]^fitness_n)/(mid_g^fitness_n + ghat[i]^fitness_n);
  }

  
}

model {
  // regular stuff
  low_fitness ~ normal(low_fitness_mu, 0.05);
  mid_g ~ normal(mid_g_mu, 10);
  fitness_n ~ normal(fitness_n_mu, 0.03);
  
  sensor_n ~ gamma(4.0, 10.0/3.0);
  //log_IC_50 ~ normal(1.81, 1);
  target += log1m(erf((-0.3 - log_IC_50)/0.5));
  target += log1m(erf((log_IC_50 - 3.8)/0.3));
  
  //log_low_level ~ normal(2, 1);
  target += log1m(erf((1.9 - log_low_level)/0.3));
  target += log1m(erf((log_low_level - 3.6)/0.3));
  //log_high_level ~ normal(3.2, 0.5);
  target += log1m(erf((1.9 - log_high_level)/0.3));
  target += log1m(erf((log_high_level - 3.6)/0.3));

  // GP
  rho ~ inv_gamma(5, 5);
  // alpha ~ normal(100, 50);
  alpha ~ std_normal();
  eta ~ std_normal();

  // observations
  y ~ normal(mean_y, sigma*y_err);

}

generated quantities {
  real log_sensor_n;
  real rms_resid;
  
  log_sensor_n = log10(sensor_n);
  
  rms_resid = distance(y, mean_y)/N;
  
}
