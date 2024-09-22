// 
//

data {
  int<lower=1> N_antibiotic;      // number of non-zero antibiotic concentrations
  
  vector[N_antibiotic] y;         // normalized fitness difference with antibiotic
  vector[N_antibiotic] y_err;     // estimated error of fitness difference
  
  real log_g_min;                    // lower bound on log_g
  real log_g_max;                    // upper bound on log_g
  
  // prior means (mu) and standard deviations for fitness calibration parameters:
  vector[N_antibiotic] low_fitness_mu;       // fitness difference at zero function
  vector[N_antibiotic] mid_g_mu;             // function level at 1/2 max fitness difference
  vector[N_antibiotic] fitness_n_mu;         // cooperativity coefficient of fitness calibration curve
  
  vector[N_antibiotic] low_fitness_std;      // fitness difference at zero function
  vector[N_antibiotic] mid_g_std;            // function level at 1/2 max fitness difference
  vector[N_antibiotic] fitness_n_std;        // cooperativity coefficient of fitness calibration curve
  
}

transformed data {
  
}

parameters {
  real<lower=log_g_min, upper=log_g_max> log_g;        // log10 of function
  
  real<lower=0> sigma;            // scale factor for standard deviation of noise in y
  
  vector[N_antibiotic] low_fitness;       // fitness difference at zero function
  vector[N_antibiotic] mid_g;             // gene expression level at 1/2 max fitness difference
  vector[N_antibiotic] fitness_n;         // cooperativity coefficient of fitness calibration curve
}

transformed parameters {
  real g;
  
  vector[N_antibiotic] mean_y;
  
  g = 10^log_g;
  
  for (n in 1:N_antibiotic) {
    mean_y[n] = low_fitness[n] - low_fitness[n]*(g^fitness_n[n])/(mid_g[n]^fitness_n[n] + g^fitness_n[n]);
  }
  
}

model {
  
  low_fitness ~ normal(low_fitness_mu, low_fitness_std);
  mid_g ~ normal(mid_g_mu, mid_g_std);
  fitness_n ~ normal(fitness_n_mu, fitness_n_std);
    
  // noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);
  
  y ~ normal(mean_y, sigma*y_err);
  
}

generated quantities {
  real rms_resid;
  
  rms_resid = distance(y, mean_y)/sqrt(N_antibiotic);
  
}
