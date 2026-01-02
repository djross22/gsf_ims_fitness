// 
//

data {
  int<lower=1> N_antibiotic;      // number of non-zero antibiotic concentrations
  int<lower=1> N_lig;             // number of ligand concentrations at each antibiotic concentration
  
  array[N_lig, N_antibiotic] real y;         // normalized fitness difference with antibiotic
  array[N_lig, N_antibiotic] real y_err;     // estimated error of fitness difference
  
  real log_g_min;                    // lower bound on log_g
  real log_g_max;                    // upper bound on log_g
  
  // prior means (mu) and standard deviations for fitness calibration parameters:
  vector[N_antibiotic] low_fitness_mu;       // fitness difference at zero function
  vector[N_antibiotic] high_fitness_mu;      // fitness difference at saturated/maximum function
  vector[N_antibiotic] mid_g_mu;             // function level at 1/2 max fitness difference
  vector[N_antibiotic] fitness_n_mu;         // cooperativity coefficient of fitness calibration curve
  
  vector[N_antibiotic] low_fitness_std;      // fitness difference at zero function
  vector[N_antibiotic] high_fitness_std;     // fitness difference at saturated/maximum function
  vector[N_antibiotic] mid_g_std;            // function level at 1/2 max fitness difference
  vector[N_antibiotic] fitness_n_std;        // cooperativity coefficient of fitness calibration curve
  
}

transformed data {
  vector[N_antibiotic] low_fitness;       // fitness difference at zero function
  vector[N_antibiotic] high_fitness;      // fitness difference at saturated/maximum function
  vector[N_antibiotic] mid_g;             // gene expression level at 1/2 max fitness difference
  vector[N_antibiotic] fitness_n;         // cooperativity coefficient of fitness calibration curve
  
  real<lower=0> sigma;            // scale factor for standard deviation of noise in y
  
  low_fitness = low_fitness_mu;
  high_fitness = high_fitness_mu;
  mid_g = mid_g_mu;
  fitness_n = fitness_n_mu;
  
  sigma = 1;
}

parameters {
  vector<lower=log_g_min, upper=log_g_max>[N_lig] log_g;        // log10 of function
  
  //real<lower=0> sigma;            // scale factor for standard deviation of noise in y
  
  //vector[N_antibiotic] low_fitness;       // fitness difference at zero function
  //vector[N_antibiotic] high_fitness;      // fitness difference at saturated/maximum function
  //vector[N_antibiotic] mid_g;             // gene expression level at 1/2 max fitness difference
  //vector[N_antibiotic] fitness_n;         // cooperativity coefficient of fitness calibration curve
}

transformed parameters {
  vector[N_lig] g;
  
  array[N_lig, N_antibiotic] real mean_y;
  
  g = 10^log_g;
  
  for (lig in 1:N_lig) {
    for (n in 1:N_antibiotic) {
      mean_y[lig][n] = low_fitness[n] + (high_fitness[n] - low_fitness[n])*(g[lig]^fitness_n[n])/(mid_g[n]^fitness_n[n] + g[lig]^fitness_n[n]);
    }
  }
  
}

model {
  
  //low_fitness ~ normal(low_fitness_mu, low_fitness_std);
  //high_fitness ~ normal(high_fitness_mu, high_fitness_std);
  //mid_g ~ normal(mid_g_mu, mid_g_std);
  //fitness_n ~ normal(fitness_n_mu, fitness_n_std);
    
  // noise scale, prior to keep it from getting too much < 1
  //sigma ~ inv_gamma(3, 0.6);
  
  for (lig in 1:N_lig) {
    for (n in 1:N_antibiotic) {
      y[lig][n] ~ normal(mean_y[lig][n], sigma*y_err[lig][n]);
    }
  }
}

generated quantities {
  real rms_resid;
  
  rms_resid = 0;
  for (lig in 1:N_lig) {
    for (n in 1:N_antibiotic) {
      rms_resid = rms_resid + (y[lig][n] - mean_y[lig][n])^2;
    }
  }
  rms_resid = sqrt(rms_resid/(N_antibiotic*N_lig));
}
