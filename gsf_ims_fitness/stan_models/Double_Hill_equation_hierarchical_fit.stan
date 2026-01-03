// Used for getting concensus Hill eqn. parameters from multiple barcodes with the same lacI sequence
//

data {
  int<lower=2> J;        // number of barcodes with the same lacI sequence
  int<lower=1> N;        // number of inducer concentrations
  vector[N] x;           // inducer concentration
  vector[N] y[J];        // fitness difference at each concentration
  vector[N] y_err[J];    // estimated error of fitness difference at each concentration
  real log_g_min;        // lower bound on log_low_level and log_high_level
  real log_g_max;        // upper bound on log_low_level and log_high_level
  real log_g_prior_scale;
  
  real low_fitness_mu;      // fitness difference at zero gene expression
  real mid_g_mu;            // gene expression level at 1/2 max fitness difference
  real fitness_n_mu;        // cooperativity coefficient of fitness difference curve
  
}

transformed data {
  real x_min;
  real x_max;
  real log_x_min;
  real log_x_max;
  
  x_max = max(x);
  x_min = x_max;
  for (i in 1:N) {
    if (x[i]>0) {
      if (x[i]<x_min) {
        x_min = x[i];
      }
    }
  }
  
  log_x_max = log10(x_max) + 1.289;
  log_x_min = log10(x_min) - 1.3;
  
}

parameters {
  real<lower=log_g_min, upper=log_g_max> log_low_level;       // log10 of gene expression level at zero induction
  real<lower=log_g_min, upper=log_g_max>  log_high_level;     // log10 of gene expression level at infinite induction
  
  real<lower=log_x_min, upper=log_x_max> log_IC_50;           // input level (x) that gives output 1/2 way between low_level and high_level
  real<lower=0> sensor_n;                                     // cooperativity exponent of sensor gene expression vs. x curve
  real<lower=0> sigma[J];                                        // scale factor for standard deviation of noise in y
  
  real low_fitness[J];      // fitness difference at zero gene expression
  real mid_g[J];            // gene expression level at 1/2 max fitness difference
  real fitness_n[J];        // cooperativity coefficient of fitness difference curve
}

transformed parameters {
  real low_level;        
  real high_level; 
  real IC_50;
  
  vector[N] mean_y[J];
  vector[N] g;   // gene expression level at each concentration
  
  IC_50 = 10^log_IC_50;
  low_level = 10^log_low_level;
  high_level = 10^log_high_level;
  
  for (i in 1:N) {
    g[i] = low_level + (high_level - low_level)*(x[i]^sensor_n)/(IC_50^sensor_n + x[i]^sensor_n);
    for (j in 1:J) {
      mean_y[j][i] = low_fitness[j] - low_fitness[j]*(g[i]^fitness_n[j])/(mid_g[j]^fitness_n[j] + g[i]^fitness_n[j]);
    }
  }
  
}

model {
  //real neg_low_fitness[J];
  //for (j in 1:J) {
  //  neg_low_fitness[j] = -1*low_fitness[j];
  //}
  
  // fitness calibration params
  low_fitness ~ student_t(8, low_fitness_mu, 0.1);
  //neg_low_fitness ~ exp_mod_normal(-1*low_fitness_mu-0.04, 0.03, 14); 
  
  //mid_g ~ normal(mid_g_mu, 27); // use with PTY1
  //fitness_n ~ normal(fitness_n_mu, 0.22); // use with pTY1
  mid_g ~ normal(mid_g_mu, 499); // use with pVER
  fitness_n ~ normal(fitness_n_mu, 0.29); // use with pVER
  
  // Prior on sensor_n
  sensor_n ~ gamma(4.0, 10.0/3.0);
  
  // Prior on log_IC_50
  target += log1m(erf((log_x_min + 0.7 - log_IC_50)/0.5));
  target += log1m(erf((log_IC_50 - log_x_max + 0.8)/0.3));
  
  // Prior on log_low_level
  target += log1m(erf((log_g_min + 0.9 - log_low_level)/log_g_prior_scale));
  target += log1m(erf((log_low_level - log_g_max + 0.9)/log_g_prior_scale));
  
  // Prior on log_high_level
  target += log1m(erf((log_g_min + 0.9 - log_high_level)/log_g_prior_scale));
  target += log1m(erf((log_high_level - log_g_max + 0.9)/log_g_prior_scale));
  
  // noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);
  
  for (j in 1:J) {
    y[j] ~ normal(mean_y[j], sigma[j]*y_err[j]);
  }

}

generated quantities {
  real log_sensor_n;
  //real rms_resid;
  real log_high_low_ratio;
  
  log_high_low_ratio = log_high_level - log_low_level;
  
  log_sensor_n = log10(sensor_n);
  
  //rms_resid = distance(y, mean_y)/sqrt(N);
  
}
