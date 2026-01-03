// 
//

data {
  int<lower=0> N_lig;                // number of non-zero ligand concentrations for each ligand
  
  vector[N_lig] x_1;                 // non-zero ligand 1 concentrations
  vector[N_lig] x_2;                 // non-zero ligand 2 concentrations
  vector[N_lig] x_3;                 // non-zero ligand 3 concentrations
  
  real log_g_min;                    // lower bound on log_g0 and log_ginf
  real log_g_max;                    // upper bound on log_g0 and log_ginf
  real log_g_prior_scale;
  
  real high_fitness_mu;      // fitness difference at +infinite gene expression, with antibiotic
  real mid_g_mu;             // gene expression level at 1/2 max fitness difference, with antibiotic
  real fitness_n_mu;         // cooperativity coefficient of fitness difference curve, with antibiotic
  
  real high_fitness_std;       // fitness difference at +infinite gene expression, with antibiotic
  real mid_g_std;             // gene expression level at 1/2 max fitness difference, with antibiotic
  real fitness_n_std;         // cooperativity coefficient of fitness difference curve, with antibiotic
  
}

transformed data {
  real x_min;
  real x_max;
  real log_x_min;
  real log_x_max;
  real low_fitness;
  
  low_fitness = 0;
  
  x_max = fmax(max(x_1), max(x_2));
  x_max = fmax(x_max, max(x_3));
  
  x_min = fmin(min(x_1), min(x_2));
  x_min = fmin(x_min, min(x_3));
  
  log_x_max = log10(x_max) + 1.4;
  log_x_min = log10(x_min) - 0.3;
  
}

parameters {
  real<lower=log_g_min, upper=log_g_max> log_g0;        // log10 of gene expression level at zero ligand
  
  real<lower=log_g_min, upper=log_g_max>  log_ginf_1;   // log10 of gene expression level at saturating concentration of ligand 1
  real<lower=log_g_min, upper=log_g_max>  log_ginf_2;   // log10 of gene expression level at saturating concentration of ligand 2
  real<lower=log_g_min, upper=log_g_max>  log_ginf_3;   // log10 of gene expression level at saturating concentration of ligand 3
  
  real<lower=log_x_min, upper=log_x_max> log_ec50_1;    // input level (x) that gives output 1/2 way between g0 and ginf for ligand 1
  real<lower=log_x_min, upper=log_x_max> log_ec50_2;    // input level (x) that gives output 1/2 way between g0 and ginf for ligand 2
  real<lower=log_x_min, upper=log_x_max> log_ec50_3;    // input level (x) that gives output 1/2 way between g0 and ginf for ligand 3
  
  real<lower=0> sensor_n_1;                             // cooperativity exponent of sensor gene expression vs. x curve for ligand 1
  real<lower=0> sensor_n_2;                             // cooperativity exponent of sensor gene expression vs. x curve for ligand 2
  real<lower=0> sensor_n_3;                             // cooperativity exponent of sensor gene expression vs. x curve for ligand 3
  
  real<lower=0> sigma;            // scale factor for standard deviation of noise in y
  
  real high_fitness;       // fitness difference at +infinite gene expression, med tet
  real mid_g;             // gene expression level at 1/2 max fitness difference, med tet
  real fitness_n;         // cooperativity coefficient of fitness difference curve, med tet
}

transformed parameters {
  
}

model {
  
  high_fitness ~ normal(high_fitness_mu, high_fitness_std);
  mid_g ~ normal(mid_g_mu, mid_g_std);
  fitness_n ~ normal(fitness_n_mu, fitness_n_std);
  
  // Prior on sensor_n; <gamma> = alpha/beta = 2.2; std = sqrt(alpha)/beta = 0.8
  sensor_n_1 ~ gamma(7.7, 3.5);
  sensor_n_2 ~ gamma(7.7, 3.5);
  sensor_n_3 ~ gamma(7.7, 3.5);
  
  // Prior on log_ec50; flat prior with erf boundaries
  target += log1m(erf((log_x_min + 0.3 - log_ec50_1)/0.2));
  target += log1m(erf((log_ec50_1 - log_x_max + 0.3)/0.3));
  
  target += log1m(erf((log_x_min + 0.3 - log_ec50_2)/0.2));
  target += log1m(erf((log_ec50_2 - log_x_max + 0.3)/0.3));
  
  target += log1m(erf((log_x_min + 0.3 - log_ec50_3)/0.2));
  target += log1m(erf((log_ec50_3 - log_x_max + 0.3)/0.3));
  
  // noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);
  
}

generated quantities {
  
}
