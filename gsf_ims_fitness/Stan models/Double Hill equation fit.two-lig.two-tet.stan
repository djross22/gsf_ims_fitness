// 
//

data {
  int<lower=1> N_lig;                // number of non-zero ligand concentrations for each ligand
  
  vector[N_lig] x_1;                 // non-zero ligand 1 concentrations
  vector[N_lig] x_2;                 // non-zero ligand 2 concentrations
  
  real y_0_low_tet;                  // fitness difference with zero ligand and medium tet concentration
  real y_0_low_tet_err;              // estimated error of fitness difference
  
  vector[N_lig] y_1_low_tet;         // fitness difference with ligand 1 and medium tet concentration
  vector[N_lig] y_1_low_tet_err;     // estimated error of fitness difference
  vector[N_lig] y_2_low_tet;         // fitness difference with ligand 2 and medium tet concentration
  vector[N_lig] y_2_low_tet_err;     // estimated error of fitness difference
  
  vector[N_lig] y_1_high_tet;        // fitness difference with ligand 1 and high tet concentration
  vector[N_lig] y_1_high_tet_err;    // estimated error of fitness difference
  vector[N_lig] y_2_high_tet;        // fitness difference with ligand 2 and high tet concentration
  vector[N_lig] y_2_high_tet_err;    // estimated error of fitness difference
  
  real log_g_min;                    // lower bound on log_g0 and log_ginf
  real log_g_max;                    // upper bound on log_g0 and log_ginf
  real log_g_prior_scale;
  
  real low_fitness_mu_low_tet;       // fitness difference at zero gene expression, medium tet
  real mid_g_mu_low_tet;             // gene expression level at 1/2 max fitness difference, medium tet
  real fitness_n_mu_low_tet;         // cooperativity coefficient of fitness difference curve, medium tet
  
  real low_fitness_std_low_tet;       // fitness difference at zero gene expression, medium tet
  real mid_g_std_low_tet;             // gene expression level at 1/2 max fitness difference, medium tet
  real fitness_n_std_low_tet;         // cooperativity coefficient of fitness difference curve, medium tet
  
  real low_fitness_mu_high_tet;      // fitness difference at zero gene expression, high tet
  real mid_g_mu_high_tet;            // gene expression level at 1/2 max fitness difference, high tet
  real fitness_n_mu_high_tet;        // cooperativity coefficient of fitness difference curve, high tet
  
  real low_fitness_std_high_tet;      // fitness difference at zero gene expression, high tet
  real mid_g_std_high_tet;            // gene expression level at 1/2 max fitness difference, high tet
  real fitness_n_std_high_tet;        // cooperativity coefficient of fitness difference curve, high tet
  
  array[2] real log_x_max;           // maximum possible value for log_ec50, previously set to log10(max(x)) + 1.289;
  
}

transformed data {
  real x_min;
  real x_max;
  real log_x_1_min;
  real log_x_1_max;
  real log_x_2_min;
  real log_x_2_max;
  
  x_max = max(x_1);
  x_min = min(x_1);
  log_x_1_max = log_x_max[1];
  log_x_1_min = log10(x_min) - 1.3;
  
  x_max = max(x_2);
  x_min = min(x_2);
  log_x_2_max = log_x_max[2];
  log_x_2_min = log10(x_min) - 1.3;
  
}

parameters {
  real<lower=log_g_min, upper=log_g_max> log_g0;        // log10 of gene expression level at zero ligand
  
  real<lower=log_g_min, upper=log_g_max>  log_ginf_1;   // log10 of gene expression level at saturating concentration of ligand 1
  real<lower=log_g_min, upper=log_g_max>  log_ginf_2;   // log10 of gene expression level at saturating concentration of ligand 2
  
  real<lower=log_x_1_min, upper=log_x_1_max> log_ec50_1;    // input level (x) that gives output 1/2 way between g0 and ginf for ligand 1
  real<lower=log_x_2_min, upper=log_x_2_max> log_ec50_2;    // input level (x) that gives output 1/2 way between g0 and ginf for ligand 2
  
  real<lower=0> sensor_n_1;                             // cooperativity exponent of sensor gene expression vs. x curve for ligand 1
  real<lower=0> sensor_n_2;                             // cooperativity exponent of sensor gene expression vs. x curve for ligand 2
  
  real<lower=0> sigma;            // scale factor for standard deviation of noise in y
  
  real low_fitness_low_tet;       // fitness difference at zero gene expression, med tet
  real mid_g_low_tet;             // gene expression level at 1/2 max fitness difference, med tet
  real fitness_n_low_tet;         // cooperativity coefficient of fitness difference curve, med tet
  
  real low_fitness_high_tet;      // fitness difference at zero gene expression, high tet
  real mid_g_high_tet;            // gene expression level at 1/2 max fitness difference, high tet
  real fitness_n_high_tet;        // cooperativity coefficient of fitness difference curve, high tet
}

transformed parameters {
  real g0;
  real ginf_1; 
  real ec50_1;
  real ginf_2; 
  real ec50_2;
  
  real mean_y_0_low_tet;
  
  vector[N_lig] mean_y_1_low_tet;
  vector[N_lig] mean_y_1_high_tet;
  vector[N_lig] g_1;                // gene expression level at each non-zero concentration of ligand 1
  
  vector[N_lig] mean_y_2_low_tet;
  vector[N_lig] mean_y_2_high_tet;
  vector[N_lig] g_2;                // gene expression level at each non-zero concentration of ligand 2
  
  g0 = 10^log_g0;
  ginf_1 = 10^log_ginf_1;
  ec50_1 = 10^log_ec50_1;
  ginf_2 = 10^log_ginf_2;
  ec50_2 = 10^log_ec50_2;
  
  mean_y_0_low_tet = low_fitness_low_tet - low_fitness_low_tet*(g0^fitness_n_low_tet)/(mid_g_low_tet^fitness_n_low_tet + g0^fitness_n_low_tet);
  
  for (i in 1:N_lig) {
    g_1[i] = g0 + (ginf_1 - g0)*(x_1[i]^sensor_n_1)/(ec50_1^sensor_n_1 + x_1[i]^sensor_n_1);
    g_2[i] = g0 + (ginf_2 - g0)*(x_2[i]^sensor_n_2)/(ec50_2^sensor_n_2 + x_2[i]^sensor_n_2);
	
    mean_y_1_low_tet[i] = low_fitness_low_tet - low_fitness_low_tet*(g_1[i]^fitness_n_low_tet)/(mid_g_low_tet^fitness_n_low_tet + g_1[i]^fitness_n_low_tet);
    mean_y_2_low_tet[i] = low_fitness_low_tet - low_fitness_low_tet*(g_2[i]^fitness_n_low_tet)/(mid_g_low_tet^fitness_n_low_tet + g_2[i]^fitness_n_low_tet);
	
    mean_y_1_high_tet[i] = low_fitness_high_tet - low_fitness_high_tet*(g_1[i]^fitness_n_high_tet)/(mid_g_high_tet^fitness_n_high_tet + g_1[i]^fitness_n_high_tet);
    mean_y_2_high_tet[i] = low_fitness_high_tet - low_fitness_high_tet*(g_2[i]^fitness_n_high_tet)/(mid_g_high_tet^fitness_n_high_tet + g_2[i]^fitness_n_high_tet);
  }
  
}

model {
  //real neg_low_fitness_low_tet;
  //real neg_low_fitness_high_tet;
  
  //neg_low_fitness_low_tet = -1*low_fitness_low_tet;
  //neg_low_fitness_high_tet = -1*low_fitness_high_tet;
  
  // fitness calibration params
  //low_fitness ~ student_t(8, low_fitness_mu, 0.1);
  //neg_low_fitness_low_tet ~ exp_mod_normal(-1*low_fitness_mu_low_tet - 0.04, 0.03, 14); 
  
  low_fitness_low_tet ~ normal(low_fitness_mu_low_tet, low_fitness_std_low_tet);
  mid_g_low_tet ~ normal(mid_g_mu_low_tet, mid_g_std_low_tet);
  fitness_n_low_tet ~ normal(fitness_n_mu_low_tet, fitness_n_std_low_tet);
  
  low_fitness_high_tet ~ normal(low_fitness_mu_high_tet, low_fitness_std_high_tet);
  mid_g_high_tet ~ normal(mid_g_mu_high_tet, mid_g_std_high_tet);
  fitness_n_high_tet ~ normal(fitness_n_mu_high_tet, fitness_n_std_high_tet);
  
  
  // Prior on sensor_n; <gamma> = alpha/beta = 1.5; std = sqrt(alpha)/beta = 0.5
  sensor_n_1 ~ gamma(9.0, 6.0);
  sensor_n_2 ~ gamma(9.0, 6.0);
  
  // Prior on log_ec50; flat prior with erf boundaries
  target += log1m(erf((log_x_1_min + 0.7 - log_ec50_1)/0.5));
  target += log1m(erf((log_ec50_1 - log_x_1_max + 0.8)/0.3));
  
  target += log1m(erf((log_x_2_min + 0.7 - log_ec50_2)/0.5));
  target += log1m(erf((log_ec50_2 - log_x_2_max + 0.8)/0.3));
  
  // noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);
  
  y_0_low_tet ~ normal(mean_y_0_low_tet, sigma*y_0_low_tet_err);
  
  y_1_low_tet ~ normal(mean_y_1_low_tet, sigma*y_1_low_tet_err);
  y_2_low_tet ~ normal(mean_y_2_low_tet, sigma*y_2_low_tet_err);
  
  y_1_high_tet ~ normal(mean_y_1_high_tet, sigma*y_1_high_tet_err);
  y_2_high_tet ~ normal(mean_y_2_high_tet, sigma*y_2_high_tet_err);
  
}

generated quantities {
  real log_sensor_n_1;
  real log_sensor_n_2;
  
  real log_ginf_g0_ratio_1;
  real log_ginf_g0_ratio_2;
  
  real rms_resid;
  
  log_ginf_g0_ratio_1 = log_ginf_1 - log_g0;
  log_ginf_g0_ratio_2 = log_ginf_2 - log_g0;
  
  log_sensor_n_1 = log10(sensor_n_1);
  log_sensor_n_2 = log10(sensor_n_2);
  
  rms_resid = sqrt((y_0_low_tet - mean_y_0_low_tet)^2 + distance(y_1_low_tet, mean_y_1_low_tet)^2 + distance(y_2_low_tet, mean_y_2_low_tet)^2 + distance(y_1_high_tet, mean_y_1_high_tet)^2 + distance(y_2_high_tet, mean_y_2_high_tet)^2)/sqrt(4*N_lig + 1);
  
}
