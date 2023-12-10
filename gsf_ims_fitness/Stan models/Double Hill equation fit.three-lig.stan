// 
//

data {
  int<lower=1> N_lig;                // number of non-zero ligand concentrations for each ligand
  
  vector[N_lig] x_1;                 // non-zero ligand 1 concentrations
  vector[N_lig] x_2;                 // non-zero ligand 2 concentrations
  vector[N_lig] x_3;                 // non-zero ligand 3 concentrations
  
  vector[2] y_0;                  // fitness difference with zero ligand and with antibiotic; the experiment has two replicate wells for this condition
  vector[2] y_0_err;              // estimated error of fitness difference
  
  vector[N_lig] y_1;         // fitness difference with ligand 1 and with antibiotic
  vector[N_lig] y_1_err;     // estimated error of fitness difference
  vector[N_lig] y_2;         // fitness difference with ligand 2 and with antibiotic
  vector[N_lig] y_2_err;     // estimated error of fitness difference
  vector[N_lig] y_3;         // fitness difference with ligand 3 and with antibiotic
  vector[N_lig] y_3_err;     // estimated error of fitness difference
  
  real log_g_min;                    // lower bound on log_g0 and log_ginf
  real log_g_max;                    // upper bound on log_g0 and log_ginf
  real log_g_prior_scale;
  
  real low_fitness_mu;      // fitness difference at +infinite gene expression, with antibiotic
  real mid_g_mu;             // gene expression level at 1/2 max fitness difference, with antibiotic
  real fitness_n_mu;         // cooperativity coefficient of fitness difference curve, with antibiotic
  
  real low_fitness_std;       // fitness difference at +infinite gene expression, with antibiotic
  real mid_g_std;             // gene expression level at 1/2 max fitness difference, with antibiotic
  real fitness_n_std;         // cooperativity coefficient of fitness difference curve, with antibiotic
  
  array[3] real log_x_max;           // maximum possible value for log_ec50, previously set to log10(max(x)) + 1.4
  
}

transformed data {
  real x_min;
  real x_max;
  
  real high_fitness;
  
  real log_x_1_min;
  real log_x_1_max;
  real log_x_2_min;
  real log_x_2_max;
  real log_x_3_min;
  real log_x_3_max;
  
  high_fitness = 0;
  
  x_max = max(x_1);
  x_min = min(x_1);
  log_x_1_max = log_x_max[1];
  log_x_1_min = log10(x_min) - 0.3;
  
  x_max = max(x_2);
  x_min = min(x_2);
  log_x_2_max = log_x_max[2];
  log_x_2_min = log10(x_min) - 0.3;
  
  x_max = max(x_3);
  x_min = min(x_3);
  log_x_3_max = log_x_max[3];
  log_x_3_min = log10(x_min) - 0.3;
  
}

parameters {
  real<lower=log_g_min, upper=log_g_max> log_g0;        // log10 of gene expression level at zero ligand
  
  real<lower=log_g_min, upper=log_g_max>  log_ginf_1;   // log10 of gene expression level at saturating concentration of ligand 1
  real<lower=log_g_min, upper=log_g_max>  log_ginf_2;   // log10 of gene expression level at saturating concentration of ligand 2
  real<lower=log_g_min, upper=log_g_max>  log_ginf_3;   // log10 of gene expression level at saturating concentration of ligand 3
  
  real<lower=log_x_1_min, upper=log_x_1_max> log_ec50_1;    // input level (x) that gives output 1/2 way between g0 and ginf for ligand 1
  real<lower=log_x_2_min, upper=log_x_2_max> log_ec50_2;    // input level (x) that gives output 1/2 way between g0 and ginf for ligand 2
  real<lower=log_x_3_min, upper=log_x_3_max> log_ec50_3;    // input level (x) that gives output 1/2 way between g0 and ginf for ligand 3
  
  real<lower=0> sensor_n_1;                             // cooperativity exponent of sensor gene expression vs. x curve for ligand 1
  real<lower=0> sensor_n_2;                             // cooperativity exponent of sensor gene expression vs. x curve for ligand 2
  real<lower=0> sensor_n_3;                             // cooperativity exponent of sensor gene expression vs. x curve for ligand 3
  
  real<lower=0> sigma;            // scale factor for standard deviation of noise in y
  
  real low_fitness;       // fitness difference at zero gene expression, med tet
  real mid_g;             // gene expression level at 1/2 max fitness difference, med tet
  real fitness_n;         // cooperativity coefficient of fitness difference curve, med tet
}

transformed parameters {
  real g0;
  real ginf_1; 
  real ec50_1;
  real ginf_2; 
  real ec50_2;
  real ginf_3; 
  real ec50_3;
  
  real mean_y_0;
  
  vector[N_lig] mean_y_1;
  vector[N_lig] g_1;                // gene expression level at each non-zero concentration of ligand 1
  vector[N_lig] mean_y_2;
  vector[N_lig] g_2;                // gene expression level at each non-zero concentration of ligand 2
  vector[N_lig] mean_y_3;
  vector[N_lig] g_3;                // gene expression level at each non-zero concentration of ligand 3
  
  g0 = 10^log_g0;
  ginf_1 = 10^log_ginf_1;
  ec50_1 = 10^log_ec50_1;
  ginf_2 = 10^log_ginf_2;
  ec50_2 = 10^log_ec50_2;
  ginf_3 = 10^log_ginf_3;
  ec50_3 = 10^log_ec50_3;
  
  mean_y_0 = low_fitness + (high_fitness - low_fitness)*(g0^fitness_n)/(mid_g^fitness_n + g0^fitness_n);
  
  for (i in 1:N_lig) {
    g_1[i] = g0 + (ginf_1 - g0)*(x_1[i]^sensor_n_1)/(ec50_1^sensor_n_1 + x_1[i]^sensor_n_1);
    g_2[i] = g0 + (ginf_2 - g0)*(x_2[i]^sensor_n_2)/(ec50_2^sensor_n_2 + x_2[i]^sensor_n_2);
    g_3[i] = g0 + (ginf_3 - g0)*(x_3[i]^sensor_n_3)/(ec50_3^sensor_n_3 + x_3[i]^sensor_n_3);
	
    mean_y_1[i] = low_fitness + (high_fitness - low_fitness)*(g_1[i]^fitness_n)/(mid_g^fitness_n + g_1[i]^fitness_n);
    mean_y_2[i] = low_fitness + (high_fitness - low_fitness)*(g_2[i]^fitness_n)/(mid_g^fitness_n + g_2[i]^fitness_n);
    mean_y_3[i] = low_fitness + (high_fitness - low_fitness)*(g_3[i]^fitness_n)/(mid_g^fitness_n + g_3[i]^fitness_n);
  }
  
}

model {
  
  low_fitness ~ normal(low_fitness_mu, low_fitness_std);
  mid_g ~ normal(mid_g_mu, mid_g_std);
  fitness_n ~ normal(fitness_n_mu, fitness_n_std);
  
  // Prior on sensor_n; <weibull> ~ 2
  sensor_n_1 ~ weibull(4, 2.2);
  sensor_n_2 ~ weibull(4, 2.2);
  sensor_n_3 ~ weibull(4, 2.2);
  
  // Prior on log_ec50; flat prior with erf boundaries
  target += log1m(erf((log_x_1_min + 0.3 - log_ec50_1)/0.2));
  target += log1m(erf((log_ec50_1 - log_x_1_max + 0.3)/0.3));
  
  target += log1m(erf((log_x_2_min + 0.3 - log_ec50_2)/0.2));
  target += log1m(erf((log_ec50_2 - log_x_2_max + 0.3)/0.3));
  
  target += log1m(erf((log_x_3_min + 0.3 - log_ec50_3)/0.2));
  target += log1m(erf((log_ec50_3 - log_x_3_max + 0.3)/0.3));
  
  // noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);
  
  y_0 ~ normal(mean_y_0, sigma*y_0_err);
  
  y_1 ~ normal(mean_y_1, sigma*y_1_err);
  y_2 ~ normal(mean_y_2, sigma*y_2_err);
  y_3 ~ normal(mean_y_3, sigma*y_3_err);
  
}

generated quantities {
  real log_sensor_n_1;
  real log_sensor_n_2;
  real log_sensor_n_3;
  
  real log_ginf_g0_ratio_1;
  real log_ginf_g0_ratio_2;
  real log_ginf_g0_ratio_3;
  
  real spec_1;  //log_ec50-based specificity for each ligand
  real spec_2;
  real spec_3;
  real mean_log_ec50;  //mean log_ec50 for all three ligands
  
  real rms_resid;
  
  spec_1 = -log_ec50_1 + (log_ec50_2 + log_ec50_3)/2;
  spec_2 = -log_ec50_2 + (log_ec50_1 + log_ec50_3)/2;
  spec_3 = -log_ec50_3 + (log_ec50_1 + log_ec50_2)/2;
  mean_log_ec50 = (log_ec50_1 + log_ec50_2 + log_ec50_3)/3;
  
  log_ginf_g0_ratio_1 = log_ginf_1 - log_g0;
  log_ginf_g0_ratio_2 = log_ginf_2 - log_g0;
  log_ginf_g0_ratio_3 = log_ginf_3 - log_g0;
  
  log_sensor_n_1 = log10(sensor_n_1);
  log_sensor_n_2 = log10(sensor_n_2);
  log_sensor_n_3 = log10(sensor_n_3);
  
  rms_resid = sqrt((y_0[1] - mean_y_0)^2 + (y_0[2] - mean_y_0)^2 + distance(y_1, mean_y_1)^2 + distance(y_2, mean_y_2)^2 + distance(y_3, mean_y_3)^2)/sqrt(3*N_lig + 2);
  
}
