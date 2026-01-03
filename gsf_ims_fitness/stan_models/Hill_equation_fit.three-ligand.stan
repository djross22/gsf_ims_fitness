// 
//

data {
  int<lower=1> N_0;                // number of data points with zero ligand
  int<lower=1> N_1;                // number of non-zero ligand concentrations for ligand 1
  int<lower=1> N_2;                // number of non-zero ligand concentrations for ligand 2
  int<lower=1> N_3;                // number of non-zero ligand concentrations for ligand 3
  
  vector[N_1] x_1;                 // non-zero ligand 1 concentrations
  vector[N_2] x_2;                 // non-zero ligand 2 concentrations
  vector[N_3] x_3;                 // non-zero ligand 3 concentrations
  
  vector[N_0] y_0;                  // measured response at zero ligand
  vector[N_0] y_0_err;              // experimental uncertainty of zero-ligand response
  
  vector[N_1] y_1;                  // measured response with ligand 1
  vector[N_1] y_1_err;              // experimental uncertainty of ligand 1 response
  vector[N_2] y_2;                  // measured response with ligand 2
  vector[N_2] y_2_err;              // experimental uncertainty of ligand 2 response
  vector[N_3] y_3;                  // measured response with ligand 3
  vector[N_3] y_3_err;              // experimental uncertainty of ligand 3 response
  
  real log_g_min;                   // lower bound on log_g0 and log_ginf
  real log_g_max;                   // upper bound on log_g0 and log_ginf
  
  array[3] real log_ec50_min;       // minimum possible value for log_ec50 for each ligand
  array[3] real log_ec50_max;       // maximum possible value for log_ec50 for each ligand
  
  real<lower=0> sensor_n_alpha;     //prior shape
  real<lower=0> sensor_n_sigma;      //prior inverse scale
  
  real baseline_mu;             // mean value of non-fluorescent control
  real baseline_sig;            //uncertainty of non-fluorescent control
}

transformed data {
  
}

parameters {
  real baseline;                                        //non-fluorescent control 
  real<lower=log_g_min, upper=log_g_max> log_g0;        // log10 of gene expression level at zero ligand
  
  real<lower=log_g_min, upper=log_g_max>  log_ginf_1;   // log10 of gene expression level at saturating concentration of ligand 1
  real<lower=log_g_min, upper=log_g_max>  log_ginf_2;   // log10 of gene expression level at saturating concentration of ligand 2
  real<lower=log_g_min, upper=log_g_max>  log_ginf_3;   // log10 of gene expression level at saturating concentration of ligand 3
  
  real<lower=log_ec50_min[1], upper=log_ec50_max[1]> log_ec50_1;    // input level (x) that gives output 1/2 way between g0 and ginf for ligand 1
  real<lower=log_ec50_min[2], upper=log_ec50_max[2]> log_ec50_2;    // input level (x) that gives output 1/2 way between g0 and ginf for ligand 2
  real<lower=log_ec50_min[3], upper=log_ec50_max[3]> log_ec50_3;    // input level (x) that gives output 1/2 way between g0 and ginf for ligand 3
  
  real<lower=0> sensor_n_1;                             // cooperativity exponent of sensor gene expression vs. x curve for ligand 1
  real<lower=0> sensor_n_2;                             // cooperativity exponent of sensor gene expression vs. x curve for ligand 2
  real<lower=0> sensor_n_3;                             // cooperativity exponent of sensor gene expression vs. x curve for ligand 3
  
  real<lower=0> sigma;            // scale factor for standard deviation of noise in y
}

transformed parameters {
  real g0;               // gene expression level at zero ligand
  real ginf_1; 
  real ec50_1;
  real ginf_2; 
  real ec50_2;
  real ginf_3; 
  real ec50_3;
  
  vector[N_1] g_1;          // gene expression level at each non-zero concentration of ligand 1
  vector[N_2] g_2;          // gene expression level at each non-zero concentration of ligand 2
  vector[N_3] g_3;          // gene expression level at each non-zero concentration of ligand 3
  
  g0 = 10^log_g0;
  ginf_1 = 10^log_ginf_1;
  ec50_1 = 10^log_ec50_1;
  ginf_2 = 10^log_ginf_2;
  ec50_2 = 10^log_ec50_2;
  ginf_3 = 10^log_ginf_3;
  ec50_3 = 10^log_ec50_3;
  
   
  for (i in 1:N_1) {
    g_1[i] = g0 + (ginf_1 - g0)*(x_1[i]^sensor_n_1)/(ec50_1^sensor_n_1 + x_1[i]^sensor_n_1);
  }
  for (i in 1:N_2) {
    g_2[i] = g0 + (ginf_2 - g0)*(x_2[i]^sensor_n_2)/(ec50_2^sensor_n_2 + x_2[i]^sensor_n_2);
  }
  for (i in 1:N_3) {
    g_3[i] = g0 + (ginf_3 - g0)*(x_3[i]^sensor_n_3)/(ec50_3^sensor_n_3 + x_3[i]^sensor_n_3);
  }
  
}

model {
  // prior on baseline, informed by measurements of control strain
  baseline ~ normal(baseline_mu, baseline_sig);
  
  // Prior on sensor_n; <weibull> ~ sensor_n_sigma*GammaFunct(1+1/sensor_n_alpha)
  sensor_n_1 ~ weibull(sensor_n_alpha, sensor_n_sigma);
  sensor_n_2 ~ weibull(sensor_n_alpha, sensor_n_sigma);
  sensor_n_3 ~ weibull(sensor_n_alpha, sensor_n_sigma);
  
  // noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);
  
  y_0 ~ normal(g0 + baseline, sigma*y_0_err);
  
  y_1 ~ normal(g_1 + baseline, sigma*y_1_err);
  y_2 ~ normal(g_2 + baseline, sigma*y_2_err);
  y_3 ~ normal(g_3 + baseline, sigma*y_3_err);
  
}

generated quantities {
  real log_sensor_n_1;
  real log_sensor_n_2;
  real log_sensor_n_3;
  
  real log_ginf_g0_ratio_1;
  real log_ginf_g0_ratio_2;
  real log_ginf_g0_ratio_3;
  
  real rms_resid;
  
  vector[N_0] g0_vect;
  
  for (i in 1:N_0) {
    g0_vect[i] = g0;
  }
  
  
  log_ginf_g0_ratio_1 = log_ginf_1 - log_g0;
  log_ginf_g0_ratio_2 = log_ginf_2 - log_g0;
  log_ginf_g0_ratio_3 = log_ginf_3 - log_g0;
  
  log_sensor_n_1 = log10(sensor_n_1);
  log_sensor_n_2 = log10(sensor_n_2);
  log_sensor_n_3 = log10(sensor_n_3);
  
  rms_resid = sqrt(distance(y_0, g0_vect + baseline)^2 + distance(y_1, g_1 + baseline)^2 + distance(y_2, g_2 + baseline)^2 + distance(y_3, g_3 + baseline)^2)/sqrt(N_0 + N_1 + N_2 + N_3);
  
}
