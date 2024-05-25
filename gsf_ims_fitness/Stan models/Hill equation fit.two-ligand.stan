// 
//

data {
  int<lower=1> N_reps;                // number of replicate measurements
  
  int<lower=1> N_0;                   // number of data points with zero ligand, per replicate
  int<lower=1> N_1;                   // number of non-zero ligand concentrations for ligand 1, per replicate
  int<lower=1> N_2;                   // number of non-zero ligand concentrations for ligand 2, per replicate
  
  vector[N_1] x_1;                    // non-zero ligand 1 concentrations
  vector[N_2] x_2;                    // non-zero ligand 2 concentrations
  
  array[N_reps] vector[N_0] y_0;      // measured response at zero ligand
  array[N_reps] vector[N_0] y_0_err;  // experimental uncertainty of zero-ligand response
  
  array[N_reps] vector[N_1] y_1;      // measured response with ligand 1
  array[N_reps] vector[N_1] y_1_err;  // experimental uncertainty of ligand 1 response
  array[N_reps] vector[N_2] y_2;      // measured response with ligand 2
  array[N_reps] vector[N_2] y_2_err;  // experimental uncertainty of ligand 2 response
  
  real log_g_min;                     // lower bound on log_g0 and log_ginf
  real log_g_max;                     // upper bound on log_g0 and log_ginf
  
  array[2] real log_ec50_min;         // minimum possible value for log_ec50 for each ligand
  array[2] real log_ec50_max;         // maximum possible value for log_ec50 for each ligand
  
  real<lower=0> sensor_n_1_alpha;     // prior shape for n, ligand 1
  real<lower=0> sensor_n_1_sigma;     // prior inverse scale, ligand 1
  real<lower=0> sensor_n_2_alpha;     // prior shape for n, ligand 2
  real<lower=0> sensor_n_2_sigma;     // prior inverse scale, ligand 2
  
  array[N_reps] real baseline_mu;         // prior mean of non-fluorescent control, can be different for each rep
  real baseline_sig;                  // prior sigma of non-fluorescent control
}

transformed data {
  real x_1_max;
  real x_2_max;
  
  x_1_max = max(x_1);
  x_2_max = max(x_2);
}

parameters {
  array[N_reps] real baseline;                               //non-fluorescent control y-value
  
  real<lower=log_g_min, upper=log_g_max> log_g0;        // log10 of gene expression level at zero ligand
  
  real<lower=log_g_min, upper=log_g_max>  log_ginf_1;   // log10 of gene expression level at saturating concentration of ligand 1
  real<lower=log_g_min, upper=log_g_max>  log_ginf_2;   // log10 of gene expression level at saturating concentration of ligand 2
  
  real<lower=log_ec50_min[1], upper=log_ec50_max[1]> log_ec50_1;    // input level (x) that gives output 1/2 way between g0 and ginf for ligand 1
  real<lower=log_ec50_min[2], upper=log_ec50_max[2]> log_ec50_2;    // input level (x) that gives output 1/2 way between g0 and ginf for ligand 2
  
  real<lower=0> sensor_n_1;                             // cooperativity exponent of sensor gene expression vs. x curve for ligand 1
  real<lower=0> sensor_n_2;                             // cooperativity exponent of sensor gene expression vs. x curve for ligand 2
  
  real<lower=0> sigma;            // scale factor for standard deviation of noise in y
}

transformed parameters {
  real g0;               // gene expression level at zero ligand
  real ginf_1; 
  real ec50_1;
  real ginf_2; 
  real ec50_2;
  
  vector[N_1] g_1;          // gene expression level at each non-zero concentration of ligand 1
  vector[N_2] g_2;          // gene expression level at each non-zero concentration of ligand 2
  
  g0 = 10^log_g0;
  ginf_1 = 10^log_ginf_1;
  ec50_1 = 10^log_ec50_1;
  ginf_2 = 10^log_ginf_2;
  ec50_2 = 10^log_ec50_2;
  
   
  for (i in 1:N_1) {
    g_1[i] = g0 + (ginf_1 - g0)*(x_1[i]^sensor_n_1)/(ec50_1^sensor_n_1 + x_1[i]^sensor_n_1);
  }
  for (i in 1:N_2) {
    g_2[i] = g0 + (ginf_2 - g0)*(x_2[i]^sensor_n_2)/(ec50_2^sensor_n_2 + x_2[i]^sensor_n_2);
  }
  
}

model {  
  // Prior on sensor_n; <weibull> ~ sensor_n_sigma*GammaFunct(1+1/sensor_n_alpha)
  sensor_n_1 ~ weibull(sensor_n_1_alpha, sensor_n_1_sigma);
  sensor_n_2 ~ weibull(sensor_n_2_alpha, sensor_n_2_sigma);
  
  // noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);
  
  for (n in 1:N_reps) {
    // prior on baseline, informed by measurements of control strain
    baseline[n] ~ normal(baseline_mu[n], baseline_sig);
  
    y_0[n] ~ normal(g0 + baseline[n], sigma*y_0_err[n]);
    
    y_1[n] ~ normal(g_1 + baseline[n], sigma*y_1_err[n]);
    y_2[n] ~ normal(g_2 + baseline[n], sigma*y_2_err[n]);
  }
  
}

generated quantities {
  real log_sensor_n_1;
  real log_sensor_n_2;
  
  real log_gmax_1;
  real log_gmax_2;
  
  real log_ginf_g0_ratio_1;
  real log_ginf_g0_ratio_2;
  
  real rms_resid;
  array[N_reps] real rep_resid;
  
  vector[N_0] g0_vect;
  
  for (i in 1:N_0) {
    g0_vect[i] = g0;
  }
  
  
  log_ginf_g0_ratio_1 = log_ginf_1 - log_g0;
  log_ginf_g0_ratio_2 = log_ginf_2 - log_g0;
  
  log_sensor_n_1 = log10(sensor_n_1);
  log_sensor_n_2 = log10(sensor_n_2);
  
  rms_resid = 0;
  for (n in 1:N_reps) {
    rep_resid[n] = sqrt(distance(y_0[n], g0_vect + baseline[n])^2 + distance(y_1[n], g_1 + baseline[n])^2 + distance(y_2[n], g_2 + baseline[n])^2)/sqrt(N_0 + N_1 + N_2);
	
	rms_resid = rms_resid + rep_resid[n]^2;
  }
  
  rms_resid = sqrt(rms_resid);
  
  log_gmax_1 = log10(g0 + (ginf_1 - g0)*(x_1_max^sensor_n_1)/(ec50_1^sensor_n_1 + x_1_max^sensor_n_1));
  log_gmax_2 = log10(g0 + (ginf_2 - g0)*(x_2_max^sensor_n_2)/(ec50_2^sensor_n_2 + x_2_max^sensor_n_2));
  
}
