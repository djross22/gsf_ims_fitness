// Fit dose-response curves to Phillips lab model for allosteric TFs
//

data {
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // gene expression (from cytometry) at each concentration
  vector[N] y_err;       // estimated error of gene expression at each concentration
  real y_max;            // geometric mean for prior on maximum gene expression value
  
  int rep[N];            // integer to indicate the measurement replicate
  int<lower=1> num_reps; // number of measurement replicates
  
  real rep_ratio_scale;   // parameter to set the scale for the half-normal prior on log_rep_ratio
  real rep_offset_scale;  // parameter to set the scale for the half-normal prior on log_rep_ratio

}

transformed data {
  real R;
  real n;
  real N_NS;
  
  R = 200;
  n = 2;
  N_NS = 4600000;
}

parameters {
  real log_k_a;         // log10 of IPTG binding affinity to active state
  real log_k_i;         // log10 of IPTG binding affinity to inactive state
  real delta_eps_AI;    // free energy difference between active and inactive states
  real delta_eps_RA;    // free energy for Active TF to operator
  
  real log_g_max;       // log10 of maximum possible gene expression
  
  real<lower=0> sigma;  // scale factor for standard deviation of noise in y
  
  vector<lower=-3*rep_ratio_scale, upper=3*rep_ratio_scale>[num_reps] log_rep_ratio;  // log10 of multiplicative correction factor for different replicates
  vector<lower=-3*rep_offset_scale, upper=3*rep_offset_scale>[num_reps] rep_offset;   // offset for different replicates
  // hyper-paramters for log_rep_ratio and rep_offset
  real<lower=0> rep_ratio_sigma;
  real<lower=0> rep_offset_sigma;
}

transformed parameters {
  real K_A;        
  real K_I;
  real g_max;
  vector[num_reps] rep_ratio;
  
  K_A = 10^log_k_a;
  K_I = 10^log_k_i;
  g_max = 10^log_g_max;
  
  for (j in 1:num_reps) {
    rep_ratio[j] = 10^log_rep_ratio[j];
  }
  
}

model {
  // Local variables
  vector[N] mean_y;
  real c1;
  real c2;
  real c3;
  
  c3 = R/N_NS * exp(-delta_eps_RA);

  for (i in 1:N) {
    c1 = (1 + x[i]/K_A)^n;
    c2 = ( (1 + x[i]/K_I)^n ) * exp(-delta_eps_AI);
    mean_y[i] = g_max*rep_ratio[rep[i]]/(1 + (c1/(c1+c2))*c3) + rep_offset[rep[i]];
  }
  
  // Priors on params
  log_k_a ~ normal(2.14, 0.3);
  log_k_i ~ normal(-0.28, 0.3);
  delta_eps_AI ~ normal(4.5, 0.6);
  delta_eps_RA ~ normal(-13.9, 3);
  
  // priors on scale hyper-paramters for log_rep_ratio and rep_offset
  rep_ratio_sigma ~ normal(0, rep_ratio_scale);
  rep_offset_sigma ~ normal(0, rep_offset_scale);
  
  // prior on max output level
  log_g_max ~ normal(log10(y_max), 0.5);
  
  // priors on scale hyper-paramters for log_rep_ratio and rep_offset
  rep_ratio_sigma ~ normal(0, rep_ratio_scale);
  rep_offset_sigma ~ normal(0, rep_offset_scale);
  
  // priors on log_rep_ratio and rep_offset
  log_rep_ratio ~ normal(0, rep_ratio_sigma);
  rep_offset ~ normal(0, rep_offset_sigma);
  
  y ~ normal(mean_y, sigma*y_err);

}

generated quantities {
  
}
