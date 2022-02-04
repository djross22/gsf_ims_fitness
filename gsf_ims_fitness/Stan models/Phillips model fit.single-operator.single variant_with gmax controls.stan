// Fit dose-response curves to Phillips lab model for allosteric TFs
//

data {
  int<lower=1> N_contr;          // number of data points for control strains LacI deletions
  vector[N_contr] y_contr;       // gene expression (from cytometry) for control strains
  vector[N_contr] y_contr_err;   // estimated error of gene expression for control strains
  
  int rep_contr[N_contr];        // integer to indicate the measurement replicate for controls
  int<lower=1> num_contr_reps;   // number of measurement replicates for controls
  
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // gene expression (from cytometry) at each concentration
  vector[N] y_err;       // estimated error of gene expression at each concentration
  real y_max;            // geometric mean for prior on maximum gene expression value
  real g_max_prior_width; // geometric std for prior on maximum gene expression value
  
  int rep[N];            // integer to indicate the measurement replicate
  int<lower=1> num_reps; // number of measurement replicates (for all variants)
  
  int variant[N];        // numerical index to indicate variants
  
  // priors on wild-type free energy parameters
  real log_k_a_wt_prior_mean;
  real log_k_a_wt_prior_std;
  real log_k_i_wt_prior_mean;
  real log_k_i_wt_prior_std;
  real delta_eps_AI_wt_prior_mean;
  real delta_eps_AI_wt_prior_std;
  real delta_eps_RA_wt_prior_mean;
  real delta_eps_RA_wt_prior_std;
  
  real rep_ratio_scale;   // parameter to set the scale for the half-normal prior on log_rep_ratio
  real rep_offset_scale;  // parameter to set the scale for the half-normal prior on log_rep_ratio
}

transformed data {
  real R;
  real hill_n;
  real N_NS;
  vector[19] x_out;
  int num_var;           // number of variants
  
  num_var = 1;
  
  R = 200;
  hill_n = 2;
  N_NS = 4600000;
  
  x_out[1] = 0;
  for (i in 2:19) {
    x_out[i] = 2^(i-2);
  }
  
}

parameters {
  // In this version of the model, the base parameters belong to the first variant (the wild-type)
  //     and there is a delta_param associated with each mutation (with additive effects)
  //     plus an epistasis term associated with each variant other than the wild-type
  real log_k_a_wt;         // log10 of IPTG binding affinity to active state
  
  real log_k_i_wt;         // log10 of IPTG binding affinity to inactive state
  
  real delta_eps_AI_wt;    // free energy difference between active and inactive states
  
  real delta_eps_RA_wt;    // free energy for Active TF to operator
  
  real log_g_max;       // log10 of maximum possible gene expression
  
  real<lower=0> sigma;  // scale factor for standard deviation of noise in y
  
  vector<lower=-3*rep_ratio_scale, upper=3*rep_ratio_scale>[num_reps] log_rep_ratio;  // log10 of multiplicative correction factor for different replicates
  vector<lower=-3*rep_offset_scale, upper=3*rep_offset_scale>[num_reps] rep_offset;   // offset for different replicates
  
  vector<lower=-3*rep_ratio_scale, upper=3*rep_ratio_scale>[num_contr_reps] log_rep_ratio_contr;  // log10 of multiplicative correction factor for control replicates
  vector<lower=-3*rep_offset_scale, upper=3*rep_offset_scale>[num_contr_reps] rep_offset_contr;   // offset for control replicates
  
  // hyper-paramters for log_rep_ratio and rep_offset
  real<lower=0> rep_ratio_sigma;
  real<lower=0> rep_offset_sigma;
}

transformed parameters {
  vector[num_var] K_A;
  vector[num_var] K_I;
  vector[num_var] log_k_a_var;
  vector[num_var] log_k_i_var;
  vector[num_var] delta_eps_AI_var;
  vector[num_var] delta_eps_RA_var;
  
  real g_max;
  
  vector[num_reps] rep_ratio;
  vector[num_contr_reps] rep_ratio_contr;
  
  vector[N] mean_y;
  vector[N_contr] mean_y_contr;
  
  log_k_a_var[1] = log_k_a_wt;
  log_k_i_var[1] = log_k_i_wt;
  delta_eps_AI_var[1] = delta_eps_AI_wt;
  delta_eps_RA_var[1] = delta_eps_RA_wt;
  
  K_A[1] = 10^log_k_a_var[1];
  K_I[1] = 10^log_k_i_var[1];
  
  g_max = 10^log_g_max;
  
  for (j in 1:num_reps) {
    rep_ratio[j] = 10^log_rep_ratio[j];
  }
  for (j in 1:num_contr_reps) {
    rep_ratio_contr[j] = 10^log_rep_ratio_contr[j];
  }
  
  for (i in 1:N) {
    real c1;
    real c2;
    real c3;
	
    c1 = (1 + x[i]/K_A[variant[i]])^hill_n;
    c2 = ( (1 + x[i]/K_I[variant[i]])^hill_n ) * exp(-delta_eps_AI_var[variant[i]]);
    c3 = R/N_NS * exp(-delta_eps_RA_var[variant[i]]);
	
    mean_y[i] = g_max*rep_ratio[rep[i]]/(1 + (c1/(c1+c2))*c3) + rep_offset[rep[i]];
  }
  
  for (i in 1:N_contr) {
    mean_y_contr[i] = g_max*rep_ratio_contr[rep_contr[i]] + rep_offset_contr[rep_contr[i]];
  }
  
}

model {
  // priors on free energy params
  log_k_a_wt ~ normal(log_k_a_wt_prior_mean, log_k_a_wt_prior_std);
  log_k_i_wt ~ normal(log_k_i_wt_prior_mean, log_k_i_wt_prior_std);
  delta_eps_AI_wt ~ normal(delta_eps_AI_wt_prior_mean, delta_eps_AI_wt_prior_std);
  delta_eps_RA_wt ~ normal(delta_eps_RA_wt_prior_mean, delta_eps_RA_wt_prior_std);
  
  // prior on max output level
  log_g_max ~ normal(log10(y_max), g_max_prior_width);
  
  // priors on scale hyper-paramters for log_rep_ratio and rep_offset
  rep_ratio_sigma ~ normal(0, rep_ratio_scale);
  rep_offset_sigma ~ normal(0, rep_offset_scale);
  
  // priors on log_rep_ratio and rep_offset
  log_rep_ratio ~ normal(0, rep_ratio_sigma);
  rep_offset ~ normal(0, rep_offset_sigma);
  log_rep_ratio_contr ~ normal(0, rep_ratio_sigma);
  rep_offset_contr ~ normal(0, rep_offset_sigma);
  
  // model of the data (dose-response curve with noise)
  y ~ normal(mean_y, sigma*y_err);
  
  // model of the control strain data (constant, max output)
  y_contr ~ normal(mean_y_contr, sigma*y_contr_err);

}

generated quantities {
  // Local variables
  real y_out[num_var, 19];
  real fc_out[num_var, 19];
  real mean_offset;
  real geo_mean_ratio;
  
  mean_offset = mean(rep_offset);
  geo_mean_ratio = 10^mean(log_rep_ratio);
  
  for (var in 1:num_var) {
    for (i in 1:19) {
	  real c1;
      real c2;
      real c3;
	  
      c1 = (1 + x_out[i]/K_A[var])^hill_n;
      c2 = ( (1 + x_out[i]/K_I[var])^hill_n ) * exp(-delta_eps_AI_var[var]);
      c3 = R/N_NS * exp(-delta_eps_RA_var[var]);
	
      y_out[var, i] = g_max*geo_mean_ratio/(1 + (c1/(c1+c2))*c3) + mean_offset;
      fc_out[var, i] = 1/(1 + (c1/(c1+c2))*c3);
    }
  }
  
}
