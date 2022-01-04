// Fit dose-response curves to Phillips lab model for allosteric TFs
//

data {
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // gene expression (from cytometry) at each concentration
  vector[N] y_err;       // estimated error of gene expression at each concentration
  real y_max;            // geometric mean for prior on maximum gene expression value
  
  int rep[N];            // integer to indicate the measurement replicate
  int<lower=1> num_reps; // number of measurement replicates (for all variants)
  
  int<lower=2> num_var;  // number of variants
  int variant[N];        // numerical index to indicate variants
  int<lower=0> num_epi_var;  // number of variants with more than one mutation (only define epistasis for these)
  
  int<lower=1> num_mut;  // number of differrent mutations
  int<lower=0, upper=1> mut_code[num_var-1, num_mut];   // one-hot encoding for  presense of mutations in each variant; variant 0 (WT) has no mutations
  
  real delta_prior_width; // width of prior on delta-parameters
  real epi_prior_width;   // width of prior on parameter epistasis

}

transformed data {
  real R;
  real hill_n;
  real N_NS;
  vector[16] x_out;
  int num_non_epi_var;  // number of variants with less than two mutations
  
  R = 200;
  hill_n = 2;
  N_NS = 4600000;
  
  x_out[1] = 0;
  for (i in 2:16) {
    x_out[i] = 2^(i-2);
  }
  
  num_non_epi_var = num_var - num_epi_var;
}

parameters {
  // In this version of the model, the base parameters belong to the first variant (the wild-type)
  //     and there is a delta_param associated with each mutation (with additive effects)
  //     plus an epistasis term associated with each variant other than the wild-type
  real log_k_a_wt;         // log10 of IPTG binding affinity to active state
  vector[num_mut] log_k_a_mut;
  vector[num_epi_var] log_k_a_epi;
  
  real log_k_i_wt;         // log10 of IPTG binding affinity to inactive state
  vector[num_mut] log_k_i_mut;
  vector[num_epi_var] log_k_i_epi;
  
  real delta_eps_AI_wt;    // free energy difference between active and inactive states
  vector[num_mut] delta_eps_AI_mut;
  vector[num_epi_var] delta_eps_AI_epi;
  
  real delta_eps_RA_wt;    // free energy for Active TF to operator
  vector[num_mut] delta_eps_RA_mut;
  vector[num_epi_var] delta_eps_RA_epi;
  
  real log_g_max;       // log10 of maximum possible gene expression
  
  real<lower=0> sigma;  // scale factor for standard deviation of noise in y
  
  vector<lower=-0.03, upper=0.03>[num_reps] log_rep_ratio;  // log10 of multiplicative correction factor for different replicates
  vector<lower=-50, upper=50>[num_reps] rep_offset;         // offset for different replicates
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
  
  log_k_a_var[1] = log_k_a_wt;
  log_k_i_var[1] = log_k_i_wt;
  delta_eps_AI_var[1] = delta_eps_AI_wt;
  delta_eps_RA_var[1] = delta_eps_RA_wt;
  
  K_A[1] = 10^log_k_a_var[1];
  K_I[1] = 10^log_k_i_var[1];
  
  for (var in 2:num_var) {
    if (var > num_non_epi_var) {
      log_k_a_var[var] = log_k_a_wt + log_k_a_epi[var-num_non_epi_var];
      log_k_i_var[var] = log_k_i_wt + log_k_i_epi[var-num_non_epi_var];
      delta_eps_AI_var[var] = delta_eps_AI_wt + delta_eps_AI_epi[var-num_non_epi_var];
      delta_eps_RA_var[var] = delta_eps_RA_wt + delta_eps_RA_epi[var-num_non_epi_var];
	}
    else {
      log_k_a_var[var] = log_k_a_wt;
      log_k_i_var[var] = log_k_i_wt;
      delta_eps_AI_var[var] = delta_eps_AI_wt;
      delta_eps_RA_var[var] = delta_eps_RA_wt;
	}
	
	for (mut in 1:num_mut) {
	  log_k_a_var[var] += mut_code[var-1, mut]*log_k_a_mut[mut];
	  log_k_i_var[var] += mut_code[var-1, mut]*log_k_i_mut[mut];
	  delta_eps_AI_var[var] += mut_code[var-1, mut]*delta_eps_AI_mut[mut];
	  delta_eps_RA_var[var] += mut_code[var-1, mut]*delta_eps_RA_mut[mut];
	}
	
	K_A[var] = 10^log_k_a_var[var];
    K_I[var] = 10^log_k_i_var[var];
  }
  
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
  
  for (i in 1:N) {
    c1 = (1 + x[i]/K_A[variant[i]])^hill_n;
    c2 = ( (1 + x[i]/K_I[variant[i]])^hill_n ) * exp(-delta_eps_AI_var[variant[i]]);
    c3 = R/N_NS * exp(-delta_eps_RA_var[variant[i]]);
	
    mean_y[i] = g_max*rep_ratio[rep[i]]/(1 + (c1/(c1+c2))*c3) + rep_offset[rep[i]];
  }
  
  // Priors on params
  log_k_a_wt ~ normal(2.14, 0.3);
  log_k_i_wt ~ normal(-0.28, 0.3);
  delta_eps_AI_wt ~ normal(4.5, 0.6);
  delta_eps_RA_wt ~ normal(-13.9, 3);
  
  log_k_a_mut ~ normal(0, delta_prior_width/2);
  log_k_a_epi ~ normal(0, epi_prior_width);
  
  log_k_i_mut ~ normal(0, delta_prior_width/2);
  log_k_i_epi ~ normal(0, epi_prior_width);
  
  delta_eps_AI_mut ~ normal(0, delta_prior_width);
  delta_eps_AI_epi ~ normal(0, epi_prior_width);
  
  delta_eps_RA_mut ~ normal(0, delta_prior_width);
  delta_eps_RA_epi ~ normal(0, epi_prior_width);
  
  log_g_max ~ normal(log10(y_max), 0.05);
  
  log_rep_ratio ~ normal(0, 0.015);
  rep_offset ~ normal(0, 25);
  
  y ~ normal(mean_y, sigma*y_err);

}

generated quantities {
  // Local variables
  real y_out[num_var, 16];
  real mean_offset;
  real geo_mean_ratio;
  
  mean_offset = mean(rep_offset);
  geo_mean_ratio = 10^mean(log_rep_ratio);
  
  for (var in 1:num_var) {
    for (i in 1:16) {
	  real c1;
      real c2;
      real c3;
	  
      c1 = (1 + x_out[i]/K_A[var])^hill_n;
      c2 = ( (1 + x_out[i]/K_I[var])^hill_n ) * exp(-delta_eps_AI_var[var]);
      c3 = R/N_NS * exp(-delta_eps_RA_var[var]);
	
      y_out[var, i] = g_max*geo_mean_ratio/(1 + (c1/(c1+c2))*c3) + mean_offset;
    }
  }
  
}
