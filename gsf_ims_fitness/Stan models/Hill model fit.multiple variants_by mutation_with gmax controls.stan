// Fit dose-response curves to Phillips lab model for allosteric TFs
//

data {
  int<lower=1> N_contr;          // number of data points for control strains LacI deletions
  vector[N_contr] y_contr;       // gene expression (from cytometry) for control strains
  vector[N_contr] y_contr_err;   // estimated error of gene expression for control strains
  
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // gene expression (from cytometry) at each concentration
  vector[N] y_err;       // estimated error of gene expression at each concentration
  
  real y_max;             // geometric mean for prior on maximum gene expression value
  real g_max_prior_width; // geometric std for prior on maximum gene expression value
  
  int<lower=1> num_var;  // number of variants
  int variant[N];        // numerical index to indicate variants
  int<lower=0> num_epi_var;  // number of variants with more than one mutation (only define epistasis for these)
  
  int<lower=0> num_mut;  // number of differrent mutations
  int<lower=0, upper=1> mut_code[num_var-1, num_mut];   // one-hot encoding for  presense of mutations in each variant; variant 0 (WT) has no mutations
  
  // priors on wild-type Hill parameters
  real logit_g0_wt_prior_mean;
  real logit_g0_wt_prior_std;
  
  real logit_ginf_wt_prior_mean;
  real logit_ginf_wt_prior_std;
  
  real log_ec50_wt_prior_mean;
  real log_ec50_wt_prior_std;
  
  real logit_n_eff_wt_prior_mean;
  real logit_n_eff_wt_prior_std;
  
  real delta_prior_width_hill; // width of prior on delta-parameters
  real epi_prior_width_1_hill;   // width of 1st mixture component of prior on parameter epistasis
  real epi_prior_width_2_hill;   // width of 2nd mixture component of prior on parameter epistasis
  real epi_prior_phi_hill;       // weight for 1st mixture component of prior on parameter epistasis
  
}

transformed data {
  real hill_n;
  vector[19] x_out;
  int num_non_epi_var;  // number of variants with less than two mutations
  real log_phi_1;
  real log_phi_2;
  
  log_phi_1 = log(epi_prior_phi_hill);
  log_phi_2 = log(1 - epi_prior_phi_hill);
  
  hill_n = 2; // Sets upper bound on n_eff
  
  x_out[1] = 0;
  for (i in 2:19) {
    x_out[i] = 2^(i-2);
  }
  
  num_non_epi_var = num_var - num_epi_var;
}

parameters {
  // In this version of the model, the base parameters belong to the first variant (the wild-type)
  //     and there is a delta_param associated with each mutation (with additive effects)
  //     plus an epistasis term associated with each variant other than the wild-type
  
  real logit_g0_wt;                 // logit-transformed g0
  vector[num_mut] logit_g0_mut;
  vector[num_epi_var] logit_g0_epi;
  
  real logit_ginf_wt;               // logit-transformed ginf
  vector[num_mut] logit_ginf_mut;
  vector[num_epi_var] logit_ginf_epi;
  
  real log_ec50_wt;                 // log10(EC_50)
  vector[num_mut] log_ec50_mut;
  vector[num_epi_var] log_ec50_epi;
  
  real logit_n_eff_wt;    // free energy for Active TF to operator
  vector[num_mut] logit_n_eff_mut;
  vector[num_epi_var] logit_n_eff_epi;
  
  real log_g_max;                // log10 of maximum possible gene expression
  
  real<lower=0> sigma;  // scale factor for standard deviation of noise in y
  
}

transformed parameters {
  vector[num_var] g0;
  vector[num_var] ginf;
  vector[num_var] ec50;
  vector[num_var] n_eff;
  
  vector[num_var] logit_g0_var;
  vector[num_var] logit_ginf_var;
  vector[num_var] log_ec50_var;
  vector[num_var] logit_n_eff_var;
  
  real g_max;
  
  vector[N] fold_change;
  vector[N] mean_y;
  vector[N_contr] mean_y_contr;
  
  g_max = 10^log_g_max;
  
  logit_g0_var[1] = logit_g0_wt;
  logit_ginf_var[1] = logit_ginf_wt;
  log_ec50_var[1] = log_ec50_wt;
  logit_n_eff_var[1] = logit_n_eff_wt;
  
  g0[1] = inv_logit(logit_g0_var[1]);
  ginf[1] = inv_logit(logit_ginf_var[1]);
  ec50[1] = exp(log_ec50_var[1]);
  n_eff[1] = hill_n*inv_logit(logit_n_eff_var[1]);
  
  for (var in 2:num_var) {
    if (var > num_non_epi_var) {
      logit_g0_var[var] = logit_g0_wt + logit_g0_epi[var-num_non_epi_var];
      logit_ginf_var[var] = logit_ginf_wt + logit_ginf_epi[var-num_non_epi_var];
      log_ec50_var[var] = log_ec50_wt + log_ec50_epi[var-num_non_epi_var];
      logit_n_eff_var[var] = logit_n_eff_wt + logit_n_eff_epi[var-num_non_epi_var];
	}
    else {
      logit_g0_var[var] = logit_g0_wt;
      logit_ginf_var[var] = logit_ginf_wt;
      log_ec50_var[var] = log_ec50_wt;
      logit_n_eff_var[var] = logit_n_eff_wt;
	}
	
	for (mut in 1:num_mut) {
	  logit_g0_var[var] += mut_code[var-1, mut]*logit_g0_mut[mut];
	  logit_ginf_var[var] += mut_code[var-1, mut]*logit_ginf_mut[mut];
	  log_ec50_var[var] += mut_code[var-1, mut]*log_ec50_mut[mut];
	  logit_n_eff_var[var] += mut_code[var-1, mut]*logit_n_eff_mut[mut];
	}
	
	g0[var] = inv_logit(logit_g0_var[var]);
    ginf[var] = inv_logit(logit_ginf_var[var]);
    ec50[var] = exp(log_ec50_var[var]);
    n_eff[var] = hill_n*inv_logit(logit_n_eff_var[var]);
  }
  
  for (i in 1:N) {
    fold_change[i] = g0[variant[i]] + (ginf[variant[i]] - g0[variant[i]])*(x[i]^n_eff[variant[i]])/(ec50[variant[i]]^n_eff[variant[i]] + x[i]^n_eff[variant[i]]);
    
    mean_y[i] = g_max*fold_change[i];
  }
  
  for (i in 1:N_contr) {
    mean_y_contr[i] = g_max;
  }
  
}

model {
  // priors on free energy params
  logit_g0_wt ~ normal(logit_g0_wt_prior_mean, logit_g0_wt_prior_std);
  logit_ginf_wt ~ normal(logit_ginf_wt_prior_mean, logit_ginf_wt_prior_std);
  log_ec50_wt ~ normal(log_ec50_wt_prior_mean, log_ec50_wt_prior_std);
  logit_n_eff_wt ~ normal(logit_n_eff_wt_prior_mean, logit_n_eff_wt_prior_std);
  
  // priors on mutational  effects
  logit_g0_mut ~ normal(0, delta_prior_width_hill);
  logit_ginf_mut ~ normal(0, delta_prior_width_hill);
  log_ec50_mut ~ normal(0, delta_prior_width_hill);
  logit_n_eff_mut ~ normal(0, delta_prior_width_hill);
  
  for (var in 1:num_epi_var) {
	target += log_sum_exp(log_phi_1 + normal_lpdf(logit_g0_epi[var] | 0, epi_prior_width_1_hill), log_phi_2 + normal_lpdf(logit_g0_epi[var] | 0, epi_prior_width_2_hill));
	target += log_sum_exp(log_phi_1 + normal_lpdf(logit_ginf_epi[var] | 0, epi_prior_width_1_hill), log_phi_2 + normal_lpdf(logit_ginf_epi[var] | 0, epi_prior_width_2_hill));
	target += log_sum_exp(log_phi_1 + normal_lpdf(log_ec50_epi[var] | 0, epi_prior_width_1_hill), log_phi_2 + normal_lpdf(log_ec50_epi[var] | 0, epi_prior_width_2_hill));
	target += log_sum_exp(log_phi_1 + normal_lpdf(logit_n_eff_epi[var] | 0, epi_prior_width_1_hill), log_phi_2 + normal_lpdf(logit_n_eff_epi[var] | 0, epi_prior_width_2_hill));
  }
  
  // prior on max output level
  log_g_max ~ normal(log10(y_max), g_max_prior_width);
  
  // prior on scale parameter for log-normal measurement error
  sigma ~ normal(0, 1);
  
  // model of the data (dose-response curve with noise)
  y ~ lognormal(log_mean_y, sigma);
  
  // model of the control strain data (constant, max output)
  y_contr ~ lognormal(log_mean_y_contr, sigma);

}

generated quantities {
  // Local variables
  real y_out[num_var, 19];
  real fc_out[num_var, 19];
  
  real log_g0[num_var];
  real log_ginf[num_var];
  real log_ec50[num_var];
  
  for (var in 1:num_var) {
    for (i in 1:19) {
      fc_out[var, i] = g0[var] + (ginf[var] - g0[var])*(x_out[i]^n_eff[var])/(ec50[var]^n_eff[var] + x_out[i]^n_eff[var]);
	
      y_out[var, i] = g_max*fc_out[var, i];
      
    }
	
	log_g0[var] = log10(g0[var]);
	log_ginf[var] = log10(ginf[var]);
	log_ec50[var] = log10(ec50[var]);
	
  }
  
}
