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
  
  real y_max;             // geometric mean for prior on maximum gene expression value
  real g_max_prior_width; // geometric std for prior on maximum gene expression value
  
  int rep[N];            // integer to indicate the measurement replicate
  int<lower=1> num_reps; // number of measurement replicates (for all variants)
  
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
  real epi_prior_width_hill;   // width of prior on parameter epistasis
  
  real rep_ratio_scale;   // parameter to set the scale for the half-normal prior on log_rep_ratio
  real rep_offset_scale;  // parameter to set the scale for the half-normal prior on log_rep_ratio
}

transformed data {
  real hill_n;
  vector[19] x_out;
  int num_non_epi_var;  // number of variants with less than two mutations
  
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
  
  vector<lower=-3*rep_ratio_scale, upper=3*rep_ratio_scale>[num_reps] log_rep_ratio;  // log10 of multiplicative correction factor for different replicates
  vector<lower=-3*rep_offset_scale, upper=3*rep_offset_scale>[num_reps] rep_offset;   // offset for different replicates
  
  vector<lower=-3*rep_ratio_scale, upper=3*rep_ratio_scale>[num_contr_reps] log_rep_ratio_contr;  // log10 of multiplicative correction factor for control replicates
  vector<lower=-3*rep_offset_scale, upper=3*rep_offset_scale>[num_contr_reps] rep_offset_contr;   // offset for control replicates
  
  // hyper-parameters for log_rep_ratio and rep_offset
  real<lower=0> rep_ratio_sigma;
  real<lower=0> rep_offset_sigma;
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
  
  vector[num_reps] rep_ratio;
  vector[num_contr_reps] rep_ratio_contr;
  
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
  
  for (j in 1:num_reps) {
    rep_ratio[j] = 10^log_rep_ratio[j];
  }
  for (j in 1:num_contr_reps) {
    rep_ratio_contr[j] = 10^log_rep_ratio_contr[j];
  }
  
  for (i in 1:N) {
    fold_change[i] = g0[variant[i]] + (ginf[variant[i]] - g0[variant[i]])*(x[i]^n_eff[variant[i]])/(ec50[variant[i]]^n_eff[variant[i]] + x[i]^n_eff[variant[i]]);
    
    mean_y[i] = g_max*rep_ratio[rep[i]]*fold_change[i] + rep_offset[rep[i]];
  }
  
  for (i in 1:N_contr) {
    mean_y_contr[i] = g_max*rep_ratio_contr[rep_contr[i]] + rep_offset_contr[rep_contr[i]];
  }
  
}

model {
  // priors on free energy params
  logit_g0_wt ~ normal(logit_g0_wt_prior_mean, logit_g0_wt_prior_std);
  logit_ginf_wt ~ normal(logit_ginf_wt_prior_mean, logit_ginf_wt_prior_std);
  log_ec50_wt ~ normal(log_ec50_wt_prior_mean, log_ec50_wt_prior_std);
  logit_n_eff_wt ~ normal(logit_n_eff_wt_prior_mean, logit_n_eff_wt_prior_std);
  
  logit_g0_mut ~ normal(0, delta_prior_width_hill);
  logit_g0_epi ~ normal(0, epi_prior_width_hill);
  
  logit_ginf_mut ~ normal(0, delta_prior_width_hill);
  logit_ginf_epi ~ normal(0, epi_prior_width_hill);
  
  log_ec50_mut ~ normal(0, delta_prior_width_hill);
  log_ec50_epi ~ normal(0, epi_prior_width_hill);
  
  logit_n_eff_mut ~ normal(0, delta_prior_width_hill);
  logit_n_eff_epi ~ normal(0, epi_prior_width_hill);
  
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
  
  real log_g0[num_var];
  real log_ginf[num_var];
  real log_ec50[num_var];
  
  mean_offset = mean(rep_offset);
  geo_mean_ratio = 10^mean(log_rep_ratio);
  
  for (var in 1:num_var) {
    for (i in 1:19) {
      fc_out[var, i] = g0[var] + (ginf[var] - g0[var])*(x_out[i]^n_eff[var])/(ec50[var]^n_eff[var] + x_out[i]^n_eff[var]);
	
      y_out[var, i] = g_max*geo_mean_ratio*fc_out[var, i] + mean_offset;
      
    }
	
	log_g0[var] = log10(g0[var]);
	log_ginf[var] = log10(ginf[var]);
	log_ec50[var] = log10(ec50[var]);
	
  }
  
}
