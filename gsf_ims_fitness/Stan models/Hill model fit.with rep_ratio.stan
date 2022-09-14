// Fit dose-response curves to Phillips lab model for allosteric TFs
//

data {
  
#include Free_energy_model.data.shared.stan
  
#include Hill_model.data.hill.stan
  
}

transformed data {

#include Free_energy_model.transformed_data_decl.shared.stan

#include Free_energy_model.transformed_data_assign.shared.stan
  
  // transformed data variable assignments specific to each model
  log_phi_1 = log(epi_prior_phi_hill);
  log_phi_2 = log(1 - epi_prior_phi_hill);
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
  real g_min;           // minimum possible fluorescence (non-fluor control level)
  
  real<lower=0> sigma;      // scale factor for standard deviation of noise in log_y
  real<lower=0> offset_sigma;  // scale factor for standard deviation of replicate variability in g_min
  real<lower=0> g_min_sigma;  // scale factor for standard deviation of noise in g_min
  
  vector[num_reps] log_rep_ratio;              // log10 of multiplicative correction factor for different replicates
  vector[num_contr_reps] log_rep_ratio_contr;  // log10 of multiplicative correction factor for control replicates
  real<lower=0> rep_ratio_sigma;               // hyper-paramters for log_rep_ratio and log_rep_ratio_contr
  
  vector<lower=-3*rep_offset_scale, upper=3*rep_offset_scale>[num_reps] rep_offset;              // additional g_min shift for different replicates
  vector<lower=-3*rep_offset_scale, upper=3*rep_offset_scale>[num_contr_reps] rep_offset_contr;  // additional g_min shift for control replicates
  vector<lower=-3*rep_offset_scale, upper=3*rep_offset_scale>[num_g_min_reps] rep_offset_g_min;  // g_min shift for zero-fluorescence control replicates
  
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
  
  vector[N] log_mean_y;
  vector[N_contr] log_mean_y_contr;
  
  // measured values with g_min subtracted
  vector[N] y_shifted;
  vector[N_contr] y_contr_shifted;
  
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
	real fold_change;
	
    fold_change = g0[variant[i]] + (ginf[variant[i]] - g0[variant[i]])*(x[i]^n_eff[variant[i]])/(ec50[variant[i]]^n_eff[variant[i]] + x[i]^n_eff[variant[i]]);
    
    log_mean_y[i] = ln_10*log_g_max + log(fold_change) + log_rep_ratio[rep[i]];
	
    // measured values with g_min and rep_offset subtracted
    y_shifted[i] = y[i] - g_min - rep_offset[rep[i]];
  }
  
  for (i in 1:N_contr) {
    log_mean_y_contr[i] = ln_10*log_g_max + log_rep_ratio_contr[rep_contr[i]];
	
    y_contr_shifted[i] = y_contr[i] - g_min - rep_offset_contr[rep_contr[i]];
  }
  
  for (i in 1:N_g_min) {
    y_g_min_shifted[i] = y_g_min[i] - g_min - rep_offset_g_min[rep_g_min[i]];
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
  // prior on min output level
  g_min ~ normal(g_min_prior_mu, g_min_prior_std);
  
  // prior on scale parameter for log-normal measurement error
  sigma ~ normal(0, 1);
  
  // model of the data (dose-response curve with noise)
  y_shifted ~ lognormal(log_mean_y, sigma);
  
  // model of the control strain data (constant, max output)
  y_contr_shifted ~ lognormal(log_mean_y_contr, sigma);
  
  // model of the g_min strain data (constant, min output)
  g_min_sigma ~ normal(0, g_min_prior_std)
  y_g_min_shifted ~ normal(0, g_min_sigma)
  rep_offset_g_min  ~ normal(0, offset_sigma);
  offset_sigma ~ normal(0, rep_offset_scale);
  
  // prior on scale hyper-parameter for log_rep_ratio
  rep_ratio_sigma ~ normal(0, rep_ratio_scale);
  
  // priors on log_rep_ratio and log_rep_ratio_contr
  log_rep_ratio ~ normal(0, rep_ratio_sigma);
  log_rep_ratio_contr ~ normal(0, rep_ratio_sigma);
  
  // priors on rep_offset and rep_offset_contr
  rep_offset ~ normal(0, offset_sigma);
  rep_offset_contr ~ normal(0, offset_sigma);

}

generated quantities {
  real y_out[num_var, 19];
  real fc_out[num_var, 19];
  
  real log_g0[num_var];
  real log_ginf[num_var];
  real log_ec50[num_var];
  
  real g_max;
  
  g_max = 10^log_g_max;
  
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
