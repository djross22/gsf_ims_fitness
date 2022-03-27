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
  vector[N] log_mean_y;
  vector[N_contr] log_mean_y_contr;
  
  log_mean_y = log(mean_y);
  log_mean_y_contr = log(mean_y_contr);
  
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
