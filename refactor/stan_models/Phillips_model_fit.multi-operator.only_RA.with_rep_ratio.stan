// Fit dose-response curves to Phillips lab model for allosteric TFs
//     This version sets the mutational shifts and epistasis strictly to zero for all free energy parameters except delta_eps_RA

data {

#include Free_energy_model.data.shared.stan

#include Free_energy_model.data.free_energy.stan

#include Free_energy_model.data.multi_operator.stan
  
}

transformed data {

#include Free_energy_model.transformed_data_decl.shared.stan

#include Free_energy_model.transformed_data_assign.shared.stan
  
  // transformed data variable assignments specific to each model
  log_phi_1 = log(epi_prior_phi);
  log_phi_2 = log(1 - epi_prior_phi);
  
  N_NS = 3*4600000;
}

parameters {
  // In this version of the model, the base parameters belong to the first variant (the wild-type)
  //     and there is a delta_param associated with each mutation (with additive effects)
  //     plus an epistasis term associated with each variant other than the wild-type
  
// ***** #include Free_energy_model.parameters.shared.stan
  // This version has no mutational effect or epistasis for the delta_eps_RA parameter
  real log_k_a_wt;         // log10 of IPTG binding affinity to active state
  
  real log_k_i_wt;         // log10 of IPTG binding affinity to inactive state
  
  real delta_eps_AI_wt;    // free energy difference between active and inactive states
  
  real delta_eps_RA_wt;    // free energy for Active TF to operator
  vector[num_mut] delta_eps_RA_mut;
  vector[num_epi_var] delta_eps_RA_epi;
  
  real log_g_max;       // log10 of maximum possible gene expression
  real g_min;           // minimum possible fluorescence (non-fluor control level)
  
  real<lower=0> sigma;      // scale factor for standard deviation of noise in log_y
  real<lower=0> offset_sigma;  // scale factor for standard deviation of noise in g_min

// *****

#include Free_energy_model.parameters.multi_operator.stan

#include Free_energy_model.parameters.rep_ratio.stan
  
}

transformed parameters {

#include Free_energy_model.transformed_parameters.shared.stan

//This include file has both declarations and assignments in it. So, it has to go after any include file that has other declarations. 
#include Free_energy_model.transformed_parameters.multi_operator.stan

// ***** #include Free_energy_model.transformed_parameters.variant_free_energies.stan
  log_k_a_var[1] = log_k_a_wt;
  log_k_i_var[1] = log_k_i_wt;
  delta_eps_AI_var[1] = delta_eps_AI_wt;
  delta_eps_RA_var[1] = delta_eps_RA_wt;
  
  K_A[1] = 10^log_k_a_var[1];
  K_I[1] = 10^log_k_i_var[1];
  
  for (var in 2:num_var) {
    if (var > num_non_epi_var) {
      log_k_a_var[var] = log_k_a_wt;
      log_k_i_var[var] = log_k_i_wt;
      delta_eps_AI_var[var] = delta_eps_AI_wt;
      delta_eps_RA_var[var] = delta_eps_RA_wt + delta_eps_RA_epi[var-num_non_epi_var];
	}
    else {
      log_k_a_var[var] = log_k_a_wt;
      log_k_i_var[var] = log_k_i_wt;
      delta_eps_AI_var[var] = delta_eps_AI_wt;
      delta_eps_RA_var[var] = delta_eps_RA_wt;
	}
	
	for (mut in 1:num_mut) {
	  delta_eps_RA_var[var] += mut_code[var-1, mut]*delta_eps_RA_mut[mut];
	}
	
	K_A[var] = 10^log_k_a_var[var];
    K_I[var] = 10^log_k_i_var[var];
  }
// *****

#include Free_energy_model.transformed_parameters.fold_change_multi_rep_ratio.stan
  
}

model {

// ***** #include Free_energy_model.model.shared.stan
  // priors on free energy params
  log_k_a_wt ~ normal(log_k_a_wt_prior_mean, log_k_a_wt_prior_std);
  log_k_i_wt ~ normal(log_k_i_wt_prior_mean, log_k_i_wt_prior_std);
  delta_eps_AI_wt ~ normal(delta_eps_AI_wt_prior_mean, delta_eps_AI_wt_prior_std);
  delta_eps_RA_wt ~ normal(delta_eps_RA_wt_prior_mean, delta_eps_RA_wt_prior_std);
  
  delta_eps_RA_mut ~ normal(0, delta_prior_width);
  
  for (var in 1:num_epi_var) {
	target += log_sum_exp(log_phi_1 + normal_lpdf(delta_eps_RA_epi[var] | 0, epi_prior_width_1), log_phi_2 + normal_lpdf(delta_eps_RA_epi[var] | 0, epi_prior_width_2));
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
  y_g_min ~ normal(g_min, offset_sigma);
  offset_sigma ~ normal(0, rep_offset_scale);
// ***** 

#include Free_energy_model.model.rep_ratio.stan

#include Free_energy_model.model.multi_operator.stan

}

generated quantities {

// ***** #include Free_energy_model.generated_quantities.shared.stan
  real g_max;
  real y_out[num_var, 19];
  real fc_out[num_var, 19];
  
  //this gets instered here becasue all declarations must come before any operations
  vector[num_mut] log_k_a_mut;
  vector[num_epi_var] log_k_a_epi;
  vector[num_mut] log_k_i_mut;
  vector[num_epi_var] log_k_i_epi;
  vector[num_mut] delta_eps_AI_mut;
  vector[num_epi_var] delta_eps_AI_epi;
  
  for (var in 1:num_epi_var) {
    log_k_a_epi[var] = 0;
    log_k_i_epi[var] = 0;
    delta_eps_AI_epi[var] = 0;
  }
  
  for (mut in 1:num_mut) {
    log_k_a_mut[mut] = 0;
    log_k_i_mut[mut] = 0;
    delta_eps_AI_mut[mut] = 0;
  }
  //
  
  
  g_max = 10^log_g_max;
// *****

#include Free_energy_model.generated_quantities.multi.stan
  
}
