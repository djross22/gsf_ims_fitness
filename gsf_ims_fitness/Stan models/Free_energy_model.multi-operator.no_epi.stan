// Fit dose-response curves to Phillips lab model for allosteric TFs
//     this version of the model has strictly zero epistasis

data {

#include dose_response_model.data.shared.stan

#include dose_response_model.data.free_energy.stan
  
  // Input data variables for multi-operator model priors
  real<lower=2> copy_num_prior_mean;  // geometric mean for prior on plasmid/operator copy number
  real<lower=0> copy_num_prior_width; // geometric std for prior on plasmid/operator copy number
  real<lower=1> R_prior_mean;         // geometric mean for prior on repressor dimer copy number
  real<lower=0> log_R_prior_width;    // geometric std for prior on repressor dimer copy number
  
}

transformed data {

#include dose_response_model.transformed_data.shared.stan
  
  N_NS = 3*4600000;
  
}

parameters {
  // In this version of the model, the base parameters belong to the first variant (the wild-type)
  //     and there is a delta_param associated with each mutation (with additive effects)
  
#include Free_energy_model.no_epi.parameters.shared.stan

#include Free_energy_model.parameters.multi_operator.stan

#include Free_energy_model.parameters.rep_ratio.stan
  
}

transformed parameters {

#include Free_energy_model.transformed_parameters.shared.stan

//This include file has both declarations and assignments in it. So, it has to go after any include file that has other declarations. 
#include Free_energy_model.transformed_parameters.multi_operator.stan

// ****** #include Free_energy_model.transformed_parameters.variant_free_energies.stan
  log_k_a_var[1] = log_k_a_wt;
  log_k_i_var[1] = log_k_i_wt;
  delta_eps_AI_var[1] = delta_eps_AI_wt;
  delta_eps_RA_var[1] = delta_eps_RA_wt;
  
  K_A[1] = 10^log_k_a_var[1];
  K_I[1] = 10^log_k_i_var[1];
  
  for (var in 2:num_var) {
    log_k_a_var[var] = log_k_a_wt;
    log_k_i_var[var] = log_k_i_wt;
    delta_eps_AI_var[var] = delta_eps_AI_wt;
    delta_eps_RA_var[var] = delta_eps_RA_wt;
	
	for (mut in 1:num_mut) {
	  log_k_a_var[var] += mut_code[var-1, mut]*log_k_a_mut[mut];
	  log_k_i_var[var] += mut_code[var-1, mut]*log_k_i_mut[mut];
	  delta_eps_AI_var[var] += mut_code[var-1, mut]*delta_eps_AI_mut[mut];
	  delta_eps_RA_var[var] += mut_code[var-1, mut]*delta_eps_RA_mut[mut];
	}
	
	K_A[var] = 10^log_k_a_var[var];
    K_I[var] = 10^log_k_i_var[var];
  }

// ******

#include Free_energy_model.transformed_parameters.fold_change_multi_rep_ratio.stan
  
}

model {

// ****** #include Free_energy_model.model.shared.stan
  // priors on free energy params
  log_k_a_wt ~ normal(log_k_a_wt_prior_mean, log_k_a_wt_prior_std);
  log_k_i_wt ~ normal(log_k_i_wt_prior_mean, log_k_i_wt_prior_std);
  delta_eps_AI_wt ~ normal(delta_eps_AI_wt_prior_mean, delta_eps_AI_wt_prior_std);
  delta_eps_RA_wt ~ normal(delta_eps_RA_wt_prior_mean, delta_eps_RA_wt_prior_std);
  
  log_k_a_mut ~ normal(0, delta_prior_width/ln_10); // factor of 1/ln_10 is to compensate for use of log10 instead of ln
  
  log_k_i_mut ~ normal(0, delta_prior_width/ln_10); // factor of 1/ln_10 is to compensate for use of log10 instead of ln
  
  delta_eps_AI_mut ~ normal(0, delta_prior_width);
  
  for (mut in 1:num_mut) {
    delta_eps_RA_mut[mut] ~ normal(0, delta_prior_width*eps_RA_prior_scale[mut]);
  }
  
  // prior on max output level
  log_g_max ~ normal(log10(y_max), g_max_prior_width);
  
  // prior on scale parameter for log-normal measurement error
  sigma ~ normal(0, 1);
  
  // model of the data (dose-response curve with noise)
  y_shifted ~ lognormal(log_mean_y, sigma);
  
  // model of the control strain data (constant, max output)
  y_contr_shifted ~ lognormal(log_mean_y_contr, sigma);
// ******

#include Free_energy_model.model.rep_ratio.stan

#include Free_energy_model.model.multi_operator.stan

}

generated quantities {

#include Free_energy_model.generated_quantities.shared.stan

#include Free_energy_model.generated_quantities.multi.stan
  
}
