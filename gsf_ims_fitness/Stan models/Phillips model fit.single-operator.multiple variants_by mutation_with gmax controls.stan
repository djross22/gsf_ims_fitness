// Fit dose-response curves to Phillips lab model for allosteric TFs
//

functions {

#include functions.fold_change_single.stan

}

data {

#include Free_energy_model.data.shared.stan

#include Free_energy_model.data.free_energy.stan
  
}

transformed data {

#include Free_energy_model.transformed_data_decl.shared.stan
  
  real R; // specific to single-operator model
  real N_S; // operator copy number

#include Free_energy_model.transformed_data_assign.shared.stan
  
  // transformed data variable assignments specific to each model
  log_phi_1 = log(epi_prior_phi);
  log_phi_2 = log(1 - epi_prior_phi);
  R = 200;
  N_NS = 4600000;
  N_S = 1;
  
}

parameters {
  // In this version of the model, the base parameters belong to the first variant (the wild-type)
  //     and there is a delta_param associated with each mutation (with additive effects)
  //     plus an epistasis term associated with each variant other than the wild-type
  
#include Free_energy_model.parameters.shared.stan
  
}

transformed parameters {

#include Free_energy_model.transformed_parameters.shared.stan

#include Free_energy_model.transformed_parameters.no_rep_ratio.stan

#include Free_energy_model.transformed_parameters.variant_free_energies.stan

#include Free_energy_model.transformed_parameters.fold_change_no_rep_ratio.stan
  
}

model {
  vector[N] log_mean_y;
  vector[N_contr] log_mean_y_contr;
  
  log_mean_y = log(mean_y);
  log_mean_y_contr = log(mean_y_contr);
  
  // priors on free energy params
  log_k_a_wt ~ normal(log_k_a_wt_prior_mean, log_k_a_wt_prior_std);
  log_k_i_wt ~ normal(log_k_i_wt_prior_mean, log_k_i_wt_prior_std);
  delta_eps_AI_wt ~ normal(delta_eps_AI_wt_prior_mean, delta_eps_AI_wt_prior_std);
  delta_eps_RA_wt ~ normal(delta_eps_RA_wt_prior_mean, delta_eps_RA_wt_prior_std);
  
  log_k_a_mut ~ normal(0, delta_prior_width/2.3); // factor of 1/2.3 is to compensate for use of log10 instead of ln
  
  log_k_i_mut ~ normal(0, delta_prior_width/2.3); // factor of 1/2.3 is to compensate for use of log10 instead of ln
  
  delta_eps_AI_mut ~ normal(0, delta_prior_width);
  
  for (mut in 1:num_mut) {
    delta_eps_RA_mut[mut] ~ normal(0, delta_prior_width*eps_RA_prior_scale[mut]);
  }
  
  for (var in 1:num_epi_var) {
    // factor of 1/2.3 is to compensate for use of log10 instead of ln
	target += log_sum_exp(log_phi_1 + normal_lpdf(log_k_a_epi[var] | 0, epi_prior_width_1/2.3), log_phi_2 + normal_lpdf(log_k_a_epi[var] | 0, epi_prior_width_2/2.3));
	target += log_sum_exp(log_phi_1 + normal_lpdf(log_k_i_epi[var] | 0, epi_prior_width_1/2.3), log_phi_2 + normal_lpdf(log_k_i_epi[var] | 0, epi_prior_width_2/2.3));
	
	target += log_sum_exp(log_phi_1 + normal_lpdf(delta_eps_AI_epi[var] | 0, epi_prior_width_1), log_phi_2 + normal_lpdf(delta_eps_AI_epi[var] | 0, epi_prior_width_2));
	target += log_sum_exp(log_phi_1 + normal_lpdf(delta_eps_RA_epi[var] | 0, epi_prior_width_1*RA_epi_prior_scale[var]), log_phi_2 + normal_lpdf(delta_eps_RA_epi[var] | 0, epi_prior_width_2*RA_epi_prior_scale[var]));
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
  
  for (var in 1:num_var) {
    for (i in 1:19) {
	  real f_c;
  
      f_c = fold_change_fnct(x_out[i], K_A[var], K_I[var], delta_eps_AI_var[var], delta_eps_RA_var[var], hill_n, N_NS, R, N_S);
	
      y_out[var, i] = g_max*f_c;
      fc_out[var, i] = f_c;
    }
  }
  
}
