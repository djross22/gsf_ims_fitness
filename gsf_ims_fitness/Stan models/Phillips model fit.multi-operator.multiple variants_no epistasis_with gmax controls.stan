// Fit dose-response curves to Phillips lab model for allosteric TFs
//

functions {

#include functions.fold_change_multi.stan

}

data {

#include Free_energy_model.data.shared.stan

#include Free_energy_model.data.free_energy.stan

#include Free_energy_model.data.multi_operator.stan
  
}

transformed data {

#include Free_energy_model.transformed_data_decl.shared.stan

  //declare these here to avoid a compilation error (in Free_energy_model.transformed_parameters.variant_free_energies.stan), 
  //  but they are not actually used.
  vector[num_epi_var] log_k_a_epi;
  vector[num_epi_var] log_k_i_epi;
  vector[num_epi_var] delta_eps_AI_epi;
  vector[num_epi_var] delta_eps_RA_epi;

#include Free_energy_model.transformed_data_assign.shared.stan

  num_non_epi_var = num_var; // this overrides the statement in the include file becasue this is the no_epistasis model
  
  // transformed data variable assignments specific to each model
  log_phi_1 = log(epi_prior_phi);
  log_phi_2 = log(1 - epi_prior_phi);
  
  N_NS = 3*4600000;
  
}

parameters {
  // In this version of the model, the base parameters belong to the first variant (the wild-type)
  //     and there is a delta_param associated with each mutation (with additive effects)
  //     Also, there is no epistasis term
  real log_k_a_wt;         // log10 of IPTG binding affinity to active state
  vector[num_mut] log_k_a_mut;
  
  real log_k_i_wt;         // log10 of IPTG binding affinity to inactive state
  vector[num_mut] log_k_i_mut;
  
  real delta_eps_AI_wt;    // free energy difference between active and inactive states
  vector[num_mut] delta_eps_AI_mut;
  
  real delta_eps_RA_wt;    // free energy for Active TF to operator
  vector[num_mut] delta_eps_RA_mut;
  
  real log_g_max;                // log10 of maximum possible gene expression
  real<lower=0> log_copy_num;    // log10 of plasmid/operator copy number
  real<lower=0> log_R;           // log10 of repressor dimer copy number
  
  real<lower=0> sigma;  // scale factor for standard deviation of noise in y
  
}

transformed parameters {

#include Free_energy_model.transformed_parameters.shared.stan

//This include file has both declarations and assignments in it. So, it has to go after and include file that has other declarations. 
#include Free_energy_model.transformed_parameters.multi_operator.stan

#include Free_energy_model.transformed_parameters.variant_free_energies.stan

#include Free_energy_model.transformed_parameters.fold_change_no_rep_ratio.stan
  
}

model {

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
  
  // prior on max output level
  log_g_max ~ normal(log10(y_max), g_max_prior_width);
  
  // prior on scale parameter for log-normal measurement error
  sigma ~ normal(0, 1);
  
  // model of the data (dose-response curve with noise)
  y ~ lognormal(log_mean_y, sigma);
  
  // model of the control strain data (constant, max output)
  y_contr ~ lognormal(log_mean_y_contr, sigma);
  
  // priors on plasmid/operator and repressor dimer copy numbers
  log_copy_num ~ normal(log10(copy_num_prior_mean), copy_num_prior_width);
  log_R ~ normal(log10(R_prior_mean), log_R_prior_width);

}

generated quantities {

#include Free_energy_model.generated_quantities.shared.stan
  
}
