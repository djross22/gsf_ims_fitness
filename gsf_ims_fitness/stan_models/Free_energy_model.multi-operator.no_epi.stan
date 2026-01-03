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

#include dose_response_model.transformed_data.shared_declare.stan

#include dose_response_model.transformed_data.shared_assign.stan
  
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

#include Free_energy_model.no_epi.transformed_parameters.variant_free_energies.stan

#include Free_energy_model.transformed_parameters.fold_change_multi_rep_ratio.stan
  
}

model {

#include Free_energy_model.no_epi.model.shared.stan

#include Free_energy_model.model.rep_ratio.stan

#include Free_energy_model.model.multi_operator.stan

}

generated quantities {

#include Free_energy_model.generated_quantities.shared.stan

#include Free_energy_model.generated_quantities.multi.stan
  
}
