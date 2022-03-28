// Fit dose-response curves to Phillips lab model for allosteric TFs
//

data {

#include Free_energy_model.data.shared.stan

#include Free_energy_model.data.free_energy.stan
  
}

transformed data {

#include Free_energy_model.transformed_data_decl.shared.stan
  
  real R;   // specific to single-operator model
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

#include Free_energy_model.transformed_parameters.variant_free_energies.stan

#include Free_energy_model.transformed_parameters.fold_change_single_no_rep_ratio.stan
  
}

model {

#include Free_energy_model.model.shared.stan

}

generated quantities {

#include Free_energy_model.generated_quantities.shared.stan
  
}
