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
  
#include Free_energy_model.parameters.shared.stan

#include Free_energy_model.parameters.multi_operator.stan
  
}

transformed parameters {

#include Free_energy_model.transformed_parameters.shared.stan

#include Free_energy_model.transformed_parameters.multi_operator.stan

  N_S = 10^log_copy_num;
  R = 10^log_R;

#include Free_energy_model.transformed_parameters.variant_free_energies.stan

#include Free_energy_model.transformed_parameters.fold_change_no_rep_ratio.stan
  
}

model {

#include Free_energy_model.model.no_rep_ratio.stan

#include Free_energy_model.model.multi_operator.stan

}

generated quantities {

#include Free_energy_model.generated_quantities.shared.stan
  
}
