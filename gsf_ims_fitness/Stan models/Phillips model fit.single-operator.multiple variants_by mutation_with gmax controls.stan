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
#include Free_energy_model.model.no_rep_ratio.stan

}

generated quantities {
  real g_max;
  real y_out[num_var, 19];
  real fc_out[num_var, 19];
  
  g_max = 10^log_g_max;
  
  for (var in 1:num_var) {
    for (i in 1:19) {
	  real f_c;
  
      f_c = fold_change_fnct(x_out[i], K_A[var], K_I[var], delta_eps_AI_var[var], delta_eps_RA_var[var], hill_n, N_NS, R, N_S);
	
      y_out[var, i] = g_max*f_c;
      fc_out[var, i] = f_c;
    }
  }
  
}
