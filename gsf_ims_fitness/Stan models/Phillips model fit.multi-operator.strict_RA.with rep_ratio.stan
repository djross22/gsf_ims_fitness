// Fit dose-response curves to Phillips lab model for allosteric TFs
//     In this version, mutations in the DNA binding domain are only allowed to affect delta_eps_RA
//     Also, log_k_a_epi, log_k_i_epi, and delta_eps_AI_epi can only be non-zero if there are multiple mutations in the core domain
//     This requires attitional data inputs that are created by ther script: make_stan_data_strict_RA.py

data {

#include Free_energy_model.data.shared.stan

#include Free_energy_model.data.free_energy.stan

#include Free_energy_model.data.multi_operator.stan

  int<lower=0, upper=1> can_shift_allostery[num_mut]; // should be zero for mutations in the DNA binding domain and one otherwise
  
  int<lower=0, upper=1> can_have_allosteric_epi[num_epi_var]; // should be one for variants with two or more mutations in the core and zero otherwise
  
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
      log_k_a_var[var] = log_k_a_wt + log_k_a_epi[var-num_non_epi_var]*can_have_allosteric_epi[var-num_non_epi_var];
      log_k_i_var[var] = log_k_i_wt + log_k_i_epi[var-num_non_epi_var]*can_have_allosteric_epi[var-num_non_epi_var];
      delta_eps_AI_var[var] = delta_eps_AI_wt + delta_eps_AI_epi[var-num_non_epi_var]*can_have_allosteric_epi[var-num_non_epi_var];
      delta_eps_RA_var[var] = delta_eps_RA_wt + delta_eps_RA_epi[var-num_non_epi_var];
	}
    else {
      log_k_a_var[var] = log_k_a_wt;
      log_k_i_var[var] = log_k_i_wt;
      delta_eps_AI_var[var] = delta_eps_AI_wt;
      delta_eps_RA_var[var] = delta_eps_RA_wt;
	}
	
	for (mut in 1:num_mut) {
	  log_k_a_var[var] += mut_code[var-1, mut]*log_k_a_mut[mut]*can_shift_allostery[mut];
	  log_k_i_var[var] += mut_code[var-1, mut]*log_k_i_mut[mut]*can_shift_allostery[mut];
	  delta_eps_AI_var[var] += mut_code[var-1, mut]*delta_eps_AI_mut[mut]*can_shift_allostery[mut];
	  delta_eps_RA_var[var] += mut_code[var-1, mut]*delta_eps_RA_mut[mut];
	}
	
	K_A[var] = 10^log_k_a_var[var];
    K_I[var] = 10^log_k_i_var[var];
  }
// *****

#include Free_energy_model.transformed_parameters.fold_change_multi_rep_ratio.stan
  
}

model {

#include Free_energy_model.model.shared.stan

#include Free_energy_model.model.rep_ratio.stan

#include Free_energy_model.model.multi_operator.stan

}

generated quantities {

#include Free_energy_model.generated_quantities.shared.stan

#include Free_energy_model.generated_quantities.multi.stan
  
}
