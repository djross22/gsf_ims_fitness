// Fit dose-response curves to Phillips lab model for allosteric TFs
//

data {

// ***** #include Free_energy_model.data.shared.stan

  // Input data variables that are shared across all models 
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // gene expression (from cytometry) at each concentration
  vector[N] y_err;       // estimated error of gene expression at each concentration
  
  int rep[N];            // integer to indicate the measurement replicate
  int<lower=1> num_reps; // number of measurement replicates (for all variants)
  
  int<lower=2> num_var;  // number of variants
  int variant[N];        // numerical index to indicate variants
  int<lower=0> num_epi_var;  // number of variants with more than one mutation (only define epistasis for these)
  
  int<lower=1> num_mut;  // number of differrent mutations
  int<lower=0, upper=1> mut_code[num_var, num_mut];   // one-hot encoding for  presence of mutations in each variant; variant 0 (WT) has no mutations
  
  real rep_ratio_scale;   // parameter to set the scale for the half-normal prior on log_rep_ratio
  
  // prior for non-fluorescent control level
  real g_min_prior_mu;
  real g_min_prior_std;
  real rep_offset_scale;  // hyperparameter parameter to set the scale for the half-normal prior on offset_sigma
  
// *****

// ***** #include Free_energy_model.data.free_energy.stan
  // Input data for priors on wild-type free energy parameters (single- and multi-operator models)
  real log_k_a_wt_prior_mu;
  real log_k_a_wt_prior_std;
  real log_k_i_wt_prior_mu;
  real log_k_i_wt_prior_std;
  real delta_eps_AI_wt_prior_mu;
  real delta_eps_AI_wt_prior_std;
  real delta_eps_RA_wt_prior_mu;
  real delta_eps_RA_wt_prior_std;
  
  // priors on mutational effects and epistasis  (single- and multi-operator models)
  real delta_prior_width;   // width of prior on "_mut" parameters (mutational effects)
  real epi_prior_width_1;   // width of 1st mixture component of prior on parameter epistasis
  real epi_prior_width_2;   // width of 2nd mixture component of prior on parameter epistasis
  real epi_prior_phi;       // weight for 1st mixture component of prior on parameter epistasis
  
  real<lower=0> eps_RA_prior_scale[num_mut];     // scale multiplier for width of priors on operator binding free energy term (delta_eps_RA_mut) 
  real<lower=0> RA_epi_prior_scale[num_epi_var]; // scale multiplier for width of priors on operator binding free energy epistasis (delta_eps_RA_epi) 
// *****

// ***** #include Free_energy_model.data.multi_operator.stan
  // Don't need any of these for this model
// *****

  real rep_ratio_sigma_prior_mu;
  real rep_ratio_sigma_prior_std;
  
  real offset_sigma_prior_mu;
  real offset_sigma_prior_std;
  
  real log_g_max_prior_mu;
  real log_g_max_prior_std;
  
  real sigma_prior_mu;
  real sigma_prior_std;
  
  real log_copy_num_prior_mu;
  real log_copy_num_prior_std;
  
  real log_R_prior_mu;
  real log_R_prior_std;
  
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
  // In this version of the model, the base parameters belong to the wild-type variant (which is not included in the input data, informed mainly by the priors)
  //     and there is a delta_param associated with each mutation (with additive effects)
  //     plus an epistasis term associated with each variant other than the wild-type
  
#include Free_energy_model.parameters.shared.stan

#include Free_energy_model.parameters.multi_operator.stan

// ***** include Free_energy_model.parameters.rep_ratio.stan
  vector[num_reps] log_rep_ratio;              // log10 of multiplicative correction factor for different replicates

  real<lower=0> rep_ratio_sigma;               // hyper-paramter for log_rep_ratio
  
  vector<lower=-3*rep_offset_scale, upper=3*rep_offset_scale>[num_reps] rep_offset;              // additional g_min shift for different replicates

// *****
  
}

transformed parameters {

// ***** include Free_energy_model.transformed_parameters.shared.stan
  vector[num_var] K_A;
  vector[num_var] K_I;
  vector[num_var] log_k_a_var;
  vector[num_var] log_k_i_var;
  vector[num_var] delta_eps_AI_var;
  vector[num_var] delta_eps_RA_var;
  
  vector[N] log_mean_y;
  
  // measured values with g_min subtracted
  vector[N] y_shifted;
// *****

//This include file has both declarations and assignments in it. So, it has to go after any include file that has other declarations. 
#include Free_energy_model.transformed_parameters.multi_operator.stan

// ***** include Free_energy_model.transformed_parameters.variant_free_energies.stan
  
  for (var in 1:num_var) {
    if (var > num_non_epi_var) {
      log_k_a_var[var] = log_k_a_wt + log_k_a_epi[var-num_non_epi_var];
      log_k_i_var[var] = log_k_i_wt + log_k_i_epi[var-num_non_epi_var];
      delta_eps_AI_var[var] = delta_eps_AI_wt + delta_eps_AI_epi[var-num_non_epi_var];
      delta_eps_RA_var[var] = delta_eps_RA_wt + delta_eps_RA_epi[var-num_non_epi_var];
	}
    else {
      log_k_a_var[var] = log_k_a_wt;
      log_k_i_var[var] = log_k_i_wt;
      delta_eps_AI_var[var] = delta_eps_AI_wt;
      delta_eps_RA_var[var] = delta_eps_RA_wt;
	}
	
	for (mut in 1:num_mut) {
	  log_k_a_var[var] += mut_code[var, mut]*log_k_a_mut[mut];
	  log_k_i_var[var] += mut_code[var, mut]*log_k_i_mut[mut];
	  delta_eps_AI_var[var] += mut_code[var, mut]*delta_eps_AI_mut[mut];
	  delta_eps_RA_var[var] += mut_code[var, mut]*delta_eps_RA_mut[mut];
	}
	
	K_A[var] = 10^log_k_a_var[var];
    K_I[var] = 10^log_k_i_var[var];
  }
// *****

// ***** include Free_energy_model.transformed_parameters.fold_change_multi_rep_ratio.stan
  for (i in 1:N) {
    real c1;
    real c2;
    real pA;
	real xRA;
	real lam;
	real fold_change;
	
    c1 = (1 + x[i]/K_A[variant[i]])^hill_n;
    c2 = ( (1 + x[i]/K_I[variant[i]])^hill_n ) * exp(-delta_eps_AI_var[variant[i]]);
	pA = c1/(c1+c2);
	xRA = exp(-delta_eps_RA_var[variant[i]]);
	
	lam = -N_NS + pA*R - N_S*xRA + pA*R*xRA;
    lam = lam + sqrt(4*pA*R*xRA*(N_NS + N_S - pA*R) + (N_NS + N_S*xRA - pA*R*(1 + xRA))^2);
    lam = lam/(2*xRA*(N_NS + N_S - pA*R));
	
    fold_change = 1/(1 + lam*xRA);
	
    log_mean_y[i] = ln_10*log_g_max + log(fold_change) + log_rep_ratio[rep[i]];
	
    // measured values with g_min and rep_offset subtracted
    y_shifted[i] = y[i] - g_min - rep_offset[rep[i]];
  }
// *****
  
}

model {

// ***** include Free_energy_model.model.shared.stan
  // priors on free energy params
  log_k_a_wt ~ normal(log_k_a_wt_prior_mu, log_k_a_wt_prior_std);
  log_k_i_wt ~ normal(log_k_i_wt_prior_mu, log_k_i_wt_prior_std);
  delta_eps_AI_wt ~ normal(delta_eps_AI_wt_prior_mu, delta_eps_AI_wt_prior_std);
  delta_eps_RA_wt ~ normal(delta_eps_RA_wt_prior_mu, delta_eps_RA_wt_prior_std);
  
  log_k_a_mut ~ normal(0, delta_prior_width/ln_10); // factor of 1/ln_10 is to compensate for use of log10 instead of ln
  
  log_k_i_mut ~ normal(0, delta_prior_width/ln_10); // factor of 1/ln_10 is to compensate for use of log10 instead of ln
  
  delta_eps_AI_mut ~ normal(0, delta_prior_width);
  
  for (mut in 1:num_mut) {
    delta_eps_RA_mut[mut] ~ normal(0, delta_prior_width*eps_RA_prior_scale[mut]);
  }
  
  for (var in 1:num_epi_var) {
    // factor of 1/ln_10 is to compensate for use of log10 instead of ln
	target += log_sum_exp(log_phi_1 + normal_lpdf(log_k_a_epi[var] | 0, epi_prior_width_1/ln_10), log_phi_2 + normal_lpdf(log_k_a_epi[var] | 0, epi_prior_width_2/ln_10));
	target += log_sum_exp(log_phi_1 + normal_lpdf(log_k_i_epi[var] | 0, epi_prior_width_1/ln_10), log_phi_2 + normal_lpdf(log_k_i_epi[var] | 0, epi_prior_width_2/ln_10));
	
	target += log_sum_exp(log_phi_1 + normal_lpdf(delta_eps_AI_epi[var] | 0, epi_prior_width_1), log_phi_2 + normal_lpdf(delta_eps_AI_epi[var] | 0, epi_prior_width_2));
	target += log_sum_exp(log_phi_1 + normal_lpdf(delta_eps_RA_epi[var] | 0, epi_prior_width_1*RA_epi_prior_scale[var]), log_phi_2 + normal_lpdf(delta_eps_RA_epi[var] | 0, epi_prior_width_2*RA_epi_prior_scale[var]));
  }
  
  // prior on max output level
  log_g_max ~ normal(log_g_max_prior_mu, log_g_max_prior_std);
  // prior on min output level
  g_min ~ normal(g_min_prior_mu, g_min_prior_std);
  
  // prior on scale parameter for log-normal measurement error
  sigma ~ normal(sigma_prior_mu, sigma_prior_std);
  
  // model of the data (dose-response curve with noise)
  y_shifted ~ lognormal(log_mean_y, sigma);
  
// *****

// ***** include Free_energy_model.model.rep_ratio.stan
  // prior on scale hyper-parameter for log_rep_ratio
  rep_ratio_sigma ~ normal(rep_ratio_sigma_prior_mu, rep_ratio_sigma_prior_std);
  
  // prior on log_rep_ratio
  log_rep_ratio ~ normal(0, rep_ratio_sigma);
  
  // prior on rep_offset
  rep_offset ~ normal(0, offset_sigma);
  offset_sigma ~ normal(offset_sigma_prior_mu, offset_sigma_prior_std);
// *****

// ***** #include Free_energy_model.model.multi_operator.stan
  // priors on plasmid/operator and repressor dimer copy numbers
  log_copy_num ~ normal(log_copy_num_prior_mu, log_copy_num_prior_std);
  log_R ~ normal(log_R_prior_mu, log_R_prior_std);
// *****

}

generated quantities {

#include Free_energy_model.generated_quantities.shared.stan

#include Free_energy_model.generated_quantities.multi.stan
  
}
