
//model {

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
  
//}

