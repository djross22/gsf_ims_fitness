
//data {

  // Input data for priors on wild-type free energy parameters (single- and multi-operator models)
  real log_k_a_wt_prior_mean;
  real log_k_a_wt_prior_std;
  real log_k_i_wt_prior_mean;
  real log_k_i_wt_prior_std;
  real delta_eps_AI_wt_prior_mean;
  real delta_eps_AI_wt_prior_std;
  real delta_eps_RA_wt_prior_mean;
  real delta_eps_RA_wt_prior_std;
  
  // priors on mutational effects and epistasis  (single- and multi-operator models)
  real delta_prior_width;   // width of prior on "_mut" parameters (mutational effects)
  
  real<lower=0> eps_RA_prior_scale[num_mut];     // scale multiplier for width of priors on operator binding free energy term (delta_eps_RA_mut) 
  
//}

