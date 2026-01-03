
//data {
  
  // priors on wild-type Hill parameters
  real logit_g0_wt_prior_mean;
  real logit_g0_wt_prior_std;
  
  real logit_ginf_wt_prior_mean;
  real logit_ginf_wt_prior_std;
  
  real log_ec50_wt_prior_mean;
  real log_ec50_wt_prior_std;
  
  real logit_n_eff_wt_prior_mean;
  real logit_n_eff_wt_prior_std;
  
  // priors on mutational effects and epistasis for Hill parameters
  real delta_prior_width_hill;   // width of prior on delta-parameters
  real epi_prior_width_1_hill;   // width of 1st mixture component of prior on parameter epistasis
  real epi_prior_width_2_hill;   // width of 2nd mixture component of prior on parameter epistasis
  real epi_prior_phi_hill;       // weight for 1st mixture component of prior on parameter epistasis
  
//}

