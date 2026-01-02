
//parameters {

  real log_k_a_wt;         // log10 of IPTG binding affinity to active state
  vector[num_mut] log_k_a_mut;
  vector[num_epi_var] log_k_a_epi;
  
  real log_k_i_wt;         // log10 of IPTG binding affinity to inactive state
  vector[num_mut] log_k_i_mut;
  vector[num_epi_var] log_k_i_epi;
  
  real delta_eps_AI_wt;    // free energy difference between active and inactive states
  vector[num_mut] delta_eps_AI_mut;
  vector[num_epi_var] delta_eps_AI_epi;
  
  real delta_eps_RA_wt;    // free energy for Active TF to operator
  vector[num_mut] delta_eps_RA_mut;
  vector[num_epi_var] delta_eps_RA_epi;
  
  real log_g_max;       // log10 of maximum possible gene expression
  
  real<lower=0> sigma;      // scale factor for standard deviation of noise in log_y
  
//}

