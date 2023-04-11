
//transformed parameters {

  vector[num_var] K_A;
  vector[num_var] K_I;
  vector[num_var] log_k_a_var;
  vector[num_var] log_k_i_var;
  vector[num_var] delta_eps_AI_var;
  vector[num_var] delta_eps_RA_var;
  
  vector[N] log_mean_y;
  vector[N_contr] log_mean_y_contr;
  
  // measured values with g_min subtracted
  vector[N] y_shifted;
  vector[N_contr] y_contr_shifted;
  
//}

