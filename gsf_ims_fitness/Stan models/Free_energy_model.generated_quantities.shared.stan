
//generated quantities {

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
  
//}

