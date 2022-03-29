
//generated quantities {
  
  for (var in 1:num_var) {
    for (i in 1:19) {
	  real c1;
      real c2;
      real c3;
	  
      c1 = (1 + x_out[i]/K_A[var])^hill_n;
      c2 = ( (1 + x_out[i]/K_I[var])^hill_n ) * exp(-delta_eps_AI_var[var]);
      c3 = R/N_NS * exp(-delta_eps_RA_var[var]);
	
      y_out[var, i] = g_max/(1 + (c1/(c1+c2))*c3) + g_min;
      fc_out[var, i] = 1/(1 + (c1/(c1+c2))*c3);
    }
  }
  
//}

