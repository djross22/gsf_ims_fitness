
//generated quantities {
  
  for (variant_num in 1:num_var) {
    for (i in 1:19) {
	  real c1;
      real c2;
      real c3;
	  
      c1 = (1 + x_out[i]/K_A[variant_num])^hill_n;
      c2 = ( (1 + x_out[i]/K_I[variant_num])^hill_n ) * exp(-delta_eps_AI_var[variant_num]);
      c3 = R/N_NS * exp(-delta_eps_RA_var[variant_num]);
	
      y_out[variant_num, i] = g_max/(1 + (c1/(c1+c2))*c3);
      fc_out[variant_num, i] = 1/(1 + (c1/(c1+c2))*c3);
    }
  }
  
//}

