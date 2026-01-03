
//transformed parameters {
  
  for (i in 1:N) {
	real fold_change;
    real c1;
    real c2;
    real c3;
	
    c1 = (1 + x[i]/K_A[variant[i]])^hill_n;
    c2 = ( (1 + x[i]/K_I[variant[i]])^hill_n ) * exp(-delta_eps_AI_var[variant[i]]);
    c3 = R/N_NS * exp(-delta_eps_RA_var[variant[i]]);
	
    fold_change = 1/(1 + (c1/(c1+c2))*c3);
	
    log_mean_y[i] = ln_10*log_g_max + log(fold_change) + log_rep_ratio[rep[i]];
	
    // measured values with g_min and rep_offset subtracted
    y_shifted[i] = y[i] - g_min - rep_offset[rep[i]];
	
  }
  
  for (i in 1:N_contr) {
    log_mean_y_contr[i] = ln_10*log_g_max + log_rep_ratio_contr[rep_contr[i]];
	
    y_contr_shifted[i] = y_contr[i] - g_min - rep_offset_contr[rep_contr[i]];
  }
  
//}

