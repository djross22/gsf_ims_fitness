
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
	
    log_mean_y[i] = ln_10*log_g_max + log(fold_change);
  }
  
  for (i in 1:N_contr) {
    log_mean_y_contr[i] = ln_10*log_g_max;
  }
  
  // measured values with g_min subtracted
  y_shifted = y - g_min;
  y_contr_shifted = y_contr - g_min;
  
//}

