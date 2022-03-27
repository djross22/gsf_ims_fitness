
//transformed parameters {
  
  for (i in 1:N) {
	real fold_change;
  
    fold_change = fold_change_fnct(x[i], K_A[variant[i]], K_I[variant[i]], delta_eps_AI_var[variant[i]], delta_eps_RA_var[variant[i]], hill_n, N_NS, R, N_S);
	
    log_mean_y[i] = ln_10*log_g_max + log(fold_change);
  }
  
  for (i in 1:N_contr) {
    log_mean_y_contr[i] = ln_10*log_g_max;
  }
  
//}

