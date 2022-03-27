
//transformed parameters {

  g_max = 10^log_g_max;
  
  for (i in 1:N) {
	real fold_change;
  
    fold_change = fold_change_fnct(x[i], K_A[variant[i]], K_I[variant[i]], delta_eps_AI_var[variant[i]], delta_eps_RA_var[variant[i]], hill_n, N_NS, R, N_S);
	
    mean_y[i] = g_max*fold_change;
  }
  
  for (i in 1:N_contr) {
    mean_y_contr[i] = g_max;
  }
  
//}

