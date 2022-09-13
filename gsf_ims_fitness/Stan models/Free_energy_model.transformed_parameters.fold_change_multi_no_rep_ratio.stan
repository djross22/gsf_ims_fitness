
//transformed parameters {
  
  for (i in 1:N) {
    real c1;
    real c2;
    real pA;
	real xRA;
	real lam;
	real fold_change;
	
    c1 = (1 + x[i]/K_A[variant[i]])^hill_n;
    c2 = ( (1 + x[i]/K_I[variant[i]])^hill_n ) * exp(-delta_eps_AI_var[variant[i]]);
	pA = c1/(c1+c2);
	xRA = exp(-delta_eps_RA_var[variant[i]]);
	
	lam = -N_NS + pA*R - N_S*xRA + pA*R*xRA;
    lam = lam + sqrt(4*pA*R*xRA*(N_NS + N_S - pA*R) + (N_NS + N_S*xRA - pA*R*(1 + xRA))^2);
    lam = lam/(2*xRA*(N_NS + N_S - pA*R));
	
    fold_change = 1/(1 + lam*xRA);
	
    log_mean_y[i] = ln_10*log_g_max + log(fold_change);
  }
  
  for (i in 1:N_contr) {
    log_mean_y_contr[i] = ln_10*log_g_max;
  }
  
  // measured values with g_min subtracted
  y_shifted = y - g_min;
  y_contr_shifted = y_contr - g_min;
  y_g_min_shifted = y_g_min - g_min;
  
//}

