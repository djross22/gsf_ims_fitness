
//generated quantities {
  
  for (var in 1:num_var) {
    for (i in 1:19) {
      real c1;
      real c2;
      real pA;
	  real xRA;
	  real lam;
	  real f_c;
	  
      c1 = (1 + x_out[i]/K_A[var])^hill_n;
      c2 = ( (1 + x_out[i]/K_I[var])^hill_n ) * exp(-delta_eps_AI_var[var]);
	  pA = c1/(c1+c2);
	  xRA = exp(-delta_eps_RA_var[var]);
	
	  lam = -N_NS + pA*R - N_S*xRA + pA*R*xRA;
      lam = lam + sqrt(4*pA*R*xRA*(N_NS + N_S - pA*R) + (N_NS + N_S*xRA - pA*R*(1 + xRA))^2);
      lam = lam/(2*xRA*(N_NS + N_S - pA*R));
	
      f_c = 1/(1 + lam*xRA);
	
      y_out[var, i] = g_max*f_c;
      fc_out[var, i] = f_c;
    }
  }
  
//}

