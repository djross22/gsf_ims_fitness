
//generated quantities {
  
  for (variant_num in 1:num_var) {
    for (i in 1:19) {
      real c1;
      real c2;
      real pA;
	  real xRA;
	  real lam;
	  real f_c;
	  
      c1 = (1 + x_out[i]/K_A[variant_num])^hill_n;
      c2 = ( (1 + x_out[i]/K_I[variant_num])^hill_n ) * exp(-delta_eps_AI_var[variant_num]);
	  pA = c1/(c1+c2);
	  xRA = exp(-delta_eps_RA_var[variant_num]);
	
	  lam = -N_NS + pA*R - N_S*xRA + pA*R*xRA;
      lam = lam + sqrt(4*pA*R*xRA*(N_NS + N_S - pA*R) + (N_NS + N_S*xRA - pA*R*(1 + xRA))^2);
      lam = lam/(2*xRA*(N_NS + N_S - pA*R));
	
      f_c = 1/(1 + lam*xRA);
	
      y_out[variant_num, i] = g_max*f_c;
      fc_out[variant_num, i] = f_c;
    }
  }
  
//}

