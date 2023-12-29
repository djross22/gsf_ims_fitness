
//transformed parameters {
  
  log_k_a_var[1] = log_k_a_wt;
  log_k_i_var[1] = log_k_i_wt;
  delta_eps_AI_var[1] = delta_eps_AI_wt;
  delta_eps_RA_var[1] = delta_eps_RA_wt;
  
  K_A[1] = 10^log_k_a_var[1];
  K_I[1] = 10^log_k_i_var[1];
  
  for (variant_num in 2:num_var) {
    if (variant_num > num_non_epi_var) {
      log_k_a_var[variant_num] = log_k_a_wt + log_k_a_epi[variant_num-num_non_epi_var];
      log_k_i_var[variant_num] = log_k_i_wt + log_k_i_epi[variant_num-num_non_epi_var];
      delta_eps_AI_var[variant_num] = delta_eps_AI_wt + delta_eps_AI_epi[variant_num-num_non_epi_var];
      delta_eps_RA_var[variant_num] = delta_eps_RA_wt + delta_eps_RA_epi[variant_num-num_non_epi_var];
	}
    else {
      log_k_a_var[variant_num] = log_k_a_wt;
      log_k_i_var[variant_num] = log_k_i_wt;
      delta_eps_AI_var[variant_num] = delta_eps_AI_wt;
      delta_eps_RA_var[variant_num] = delta_eps_RA_wt;
	}
	
	for (mut in 1:num_mut) {
	  log_k_a_var[variant_num] += mut_code[variant_num-1, mut]*log_k_a_mut[mut];
	  log_k_i_var[variant_num] += mut_code[variant_num-1, mut]*log_k_i_mut[mut];
	  delta_eps_AI_var[variant_num] += mut_code[variant_num-1, mut]*delta_eps_AI_mut[mut];
	  delta_eps_RA_var[variant_num] += mut_code[variant_num-1, mut]*delta_eps_RA_mut[mut];
	}
	
	K_A[variant_num] = 10^log_k_a_var[variant_num];
    K_I[variant_num] = 10^log_k_i_var[variant_num];
  }
  
//}

