
//transformed data {

  // transformed data variable assignments shared by all models
  hill_n = 2;
  
  x_out[1] = 0;
  for (i in 2:19) {
    x_out[i] = 2^(i-2);
  }
  
  num_non_epi_var = num_var - num_epi_var;
  
  ln_10 = log(10.0);
  
  min_y = min(y);
  
  g_min = g_min_prior_mu;
  
//}

