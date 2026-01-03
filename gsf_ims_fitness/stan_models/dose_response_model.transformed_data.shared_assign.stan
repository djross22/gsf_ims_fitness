
//transformed data {
  
  hill_n = 2;
  
  x_out[1] = 0;
  for (i in 2:19) {
    x_out[i] = 2^(i-2);
  }
    
  ln_10 = log(10.0);
  
  min_y = min(y);
  
  g_min = g_min_prior_mu;
  
//}

