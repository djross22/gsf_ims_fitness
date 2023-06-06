
//transformed data {

  // transformed data variable declarations shared by all models
  real ln_10;
  real hill_n;
  real N_NS;
  vector[19] x_out;
  real log_phi_1;
  real log_phi_2;
  real min_y;
  
  hill_n = 2;
  
  x_out[1] = 0;
  for (i in 2:19) {
    x_out[i] = 2^(i-2);
  }
    
  ln_10 = log(10.0);
  
  min_y = min(y);
  
//}

