
//transformed parameters {

  real N_S;
  real R;

  N_S = 10^log_copy_num;
  R = 10^log_R;
  
  // measured values with g_min subtracted
  y_shifted = y - g_min;
  y_contr_shifted = y_contr - g_min;
  
//}

