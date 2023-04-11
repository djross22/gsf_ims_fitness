
//parameters {
  
  vector[num_reps] log_rep_ratio;              // log10 of multiplicative correction factor for different replicates
  vector[num_contr_reps] log_rep_ratio_contr;  // log10 of multiplicative correction factor for control replicates
  real<lower=0> rep_ratio_sigma;               // hyper-parameters for log_rep_ratio and log_rep_ratio_contr
  
  real<lower=0> offset_sigma;  // hyper-parameter for standard deviation of replicate variability in g_min
  
  vector<lower=-3*rep_offset_scale, upper=3*rep_offset_scale>[num_reps] rep_offset;              // additional g_min shift for different replicates
  vector<lower=-3*rep_offset_scale, upper=3*rep_offset_scale>[num_contr_reps] rep_offset_contr;  // additional g_min shift for control replicates
  
//}

