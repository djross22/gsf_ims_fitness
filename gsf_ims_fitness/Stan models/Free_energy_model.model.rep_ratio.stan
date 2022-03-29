
//model {
  
  // prior on scale hyper-paramter for log_rep_ratio
  rep_ratio_sigma ~ normal(0, rep_ratio_scale);
  rep_offset_sigma ~ normal(0, rep_offset_scale);
  
  // priors on log_rep_ratio and log_rep_ratio_contr
  log_rep_ratio ~ normal(0, 1);
  log_rep_ratio_contr ~ normal(0, 1);
  
  // priors on rep_offset and rep_offset_contr
  rep_offset ~ normal(0, 1);
  rep_offset_contr ~ normal(0, 1);
  
//}

