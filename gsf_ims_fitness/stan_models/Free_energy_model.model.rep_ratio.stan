
//model {
  
  // prior on scale hyper-parameters for log_rep_ratio and rep_offset
  rep_ratio_sigma ~ normal(0, rep_ratio_scale);
  offset_sigma ~ normal(0, rep_offset_scale);
  
  // priors on log_rep_ratio and log_rep_ratio_contr
  log_rep_ratio ~ normal(0, rep_ratio_sigma);
  log_rep_ratio_contr ~ normal(0, rep_ratio_sigma);
  
  // priors on rep_offset, rep_offset_contr
  rep_offset ~ normal(0, offset_sigma);
  rep_offset_contr ~ normal(0, offset_sigma);
  
//}

