// 
//

data {
  int<lower=1> N;        // number of data points
  int x[N];              // time points, i.e. plate number
  int n_reads[N];        // barcode read count for each time point
  int spike_in_reads[N]; // spike-in read count for each time point
  
  real<lower=0> tau[N];
  
  real slope_0_mu;
  real slope_0_sig;
  
  real<lower=0> alpha;
  
  real dilution_factor;
  real lower_bound_width;
  
}

transformed data {
  real log_starting_ratio;
  real min_delta_slope;
  
  min_delta_slope = -1*log(dilution_factor);
  
  log_starting_ratio = log(n_reads[1]) - log(spike_in_reads[1]);
}

parameters {
  real delta_slope;   // log_slope - slope_0
  real log_intercept;                        // intercept of (log(n_reads) - log(spike_in_reads)) - log(starting_ratio)
  
  real slope_0_tilda;         // initial slope (before addition of antibiotic), normalized
  
  real log_err_tilda[N];      // additional error term, normalized
  
}

transformed parameters {
  real n_mean[N];          // mean expected value for n_reads
  real log_err[N];         // additional error term
  real slope_0;            // initial slope (before addition of antibiotic)
  real log_slope;          // final slope ((after addition of antibiotic)
  
  slope_0 = slope_0_mu + slope_0_sig*slope_0_tilda;
  log_slope = slope_0 + delta_slope;
  
  for (i in 1:N) {
    log_err[i] = tau[i]*log_err_tilda[i];
    n_mean[i] = spike_in_reads[i]*exp(log_starting_ratio + log_intercept + log_slope*x[i] + (slope_0 - log_slope + (log_slope-slope_0)*exp(-x[i]*alpha))/alpha + log_err[i]);
  }
  
}

model {
  n_reads ~ poisson(n_mean);
  
  log_err_tilda ~ normal(0, 1);
  
  slope_0_tilda ~ normal(0, 1);
  
  // soft lower bound prior on delta_slope
  target += log1m(erf((min_delta_slope - delta_slope)/lower_bound_width));

}

generated quantities {
  real log_ratio_out[N];
  real ratio_out[N];
  real n_mean_out[N];
  
  for (i in 1:N) {
    n_mean_out[i] = spike_in_reads[i]*exp(log_starting_ratio + log_intercept + log_slope*x[i] + (slope_0 - log_slope + (log_slope - slope_0)*exp(-x[i]*alpha))/alpha);
	
    log_ratio_out[i] = log(n_mean_out[i]) - log(spike_in_reads[i]);
    ratio_out[i] = exp(log_ratio_out[i]);
  }
}
