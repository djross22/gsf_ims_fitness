// 
//

data {
  int<lower=1> N;        // number of data points
  int x[N];              // time points, i.e. plate number
  int n_reads[N];        // barcode read count for each time point
  int spike_in_reads[N]; // spike-in read count for each time point
  
  real<lower=0> tau[N];
  
}

transformed data {
  real log_starting_ratio;
  
  if (n_reads[1] == 0) {
    log_starting_ratio = log(0.1) - log(spike_in_reads[1]);
  } else {
    log_starting_ratio = log(n_reads[1]) - log(spike_in_reads[1]);
  }
}

parameters {
  real log_slope;       // slope of log(n_reads) - log(spike_in_reads)
  real log_intercept;   // intercept of (log(n_reads) - log(spike_in_reads)) - log(starting_ratio)
  
  real log_err_tilda[N];      // additional error term, normalized
}

transformed parameters {
  real n_mean[N];          // mean expected value for n_reads
  real log_err[N];         // additional error term
  
  for (i in 1:N) {
    log_err[i] = tau[i]*log_err_tilda[i];
    n_mean[i] = spike_in_reads[i]*exp(log_starting_ratio + log_intercept + log_slope*x[i] + log_err[i]);
  }
  
}

model {
  n_reads ~ poisson(n_mean);
  
  log_err_tilda ~ normal(0, 1);

}

generated quantities {
  real log_ratio_out[N];
  real ratio_out[N];
  real n_mean_out[N];
  
  for (i in 1:N) {
    n_mean_out[i] = spike_in_reads[i]*exp(log_starting_ratio + log_intercept + log_slope*x[i]);
    log_ratio_out[i] = log(n_mean_out[i]) - log(spike_in_reads[i]);
    ratio_out[i] = exp(log_ratio_out[i]);
  }
}
