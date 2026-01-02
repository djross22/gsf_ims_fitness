// This model takes in datasets for all samples in two groups:
//     The reference group ('_ref') are samples without antibiotic and low (or zero) ligand
//         The model assumes that the samples in the refernce group all have the same slope ('slope_ref')
//     The reference group is fit with a simple linear model (actually, log-linear Poisson) 
//     The with_tet group ('_with_tet') are samples with antibiotic
//         They are fit with a bi-linear model with the intial slope equal to slope_ref 

data {
  int<lower=2> N;           // number of data points for each sample
  int<lower=1> M_ref;       // number of reference samples
  int<lower=1> M_with_tet;  // number of with_tet samples
  
  int x[N];                 // time points (i.e. plate number - 2)
  
  int n_reads_ref[N, M_ref];           // ref sample barcode read counts for each time point
  int n_spike_ref[N, M_ref];           // ref sample spike-in read count for each time point
  int n_reads_with_tet[N, M_with_tet]; // with_tet barcode read count for each time point
  int n_spike_with_tet[N, M_with_tet]; // with_tet spike-in read count for each time point
  
  vector[M_ref] tau_ref[N];            // extra error term, used for de-weighting/ignoring samples
  vector[M_with_tet] tau_with_tet[N];  // extra error term, used for de-weighting/ignoring samples
  
  real slope_ref_prior_std;  // scale for normal prior on ref_slope
  
  real<lower=0> alpha;       // decay rate from initial slope to final slope for bi-linear model  
  
  real dilution_factor;      // used to set a lower bound on the difference between the final slope and initial slope
  real lower_bound_width;    // width of erf used for soft lower bound on (final - initial) slope
  
}

transformed data {
  vector[M_ref] log_ratio_0_ref;
  vector[M_with_tet] log_ratio_0_with_tet;
  
  vector[M_ref] log_ratio_in_ref[N];
  vector[M_with_tet] log_ratio_in_with_tet[N];
  
  vector[M_ref] log_spike_ref[N];  // log of spike-in counts
  vector[M_with_tet] log_spike_with_tet[N];  // log of spike-in counts
  
  real min_delta_slope;
  
  min_delta_slope = -1*log(dilution_factor);
  
  for (m in 1:M_ref) {
	for (n in 1:N) {
	  log_spike_ref[n][m] = log(n_spike_ref[n, m]);
	  log_ratio_in_ref[n][m] = log(n_reads_ref[n, m] + 0.1) - log(n_spike_ref[n, m]);
	}
    log_ratio_0_ref[m] = log_ratio_in_ref[1][m];
  }

  for (m in 1:M_with_tet) {
	for (n in 1:N) {
	  log_spike_with_tet[n][m] = log(n_spike_with_tet[n, m]);
	  log_ratio_in_with_tet[n][m] = log(n_reads_with_tet[n, m] + 0.1) - log(n_spike_with_tet[n, m]);
	}
    log_ratio_0_with_tet[m] = log_ratio_in_with_tet[1][m];
  }
}

parameters {
  real slope_ref;                      // reference slope
  vector[M_ref] intrc_ref;             // intercept of ref samples
  vector[M_ref] log_err_tilda_ref[N];  // additional error term, ref samples, normalized
  
  vector[M_with_tet] delta_slope_with_tet;       // slope_with_tet - slope_ref for with_tet samples
  vector[M_with_tet] intrc_with_tet;             // intercept of with_tet samples
  vector[M_with_tet] log_err_tilda_with_tet[N];  // additional error term, with_tet samples, normalized
  
}

transformed parameters {
  vector[M_ref] n_mean_ref[N];   // mean expected value for n_reads_ref
  vector[M_ref] log_err_ref[N];  // additional error term, ref samples
  
  vector[M_with_tet] n_mean_with_tet[N];   // mean expected value for n_reads_with_tet
  vector[M_with_tet] log_err_with_tet[N];  // additional error term, ref samples
  
  vector[M_with_tet] slope_with_tet;       // final slope of with_tet samples (after addition of antibiotic)
  
  
  slope_with_tet = slope_ref + delta_slope_with_tet;
  
  for (n in 1:N) {
    for (m in 1:M_ref) {
      log_err_ref[n, m] = tau_ref[n, m]*log_err_tilda_ref[n, m];
	}
    for (m in 1:M_with_tet) {
      log_err_with_tet[n, m] = tau_with_tet[n, m]*log_err_tilda_with_tet[n, m];
	}
    
	n_mean_ref[n] = exp(log_spike_ref[n] + log_ratio_0_ref + intrc_ref + slope_ref*x[n] + log_err_ref[n]);
    
    n_mean_with_tet[n] = exp(log_spike_with_tet[n] + log_ratio_0_with_tet + intrc_with_tet + slope_with_tet*x[n] + (slope_ref - slope_with_tet + (slope_with_tet - slope_ref)*exp(-x[n]*alpha))/alpha + log_err_with_tet[n]);
  }
  
}

model {
  slope_ref ~ normal(0, slope_ref_prior_std);
  
  for (n in 1:N) {
    n_reads_ref[n] ~ poisson(n_mean_ref[n]);
	log_err_tilda_ref[n] ~ normal(0, 1);
    
    n_reads_with_tet[n] ~ poisson(n_mean_with_tet[n]);
	log_err_tilda_with_tet[n] ~ normal(0, 1);
  }
  
  for (m in 1:M_with_tet) {
    // soft lower bound prior on delta_slope
    target += log1m(erf((min_delta_slope - delta_slope_with_tet[m])/lower_bound_width));
  }
  

}

generated quantities {
  vector[M_ref] log_ratio_out_ref[N];
  vector[M_ref] ratio_out_ref[N];
  vector[M_ref] n_mean_out_ref[N];
  vector[M_ref] residuals_ref[N];
  
  vector[M_with_tet] log_ratio_out_with_tet[N];
  vector[M_with_tet] ratio_out_with_tet[N];
  vector[M_with_tet] n_mean_out_with_tet[N];
  vector[M_with_tet] residuals_with_tet[N];
  
  for (n in 1:N) {
    log_ratio_out_ref[n] = log(n_mean_ref[n]) - log_spike_ref[n] - log_err_ref[n];
    ratio_out_ref[n] = exp(log_ratio_out_ref[n]);
	n_mean_out_ref[n] = exp(log_ratio_out_ref[n] + log_spike_ref[n]);
	residuals_ref[n] = log_ratio_in_ref[n] - log_ratio_out_ref[n];
	
    log_ratio_out_with_tet[n] = log(n_mean_with_tet[n]) - log_spike_with_tet[n] - log_err_with_tet[n];
    ratio_out_with_tet[n] = exp(log_ratio_out_with_tet[n]);
	n_mean_out_with_tet[n] = exp(log_ratio_out_with_tet[n] + log_spike_with_tet[n]);
	residuals_with_tet[n] = log_ratio_in_with_tet[n] - log_ratio_out_with_tet[n];
  }
}
