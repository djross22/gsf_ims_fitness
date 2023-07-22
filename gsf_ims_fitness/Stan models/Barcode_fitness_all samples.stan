// This model takes in datasets for all samples in three groups:
//     The reference group ('_ref') are samples without antibiotic and low (or zero) ligand
//         The model assumes that the samples in the refernce group all have the same slope ('slope_ref')
//     The no_tet group ('_no_tet') are samples without antibiotic but higher ligand - 
//         So, they might have a differnt (lower) slope than the reference group
//     The reference group and no_tet group are fit with a simple linear model (actually, log-linear Poisson) 
//     The with_tet group ('_with_tet') are samples with antibiotic
//         They are fit with a bi-linear model with the intial slope equal to slope_ref 

data {
  int<lower=2> N;           // number of data points for each sample
  int<lower=1> M_ref;       // number of reference samples
  int<lower=1> M_no_tet;    // number of no_tet samples
  int<lower=1> M_with_tet;  // number of with_tet samples
  
  array[N] int x;                 // time points (i.e. plate number - 2)
  
  array[N, M_ref] int n_reads_ref;           // ref sample barcode read counts for each time point
  array[N, M_ref] int n_spike_ref;           // ref sample spike-in read count for each time point
  array[N, M_no_tet] int n_reads_no_tet;     // no_tet barcode read count for each time point
  array[N, M_no_tet] int n_spike_no_tet;     // no_tet spike-in read count for each time point
  array[N, M_with_tet] int n_reads_with_tet; // with_tet barcode read count for each time point
  array[N, M_with_tet] int n_spike_with_tet; // with_tet spike-in read count for each time point
  
  array[N] vector[M_ref] tau_ref;            // extra error term, used for de-weighting/ignoring samples
  array[N] vector[M_no_tet] tau_no_tet;      // extra error term, used for de-weighting/ignoring samples
  array[N] vector[M_with_tet] tau_with_tet;  // extra error term, used for de-weighting/ignoring samples
  
  real slope_ref_prior_std;  // scale for normal prior on ref_slope
  
  real<lower=0> alpha;       // decay rate from initial slope to final slope for bi-linear model  
  
  real dilution_factor;      // used to set a lower bound on the difference between the final slope and initial slope
  real lower_bound_width;    // width of erf used for soft lower bound on (final - initial) slope
  
}

transformed data {
  vector[M_ref] log_ratio_0_ref;
  vector[M_no_tet] log_ratio_0_no_tet;
  vector[M_with_tet] log_ratio_0_with_tet;
  
  array[N] vector[M_ref] log_ratio_in_ref;
  array[N] vector[M_no_tet] log_ratio_in_no_tet;
  array[N] vector[M_with_tet] log_ratio_in_with_tet;
  
  array[N] vector[M_ref] log_spike_ref;  // log of spike-in counts
  array[N] vector[M_no_tet] log_spike_no_tet;  // log of spike-in counts
  array[N] vector[M_with_tet] log_spike_with_tet;  // log of spike-in counts
  
  real min_delta_slope;
  
  min_delta_slope = -1*log(dilution_factor);
  
  for (m in 1:M_ref) {
	for (n in 1:N) {
	  log_spike_ref[n][m] = log(n_spike_ref[n, m]);
	  log_ratio_in_ref[n][m] = log(n_reads_ref[n, m] + 0.1) - log(n_spike_ref[n, m]);
	}
    log_ratio_0_ref[m] = log_ratio_in_ref[1][m];
  }
  for (m in 1:M_no_tet) {
	for (n in 1:N) {
	  log_spike_no_tet[n][m] = log(n_spike_no_tet[n, m]);
	  log_ratio_in_no_tet[n][m] = log(n_reads_no_tet[n, m] + 0.1) - log(n_spike_no_tet[n, m]);
	}
    log_ratio_0_no_tet[m] = log_ratio_in_no_tet[1][m];
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
  array[N] vector[M_ref] log_err_tilda_ref;  // additional error term, ref samples, normalized
  
  vector[M_no_tet] slope_no_tet;             // slope of no_tet samples
  vector[M_no_tet] intrc_no_tet;             // intercept of no_tet samples
  array[N] vector[M_no_tet] log_err_tilda_no_tet;  // additional error term, no_tet samples, normalized
  
  vector[M_with_tet] delta_slope_with_tet;       // slope_with_tet - slope_ref for with_tet samples
  vector[M_with_tet] intrc_with_tet;             // intercept of with_tet samples
  array[N] vector[M_with_tet] log_err_tilda_with_tet;  // additional error term, with_tet samples, normalized
  
}

transformed parameters {
  array[N] vector[M_ref] n_mean_ref;   // mean expected value for n_reads_ref
  array[N] vector[M_ref] log_err_ref;  // additional error term, ref samples
  
  array[N] vector[M_no_tet] n_mean_no_tet;   // mean expected value for n_reads_no_tet
  array[N] vector[M_no_tet] log_err_no_tet;  // additional error term, ref samples
  
  array[N] vector[M_with_tet] n_mean_with_tet;   // mean expected value for n_reads_with_tet
  array[N] vector[M_with_tet] log_err_with_tet;  // additional error term, ref samples
  
  vector[M_with_tet] slope_with_tet;       // final slope of with_tet samples (after addition of antibiotic)
  
  
  slope_with_tet = slope_ref + delta_slope_with_tet;
  
  for (n in 1:N) {
    for (m in 1:M_ref) {
      log_err_ref[n, m] = tau_ref[n, m]*log_err_tilda_ref[n, m];
	}
    for (m in 1:M_no_tet) {
      log_err_no_tet[n, m] = tau_no_tet[n, m]*log_err_tilda_no_tet[n, m];
	}
    for (m in 1:M_with_tet) {
      log_err_with_tet[n, m] = tau_with_tet[n, m]*log_err_tilda_with_tet[n, m];
	}
    
	n_mean_ref[n] = exp(log_spike_ref[n] + log_ratio_0_ref + intrc_ref + slope_ref*x[n] + log_err_ref[n]);
    
	n_mean_no_tet[n] = exp(log_spike_no_tet[n] + log_ratio_0_no_tet + intrc_no_tet + slope_no_tet*x[n] + log_err_no_tet[n]);
    
    n_mean_with_tet[n] = exp(log_spike_with_tet[n] + log_ratio_0_with_tet + intrc_with_tet + slope_with_tet*x[n] + (slope_ref - slope_with_tet + (slope_with_tet - slope_ref)*exp(-x[n]*alpha))/alpha + log_err_with_tet[n]);
  }
  
}

model {
  slope_ref ~ normal(0, slope_ref_prior_std);
  
  for (n in 1:N) {
    n_reads_ref[n] ~ poisson(n_mean_ref[n]);
	log_err_tilda_ref[n] ~ normal(0, 1);
    
    n_reads_no_tet[n] ~ poisson(n_mean_no_tet[n]);
	log_err_tilda_no_tet[n] ~ normal(0, 1);
    
    n_reads_with_tet[n] ~ poisson(n_mean_with_tet[n]);
	log_err_tilda_with_tet[n] ~ normal(0, 1);
  }
  
  for (m in 1:M_with_tet) {
    // soft lower bound prior on delta_slope
    target += log1m(erf((min_delta_slope - delta_slope_with_tet[m])/lower_bound_width));
  }
  

}

generated quantities {
  array[N] vector[M_ref] log_ratio_out_ref;
  array[N] vector[M_ref] ratio_out_ref;
  array[N] vector[M_ref] n_mean_out_ref;
  array[N] vector[M_ref] residuals_ref;
  
  array[N] vector[M_no_tet] log_ratio_out_no_tet;
  array[N] vector[M_no_tet] ratio_out_no_tet;
  array[N] vector[M_no_tet] n_mean_out_no_tet;
  array[N] vector[M_no_tet] residuals_no_tet;
  
  array[N] vector[M_with_tet] log_ratio_out_with_tet;
  array[N] vector[M_with_tet] ratio_out_with_tet;
  array[N] vector[M_with_tet] n_mean_out_with_tet;
  array[N] vector[M_with_tet] residuals_with_tet;
  
  for (n in 1:N) {
    log_ratio_out_ref[n] = log(n_mean_ref[n]) - log_spike_ref[n] - log_err_ref[n];
    ratio_out_ref[n] = exp(log_ratio_out_ref[n]);
	n_mean_out_ref[n] = exp(log_ratio_out_ref[n] + log_spike_ref[n]);
	residuals_ref[n] = log_ratio_in_ref[n] - log_ratio_out_ref[n];
	
    log_ratio_out_no_tet[n] = log(n_mean_no_tet[n]) - log_spike_no_tet[n] - log_err_no_tet[n];
    ratio_out_no_tet[n] = exp(log_ratio_out_no_tet[n]);
	n_mean_out_no_tet[n] = exp(log_ratio_out_no_tet[n] + log_spike_no_tet[n]);
	residuals_no_tet[n] = log_ratio_in_no_tet[n] - log_ratio_out_no_tet[n];
	
    log_ratio_out_with_tet[n] = log(n_mean_with_tet[n]) - log_spike_with_tet[n] - log_err_with_tet[n];
    ratio_out_with_tet[n] = exp(log_ratio_out_with_tet[n]);
	n_mean_out_with_tet[n] = exp(log_ratio_out_with_tet[n] + log_spike_with_tet[n]);
	residuals_with_tet[n] = log_ratio_in_with_tet[n] - log_ratio_out_with_tet[n];
  }
}
