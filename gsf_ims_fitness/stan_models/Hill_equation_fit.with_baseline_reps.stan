// 
//

data {
  int<lower=1> N_reps;               // number of replicate measurements
  
  array[N_reps] int<lower=1> N;      // number of data points for each replicate
  int N_max;                         // maximum number of data points for a replicate
  
  array[N_reps] vector[N_max] x;     // inducer concentration, padded with inf if reps have different numbers of points
  array[N_reps] vector[N_max] y;     // gene expression (from cytometry) at each concentration
  array[N_reps] vector[N_max] y_err; // estimated error of gene expression at each concentration
  
  real log_g_min;        // lower bound on log_g0 and log_ginf
  real log_g_max;        // upper bound on log_g0 and log_ginf
  real log_ec50_min;
  real log_ec50_max;
  real<lower=0> sensor_n_alpha;   //prior shape
  real<lower=0> sensor_n_sigma;   //prior scale
  
  array[N_reps] real baseline_mu;             // mean value of non-fluorescent control
  array[N_reps] real baseline_sig;            //uncertainty of non-fluorescent control
}

transformed data {
  real max_x_in;    // max inducer concentration
  real min_x_in;    // min non-zero inducer concentration
  real log_x_out_spacing;
  vector[30] x_out;
  
  min_x_in = 10^20;
  max_x_in = -1;
  
  for (rep in 1:N_reps) {
    for (i in 1:N[rep]) {
      if (x[rep][i]!=0) {
	    if (x[rep][i]<min_x_in) {
	      min_x_in = x[rep][i];
	    }
	  }
	  
	  if (x[rep][i]>max_x_in) {
	    max_x_in = x[rep][i];
	  }
	  
    }
  }
  log_x_out_spacing = (log(max_x_in*2) - log(min_x_in/2))/28.0;
  
  x_out[1] = 0;
  x_out[2] = min_x_in/2;
  for (i in 3:30) {
    x_out[i] = x_out[i-1]*exp(log_x_out_spacing);
  }
  
}

parameters {
  array[N_reps] real baseline;                             // non-fluorescent controls 
  
  real<lower=log_g_min, upper=log_g_max> log_g0;           // log10 of gene expression level at zero induction
  real<lower=log_g_min, upper=log_g_max> log_ginf;         // log10 of gene expression level at infinite induction
  
  real<lower=log_ec50_min, upper=log_ec50_max> log_ec50;   // input level (x) that gives output 1/2 way between g0 and ginf
  real<lower=0> sensor_n;                                  // cooperativity exponent of sensor gene expression vs. x curve
  
  real<lower=0> sigma;                                     // scale factor for standard deviation of noise in y

}

transformed parameters {
  real g0;        
  real ginf; 
  real ec50;
  
  array[N_reps] vector[N_max] mean_y;  
  
  ec50 = 10^log_ec50;
  g0 = 10^log_g0;
  ginf = 10^log_ginf;
  
  for (rep in 1:N_reps) {
    for (i in 1:N[rep]) {
      mean_y[rep][i] = g0 + (ginf - g0)*(x[rep][i]^sensor_n)/(ec50^sensor_n + x[rep][i]^sensor_n);
    }
  }
  
}

model {
  // Prior on sensor_n; <weibull> ~ sensor_n_sigma*GammaFunct(1+1/sensor_n_alpha)
  sensor_n ~ weibull(sensor_n_alpha, sensor_n_sigma);
  
  // noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);
  
  for (rep in 1:N_reps) {
    // prior on baseline, informed by measurements of control strain
    baseline[rep] ~ normal(baseline_mu[rep], baseline_sig);
    
	for (i in 1:N[rep]) {
	  y[rep][i] ~ normal(mean_y[rep][i] + baseline[rep], sigma*y_err[rep][i]);
	}
  }

}

generated quantities {
  real log_sensor_n;
  real rms_resid;
  real log_ginf_g0_ratio;
  vector[30] y_out;
  real max_level;        // output at maximum  value of x that was measured
  
  array[N_reps] vector[N_max] y_resid;  
  
  for (i in 1:30) {
    y_out[i] = g0 + (ginf - g0)*(x_out[i]^sensor_n)/(ec50^sensor_n + x_out[i]^sensor_n);
  }
  
  max_level = g0 + (ginf - g0)*(max_x_in^sensor_n)/(ec50^sensor_n + max_x_in^sensor_n);
  
  log_ginf_g0_ratio = log_ginf - log_g0;
  
  log_sensor_n = log10(sensor_n);
  
  rms_resid = 0; //distance(y, mean_y + baseline)/sqrt(N);
  {
  int N_total;
  N_total = 0;
    for (rep in 1:N_reps) {
      for (i in 1:N[rep]) {
        y_resid[rep][i] = y[rep][i] - mean_y[rep][i] - baseline[rep];
	    rms_resid = rms_resid + ((y[rep][i] - mean_y[rep][i] - baseline[rep])^2)/N[rep];
		N_total = N_total + 1;
      }
    }
	rms_resid = sqrt(rms_resid/N_total);
  }
  
}
