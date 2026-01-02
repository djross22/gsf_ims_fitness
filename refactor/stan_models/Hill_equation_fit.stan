// 
//

data {
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // gene expression (from cytometry) at each concentration
  vector[N] y_err;       // estimated error of gene expression at each concentration
  real log_g_min;        // lower bound on log_g0 and log_ginf
  real log_g_max;        // upper bound on log_g0 and log_ginf
  real log_ec50_min;
  real log_ec50_max;
  real<lower=0> sensor_n_alpha;   //prior shape
  real<lower=0> sensor_n_sigma;   //prior scale
  
  real baseline_mu;             // mean value of non-fluorescent control
  real baseline_sig;            //uncertainty of non-fluorescent control
}

transformed data {
  real max_x_in;    // max inducer concentration
  real min_x_in;    // min non-zero inducer concentration
  real log_x_out_spacing;
  vector[30] x_out;
  
  max_x_in = max(x);
  min_x_in = max_x_in;
  for (i in 1:N) {
    if (x[i]!=0) {
	  if (x[i]<min_x_in) {
	    min_x_in = x[i];
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
  real baseline;                                           //non-fluorescent control 
  
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
  
  vector[N] mean_y;
  
  ec50 = 10^log_ec50;
  g0 = 10^log_g0;
  ginf = 10^log_ginf;
  
  for (i in 1:N) {
    mean_y[i] = g0 + (ginf - g0)*(x[i]^sensor_n)/(ec50^sensor_n + x[i]^sensor_n);
  }
  
}

model {
  // prior on baseline, informed by measurements of control strain
  baseline ~ normal(baseline_mu, baseline_sig);
  
  // Prior on sensor_n; <weibull> ~ sensor_n_sigma*GammaFunct(1+1/sensor_n_alpha)
  sensor_n ~ weibull(sensor_n_alpha, sensor_n_sigma);
  //sensor_n ~ gamma(4.0, 10.0/3.0); older version
  //sensor_n ~ gamma(sensor_n_alpha, sensor_n_beta); not quite so-older version
  
  // noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);
  
  y ~ normal(mean_y + baseline, sigma*y_err);

}

generated quantities {
  real log_sensor_n;
  real rms_resid;
  real log_ginf_g0_ratio;
  vector[30] y_out;
  real max_level;        // output at maximum  value of x that was measured
  
  vector[N] y_resid;
  
  for (i in 1:30) {
    y_out[i] = g0 + (ginf - g0)*(x_out[i]^sensor_n)/(ec50^sensor_n + x_out[i]^sensor_n);
  }
  
  max_level = g0 + (ginf - g0)*(max_x_in^sensor_n)/(ec50^sensor_n + max_x_in^sensor_n);
  
  log_ginf_g0_ratio = log_ginf - log_g0;
  
  log_sensor_n = log10(sensor_n);
  
  rms_resid = distance(y, mean_y + baseline)/sqrt(N);
  
  for (i in 1:N) {
    y_resid[i] = y[i] - mean_y[i] - baseline;
  }
  
}
