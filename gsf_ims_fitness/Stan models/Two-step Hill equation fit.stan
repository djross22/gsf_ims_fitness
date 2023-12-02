// 
//

data {
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // gene expression (from cytometry) at each concentration
  vector[N] y_err;       // estimated error of gene expression at each concentration
  real log_g_min;        // lower bound on log_low_level and log_high_level
  real log_g_max;        // upper bound on log_low_level and log_high_level
  real log_x_min;
  real log_x_max;
  real<lower=0> sensor_n_alpha;  //prior shape
  real<lower=0> sensor_n_beta;   //prior inverse scale
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
  real<lower=log_g_min, upper=log_g_max> log_low_level;               // log10 of gene expression level at zero induction
  real<lower=log_g_min, upper=log_g_max> log_mid_level;               // log10 of gene expression level at top of first step
  real<lower=log_g_min, upper=log_g_max> log_high_level;             // log10 of gene expression level at infinite induction
  
  ordered[2] log_IC_50;   // input level (x) that gives output 1/2 way between low_level and mid_level or between mid_level and high_level
  real<lower=0> sensor_n;                                             // cooperativity exponent of sensor gene expression vs. x curve, assumed to be the same for both steps
  real<lower=0> sigma;                                                // scale factor for standard deviation of noise in y

}

transformed parameters {
  real low_level;        
  real mid_level;        
  real high_level; 
  real IC_50_1;
  real IC_50_2;
  vector[N] mean_y;
  
  IC_50_1 = 10^log_IC_50[1];
  IC_50_2 = 10^log_IC_50[2];
  
  low_level = 10^log_low_level;
  mid_level = 10^log_mid_level;
  high_level = 10^log_high_level;
  
  for (i in 1:N) {
    mean_y[i] = low_level + (mid_level - low_level)*(x[i]^sensor_n)/(IC_50_1^sensor_n + x[i]^sensor_n) + (high_level - mid_level)*(x[i]^sensor_n)/(IC_50_2^sensor_n + x[i]^sensor_n);
  }
  
}

model {
  // Prior on sensor_n
  //sensor_n ~ gamma(4.0, 10.0/3.0); older version
  sensor_n ~ gamma(sensor_n_alpha, sensor_n_beta);
  
  // Prior on log_IC_50
  for (i in 1:2) {
    target += log1m(erf((log_x_min + 0.7 - log_IC_50[i])/0.5));
    target += log1m(erf((log_IC_50[i] - log_x_max + 0.8)/0.3));
  }
  
  // Prior on log_low_level
  //target += log1m(erf((log_g_min + 0.9 - log_low_level)/0.3));
  //target += log1m(erf((log_low_level - log_g_max + 0.9)/0.3));
  
  // Prior on log_mid_level
  //target += log1m(erf((log_g_min + 0.9 - log_mid_level)/0.3));
  //target += log1m(erf((log_mid_level - log_g_max + 0.9)/0.3));
  
  // Prior on log_high_level
  //target += log1m(erf((log_g_min + 0.9 - log_high_level)/0.3));
  //target += log1m(erf((log_high_level - log_g_max + 0.9)/0.3));
  
  y ~ normal(mean_y, sigma*y_err);

}

generated quantities {
  real log_sensor_n;
  real rms_resid;
  vector[30] y_out;
  real max_level;  // output at maximum  value of x that was measured
  
  for (i in 1:30) {
    y_out[i] = low_level + (mid_level - low_level)*(x_out[i]^sensor_n)/(IC_50_1^sensor_n + x_out[i]^sensor_n) + (high_level - mid_level)*(x_out[i]^sensor_n)/(IC_50_2^sensor_n + x_out[i]^sensor_n);
  }
  
  max_level = low_level + (mid_level - low_level)*(max_x_in^sensor_n)/(IC_50_1^sensor_n + max_x_in^sensor_n) + (high_level - mid_level)*(max_x_in^sensor_n)/(IC_50_2^sensor_n + max_x_in^sensor_n);
    
  log_sensor_n = log10(sensor_n);
  
  rms_resid = distance(y, mean_y)/sqrt(N);
  
}
