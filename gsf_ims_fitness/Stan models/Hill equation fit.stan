// 
//

data {
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // gene expression (from cytometry) at each concentration
  vector[N] y_err;       // estimated error of gene expression at each concentration
  
}

transformed data {
  real x_min;
  real x_max;
  real log_x_min;
  real log_x_max;
  
  x_max = max(x);
  x_min = x_max;
  for (i in 1:N) {
    if (x[i]>0) {
      if (x[i]<x_min) {
	    x_min = x[i];
	  }
	}
  }
  
  log_x_max = log10(x_max) + 1.289;
  log_x_min = log10(x_min) - 1.3;
  
}

parameters {
  real<lower=1., upper=4> log_low_level;               // log10 of gene expression level at zero induction
  real<lower=1., upper=4>  log_high_level;             // log10 of gene expression level at infinite induction
  //real<lower=-1, upper=4.6> log_IC_50;        // input level (x) that gives output 1/2 way between low_level and high_level
  real<lower=log_x_min, upper=log_x_max> log_IC_50;        // input level (x) that gives output 1/2 way between low_level and high_level
  real<lower=0> sensor_n;                    // cooperativity exponent of sensor gene expression vs. x curve
  real<lower=0> sigma;                     // scale factor for standard deviation of noise in y

}

transformed parameters {
  real low_level;        
  real high_level; 
  real IC_50;
  
  vector[N] mean_y;
  
  IC_50 = 10^log_IC_50;
  low_level = 10^log_low_level;
  high_level = 10^log_high_level;
  
  for (i in 1:N) {
    mean_y[i] = low_level + (high_level - low_level)*(x[i]^sensor_n)/(IC_50^sensor_n + x[i]^sensor_n);
  }
  
}

model {
  sensor_n ~ gamma(4.0, 10.0/3.0);
  //log_IC_50 ~ normal(1.81, 1);
  target += log1m(erf((log_x_min + 0.7 - log_IC_50)/0.5));
  target += log1m(erf((log_IC_50 - log_x_max + 0.8)/0.3));
  
  //log_low_level ~ normal(2, 1);
  target += log1m(erf((1.9 - log_low_level)/0.3));
  target += log1m(erf((log_low_level - 3.6)/0.3));
  //log_high_level ~ normal(3.2, 0.5);
  target += log1m(erf((1.9 - log_high_level)/0.3));
  target += log1m(erf((log_high_level - 3.6)/0.3));
  
  y ~ normal(mean_y, sigma*y_err);

}

generated quantities {
  real log_sensor_n;
  real rms_resid;
  
  log_sensor_n = log10(sensor_n);
  
  rms_resid = distance(y, mean_y)/N;
  
}
