// 
//

data {
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // gene expression (from cytometry) at each concentration
  vector[N] y_err;       // estimated error of gene expression at each concentration
  real log_g_min;        // lower bound on log_low_level and log_high_level
  real log_g_max;        // upper bound on log_low_level and log_high_level
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
  
  log_x_max = 6;
  log_x_min = -1;
  
}

parameters {
  real<lower=log_g_min, upper=log_g_max> log_low_level;               // log10 of gene expression level at zero induction
  real<lower=log_g_min, upper=log_g_max>  log_high_level;             // log10 of gene expression level at infinite induction
  
  real<lower=log_x_min, upper=log_x_max> log_IC_50;                   // input level (x) that gives output 1/2 way between low_level and high_level
  real<lower=0> sensor_n;                                             // cooperativity exponent of sensor gene expression vs. x curve
  real<lower=0> sigma;                                                // scale factor for standard deviation of noise in y
  real<lower=0> asym;                                                // asymmetry parameter

}

transformed parameters {
  real low_level;        
  real high_level; 
  real IC_50;
  
  vector[N] mean_y;
  real logXb;
  
  IC_50 = 10^log_IC_50;
  low_level = 10^log_low_level;
  high_level = 10^log_high_level;
  
  logXb = log_IC_50 + 1/sensor_n*log10(2^(1/asym) - 1);
  
  for (i in 1:N) {
    if (x[i]>0) {
      mean_y[i] = low_level + (high_level - low_level)/(1 + 1/(x[i]^sensor_n)*10^(sensor_n*logXb))^asym;
    }
    else {
      mean_y[i] = low_level;
    }
    //mean_y[i] = low_level + (high_level - low_level)*(x[i]^sensor_n)/(IC_50^sensor_n + x[i]^sensor_n);
  }
  
}

model {
  // Prior on sensor_n
  sensor_n ~ gamma(4.0, 4.0);
  // Prior on asym
  asym ~ gamma(1.0, 1.0);
  
  // Prior on log_IC_50
  //target += log1m(erf((log_x_min + 0.7 - log_IC_50)/0.5));
  //target += log1m(erf((log_IC_50 - log_x_max + 0.8)/0.3));
  
  // Prior on log_low_level
  //target += log1m(erf((log_g_min + 0.9 - log_low_level)/0.3));
  //target += log1m(erf((log_low_level - log_g_max + 0.9)/0.3));
  
  // Prior on log_high_level
  //target += log1m(erf((log_g_min + 0.9 - log_high_level)/0.3));
  //target += log1m(erf((log_high_level - log_g_max + 0.9)/0.3));
  
  y ~ normal(mean_y, sigma*y_err);

}

generated quantities {
  real log_sensor_n;
  real rms_resid;
  real log_high_low_ratio;
  
  log_high_low_ratio = log_high_level - log_low_level;
  
  log_sensor_n = log10(sensor_n);
  
  rms_resid = distance(y, mean_y)/sqrt(N);
  
}
