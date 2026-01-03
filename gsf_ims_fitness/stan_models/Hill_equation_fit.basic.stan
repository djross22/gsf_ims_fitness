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
  real<upper=0> low_level;         // output at zero input
  real<upper=0> high_level;         // output at saturated input
  real IC_50;                      // input level (x) that gives output 1/2 way between low_level and zero
  real<lower=0> hill_n;            // cooperativity exponent 
  real<lower=0> sigma;             // scale factor for standard deviation of noise in y

}

transformed parameters {
  vector[N] mean_y;
  
  for (i in 1:N) {
    mean_y[i] = low_level + (high_level - low_level)*(x[i]^hill_n)/(IC_50^hill_n + x[i]^hill_n);
  }
  
}

model {
  
  y ~ normal(mean_y, sigma*y_err);

}

generated quantities {
  real rms_resid;
  
  rms_resid = distance(y, mean_y)/sqrt(N);
  
}
