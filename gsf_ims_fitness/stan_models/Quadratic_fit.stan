// 
//

data {
  int<lower=1> N;        // number of data points
  vector[N] x;           // x values
  vector[N] y;           // y values
  vector[N] y_err;       // y uncertainty
}

transformed data {
  real sigma;      // scale factor for standard deviation of noise in y
  real max_x_in;    // max inducer concentration
  real min_x_in;    // min non-zero inducer concentration
  real log_x_out_spacing;
  
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
  
  sigma = 1;
  
}

parameters {
  real c_0;           // zeroth order term
  real c_1;           // first order term
  real c_2;           // second order term

}

transformed parameters {
  
  vector[N] mean_y;
  
  for (i in 1:N) {
    mean_y[i] = c_0 + c_1*x[i] + c_2*x[i]^2;
  }
  
}

model {
  
  y ~ normal(mean_y, sigma*y_err);

}

generated quantities {
  vector[30] x_out;
  vector[30] y_out;
  
  
  x_out[1] = 0;
  x_out[2] = min_x_in/2;
  for (i in 3:30) {
    x_out[i] = x_out[i-1]*exp(log_x_out_spacing);
  }
  
  for (i in 1:30) {
    y_out[i] = c_0 + c_1*x_out[i] + c_2*x_out[i]^2;
  }
  
}
