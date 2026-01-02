// 
//

data {
  int<lower=1> N;                 // number of data points
  vector[N] x;                    // x values
  vector[N] y;                    // y values
  vector[N] y_err;                // y uncertainty
  real baseline_mu;               // prior mean for baseline
  real<lower=2.2> baseline_df;               // prior degrees of freedom for baseline
  real<lower=0> baseline_scale;   // prior scale for baseline 
  
  real min_gamma;
  real max_gamma;
}

transformed data {
  real x_out_spacing;
  real min_x;
  real max_x;
  
  min_x = min(x);
  max_x = max(x);
  
  x_out_spacing = (max_x - min_x)/19;
  
  
}

parameters {
  real<lower=0> sigma;                           // scale factor for standard deviation of noise in y
  real baseline;                                 // baseline for exponential fit 
  real magnitude;                                // magnitude/prefactor for exponential term
  real<lower=min_gamma, upper=max_gamma> gamma;  // exponential growth/decay rate

}

transformed parameters {
  
  vector[N] mean_y;
  
  for (i in 1:N) {
    mean_y[i] = baseline + magnitude*exp(x[i]*gamma);
  }
  
}

model {
  
  sigma ~ normal(0, 3);

  baseline ~ student_t(baseline_df, baseline_mu, baseline_scale);
  
  y ~ normal(mean_y, sigma*y_err);

}

generated quantities {
  vector[20] y_out;
  vector[20] x_out;
  
  
  x_out[1] = min(x);
  for (i in 2:20) {
    x_out[i] = x_out[i-1] + x_out_spacing;
  }
  
  for (i in 1:20) {
    y_out[i] = baseline + magnitude*exp(x_out[i]*gamma);
  }
  
}
