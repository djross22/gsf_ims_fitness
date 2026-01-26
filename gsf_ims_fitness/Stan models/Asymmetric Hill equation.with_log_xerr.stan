// 
//

data {
  int<lower=1> N;            // number of data points
  vector[N] log_x;           // log10 of input parameter
  vector[N] log_xerr;        // estimated error of log-input
  vector[N] y;               // fitness
  vector[N] yerr;            // estimated error of fitness
  
  real g0_mu;               //prior mean
  real<lower=0> g0_sigma;   //prior std
  
  real log_ec50_mu;               //prior mean
  real<lower=0> log_ec50_sigma;   //prior std
  
  real<lower=0> hill_n_mu;      //prior mean
  real<lower=0> hill_n_sigma;   //prior std
  
  real<lower=0> asym_mu;      //prior mean
  real<lower=0> asym_sigma;   //prior std
  
  int<lower=1> N_out;        // number of data points
  vector[N_out] x_out;           // for posterior y_out
}

transformed data {
  real ginf;
  
  ginf = 0; // Fitness impact of antibiotic has zero upper limit
  
}

parameters {
  real g0;                      // minimum fitness level (at x = 0)
  
  real log_ec50;                   // input level (x) that gives output 1/2 way between g0 and ginf
  real<lower=0> hill_n;            // cooperativity exponent of sensor gene expression vs. x curve
  real<lower=0> asym;              // asymmetry parameter
  
  vector[N] mean_log_x;           // log10 of "true" input parameter
  
  real<lower=0> sigma;             // scale factor for standard deviation of noise in y
}

transformed parameters {
  real ec50;
  
  vector[N] mean_y;
  real logXb;
  
  vector[N] x;
  
  x = 10^mean_log_x;
  
  ec50 = 10^log_ec50;
  
  logXb = log_ec50 + 1/hill_n*log10(2^(1/asym) - 1);
  
  for (i in 1:N) {
    if (x[i]>0) {
      mean_y[i] = g0 + (ginf - g0)/(1 + 1/(x[i]^hill_n)*10^(hill_n*logXb))^asym;
    }
    else {
      mean_y[i] = g0;
    }
    //mean_y[i] = g0 + (ginf - g0)*(x[i]^hill_n)/(ec50^hill_n + x[i]^hill_n);
  }
  
}

model {
  // Prior on g0
  g0 ~ normal(g0_mu, g0_sigma);
  
  // Prior on log_ec50
  log_ec50 ~ normal(log_ec50_mu, log_ec50_sigma);
  
  // Prior on hill_n
  hill_n ~ normal(hill_n_mu, hill_n_sigma);

  // Prior on asym
  asym ~ normal(asym_mu, asym_sigma);
  
  // noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);
  
  y ~ normal(mean_y, sigma*yerr);
  
  mean_log_x ~ normal(log_x, log_xerr);

}

generated quantities {
  real rms_resid;
  vector[N_out] y_out;           // posterior fitness
    
  rms_resid = distance(y, mean_y)/sqrt(N);
  
  for (i in 1:N_out) {
    y_out[i] = g0 + (ginf - g0)/(1 + 1/(x_out[i]^hill_n)*10^(hill_n*logXb))^asym;
  }
}
