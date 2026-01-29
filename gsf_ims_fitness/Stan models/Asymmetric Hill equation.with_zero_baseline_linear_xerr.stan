// 
//

data {
  int<lower=1> N;            // number of data points
  vector[N] x;               // log10 of input parameter
  vector[N] xerr;            // estimated error of log-input
  vector[N] y;               // fitness
  vector[N] yerr;            // estimated error of fitness
  
  real ginf_mu;               //prior mean
  real<lower=0> ginf_sigma;   //prior std
  
  real log_ec50_mu;               //prior mean
  real<lower=0> log_ec50_sigma;   //prior std
  
  //real<lower=0> hill_n_mu;      //prior mean
  //real<lower=0> hill_n_sigma;   //prior std
  
  real<lower=0> asym_mu;        //prior mean
  real<lower=0> asym_sigma;     //prior std
  
  int<lower=1> N_out;           // number of data points
  vector[N_out] x_out;          // for posterior y_out
  vector[N_out] xerr_out;       // for posterior y_out
}

transformed data {
  real g0;
  real<lower=0> hill_n;            // cooperativity exponent of sensor gene expression vs. x curve
  
  g0 = 0;       // This Hill model constrained to pass through 0.
  hill_n = 1;   //    and constrains the cooperativity to 1
  
}

parameters {
  real ginf;                      // minimum fitness level (at x = 0)
  
  real log_ec50;                   // input level (x) that gives output 1/2 way between g0 and ginf
  real<lower=0> asym;              // asymmetry parameter
  
  vector[N] mean_x;               // "true" input parameter
  
  real<lower=0> sigma;             // scale factor for standard deviation of noise in y
}

transformed parameters {
  real ec50;
  
  vector[N] mean_y;
  real logXb;
  
  ec50 = 10^log_ec50;
  
  logXb = log_ec50 + 1/hill_n*log10(2^(1/asym) - 1);
  
  for (i in 1:N) {
    if (mean_x[i]>0) {
      mean_y[i] = g0 + (ginf - g0)/(1 + 1/(mean_x[i]^hill_n)*10^(hill_n*logXb))^asym;
    }
    else {
      mean_y[i] = g0;
    }
    //mean_y[i] = g0 + (ginf - g0)*(mean_x[i]^hill_n)/(ec50^hill_n + mean_x[i]^hill_n);
  }
  
}

model {
  // Prior on ginf
  ginf ~ normal(ginf_mu, ginf_sigma);
  
  // Prior on log_ec50
  log_ec50 ~ normal(log_ec50_mu, log_ec50_sigma);
  
  // Prior on hill_n
  //hill_n ~ normal(hill_n_mu, hill_n_sigma);

  // Prior on asym
  asym ~ normal(asym_mu, asym_sigma);
  
  // noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);
  
  y ~ normal(mean_y, sigma*yerr);
  
  mean_x ~ normal(x, xerr);

}

generated quantities {
  real rms_resid;
  vector[N_out] x_out_samp;      // sample from x_out distribution
  vector[N_out] y_out;           // posterior fitness
    
  rms_resid = distance(y, mean_y)/sqrt(N);
  
  for (i in 1:N_out) {
    x_out_samp[i] = normal_rng(x_out[i], xerr_out[i]); // Generates a random draw
	if (x_out_samp[i]<0) {
	  x_out_samp[i] = 0;
	}
    y_out[i] = g0 + (ginf - g0)/(1 + 1/(x_out_samp[i]^hill_n)*10^(hill_n*logXb))^asym;
  }
}
