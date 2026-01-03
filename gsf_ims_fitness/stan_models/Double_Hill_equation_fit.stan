// 
//

data {
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // fitness difference at each concentration
  vector[N] y_err;       // estimated error of fitness difference at each concentration
  real log_g_min;        // lower bound on log_g0 and log_ginf_1
  real log_g_max;        // upper bound on log_g0 and log_ginf_1
  real log_g_prior_scale;
  
  real low_fitness_mu;      // fitness difference at zero gene expression
  real mid_g_mu;            // gene expression level at 1/2 max fitness difference
  real fitness_n_mu;        // cooperativity coefficient of fitness difference curve
  
  array[1] real log_x_max;  // maximum possible value for log_ec50, previously set to log10(max(x)) + 1.289;
  
}

transformed data {
  real x_min;
  real x_max;
  real log_x_1_min;
  real log_x_1_max;
  
  x_max = max(x);
  x_min = x_max;
  for (i in 1:N) {
    if (x[i]>0) {
      if (x[i]<x_min) {
	    x_min = x[i];
	  }
	}
  }
  
  log_x_1_min = log10(x_min) - 1.3;
  log_x_1_max = log_x_max[1];
  
}

parameters {
  real<lower=log_g_min, upper=log_g_max> log_g0;       // log10 of gene expression level at zero induction
  real<lower=log_g_min, upper=log_g_max> log_ginf_1;     // log10 of gene expression level at infinite induction
  
  real<lower=log_x_1_min, upper=log_x_1_max> log_ec50_1;           // input level (x) that gives output 1/2 way between g0 and ginf
  real<lower=0> sensor_n_1;                                     // cooperativity exponent of sensor gene expression vs. x curve
  real<lower=0> sigma;                                        // scale factor for standard deviation of noise in y
  
  real low_fitness_high_tet;      // fitness difference at zero gene expression
  real mid_g_high_tet;            // gene expression level at 1/2 max fitness difference
  real fitness_n_high_tet;        // cooperativity coefficient of fitness difference curve
}

transformed parameters {
  real g0;        
  real ginf; 
  real ec50;
  
  vector[N] mean_y;
  vector[N] g;   // gene expression level at each concentration
  
  ec50 = 10^log_ec50_1;
  g0 = 10^log_g0;
  ginf = 10^log_ginf_1;
  
  for (i in 1:N) {
    g[i] = g0 + (ginf - g0)*(x[i]^sensor_n_1)/(ec50^sensor_n_1 + x[i]^sensor_n_1);
    mean_y[i] = low_fitness_high_tet - low_fitness_high_tet*(g[i]^fitness_n_high_tet)/(mid_g_high_tet^fitness_n_high_tet + g[i]^fitness_n_high_tet);
  }
  
}

model {
  real neg_low_fitness;
  
  neg_low_fitness = -1*low_fitness_high_tet;
  
  // fitness calibration params
  //low_fitness_high_tet ~ student_t(8, low_fitness_mu, 0.1);
  neg_low_fitness ~ exp_mod_normal(-1*low_fitness_mu-0.04, 0.03, 14); 
  
  //mid_g_high_tet ~ normal(mid_g_mu, 27); // use with PTY1
  //fitness_n_high_tet ~ normal(fitness_n_mu, 0.22); // use with pTY1
  mid_g_high_tet ~ normal(mid_g_mu, 499); // use with pVER
  fitness_n_high_tet ~ normal(fitness_n_mu, 0.29); // use with pVER
  
  // Prior on sensor_n; <gamma> = alpha/beta = 1.5; std = sqrt(alpha)/beta = 0.5
  sensor_n_1 ~ gamma(9.0, 6.0);
  
  // Prior on log_ec50_1
  target += log1m(erf((log_x_1_min + 0.7 - log_ec50_1)/0.5));
  target += log1m(erf((log_ec50_1 - log_x_1_max + 0.8)/0.3));
  
  // noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);
  
  y ~ normal(mean_y, sigma*y_err);

}

generated quantities {
  real log_sensor_n_1;
  real rms_resid;
  real log_ginf_g0_ratio_1;
  
  log_ginf_g0_ratio_1 = log_ginf_1 - log_g0;
  
  log_sensor_n_1 = log10(sensor_n_1);
  
  rms_resid = distance(y, mean_y)/sqrt(N);
  
}
