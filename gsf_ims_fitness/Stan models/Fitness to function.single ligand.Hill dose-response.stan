// 
//

data {
  int<lower=1> N_antibiotic;  // number of non-zero antibiotic concentrations
  int<lower=1> N;             // total number of data points across all non-zero antibiotic concentrations
  int s[N_antibiotic];        // array of the number of data points for each non-zero antibiotic concentration
  
  vector[N] x;           // ligand concentrations across all non-zero antibiotic concentrations
  vector[N] y;           // normalized fitness difference datapoints across all non-zero antibiotic concentrations
  vector[N] y_err;       // estimated error of y
  
  real log_g_min;                    // lower bound on log_g
  real log_g_max;                    // upper bound on log_g
  
  // prior means (mu) and standard deviations for fitness calibration parameters:
  vector[N_antibiotic] low_fitness_mu;       // fitness difference at zero function
  vector[N_antibiotic] log_mid_g_mu;             // function level at 1/2 max fitness difference
  vector[N_antibiotic] fitness_n_mu;         // cooperativity coefficient of fitness calibration curve
  
  vector[N_antibiotic] low_fitness_std;      // fitness difference at zero function
  vector[N_antibiotic] log_mid_g_std;            // function level at 1/2 max fitness difference
  vector[N_antibiotic] fitness_n_std;        // cooperativity coefficient of fitness calibration curve
  
  vector[1] log_x_max;    // maximum possible value for log_ec50, vector is used instead of real for consistency with multi-ligand models
  
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
  real<lower=log_g_min, upper=log_g_max> log_g0;          // log10 of function at zero ligand
  real<lower=log_g_min, upper=log_g_max> log_ginf_1;      // log10 of gene expression level at infinite induction
  real<lower=log_x_1_min, upper=log_x_1_max> log_ec50_1;  // input level (x) that gives output 1/2 way between g0 and ginf
  real<lower=0> sensor_n_1;                               // cooperativity exponent of sensor gene expression vs. x curve
  
  real<lower=0> sigma;            // scale factor for standard deviation of noise in y
  
  vector[N_antibiotic] low_fitness;       // fitness difference at zero function
  vector[N_antibiotic] log_mid_g;         // log10 of gene expression level at 1/2 max fitness difference
  vector[N_antibiotic] fitness_n;         // cooperativity coefficient of fitness calibration curve
}

transformed parameters {
  vector[N_antibiotic] mid_g;
  real g0;        
  real ginf; 
  real ec50;
  
  vector[N] g;
  vector[N] mean_y;
  
  ec50 = 10^log_ec50_1;
  g0 = 10^log_g0;
  ginf = 10^log_ginf_1;
  mid_g = 10^log_mid_g;
  
  {
    int pos;
    pos = 1;
    
    // g is the dose-response curve:
    for (n in 1:N) {
      g[n] = g0 + (ginf - g0)*(x[n]^sensor_n_1)/(ec50^sensor_n_1 + x[n]^sensor_n_1);
    }
    //g = g0 + (ginf - g0)*(x^sensor_n_1)/(ec50^sensor_n_1 + x^sensor_n_1);
    
    // y is the fitness curve derived from the dose-response:
    for (k in 1:N_antibiotic) {
      for (p in pos:pos+s[k]-1) {
        mean_y[p] = low_fitness[k] - low_fitness[k]*(g[p]^fitness_n[k])/(mid_g[k]^fitness_n[k] + g[p]^fitness_n[k]);
      }
      //mean_y[pos:pos+s[k]-1] = low_fitness[k] - low_fitness[k]*(g[pos:pos+s[k]-1]^fitness_n[k])/(mid_g[k]^fitness_n[k] + g[pos:pos+s[k]-1]^fitness_n[k]);
      
      pos = pos + s[k];
    }
  }
  
}

model {
  // Priors on fitness calibration curve parameters:
  low_fitness ~ normal(low_fitness_mu, low_fitness_std);
  log_mid_g ~ normal(log_mid_g_mu, log_mid_g_std);
  fitness_n ~ normal(fitness_n_mu, fitness_n_std);
  
  // Prior on sensor_n; mean = alpha/beta = 1.5; std = sqrt(alpha)/beta = 0.5
  sensor_n_1 ~ gamma(9.0, 6.0);
  
  // Prior on log_ec50_1
  target += log1m(erf((log_x_1_min + 0.7 - log_ec50_1)/0.5));
  target += log1m(erf((log_ec50_1 - log_x_1_max + 0.8)/0.3));
  
  // prior noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);
  
  y ~ normal(mean_y, sigma*y_err);
  
}

generated quantities {
  real log_sensor_n_1;
  real rms_resid;
  real log_ginf_g0_ratio_1;
  //vector[N] log_g;
  real log_gxmax_1;
  real log_gxmax_g0_ratio_1;
  
  //log_g = log10(g);
  log_sensor_n_1 = log10(sensor_n_1);
  
  log_ginf_g0_ratio_1 = log_ginf_1 - log_g0;
  
  log_gxmax_1 = log10(g0 + (ginf - g0)*(x_max^sensor_n_1)/(ec50^sensor_n_1 + x_max^sensor_n_1));
  log_gxmax_g0_ratio_1 = log_gxmax_1 - log_g0;
  
  rms_resid = distance(y, mean_y)/sqrt(N);
}
