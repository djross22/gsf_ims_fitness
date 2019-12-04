// 
//

data {
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // fitness difference at each concentration
  
  real low_fitness_mu;      // fitness diference at zero gene expression
  real mid_g_mu;            // gene expression evel at 1/2 max fitness difference
  real fitness_n_mu;        // cooperativity coeficient of fitness difference curve
  
}

transformed data {
}

parameters {
  real<lower=1.5, upper=4> log_low_level;               // log10 of gene expression level at zero induction
  real<lower=1.5, upper=4>  log_high_level;             // log10 of gene expression level at infinite induction
  real<lower=-1, upper=4.6> log_IC_50;        // input level (x) that gives output 1/2 way between low_level and high_level
  real<lower=0> sensor_n;                    // cooperativity exponent of sensor gene expression vs. x curve
  real<lower=0> sigma;                     // standard deviation of noise in y
  
  real low_fitness;      // fitness diference at zero gene expression
  real mid_g;            // gene expression evel at 1/2 max fitness difference
  real fitness_n;        // cooperativity coeficient of fitness difference curve
}

transformed parameters {
  real low_level;        
  real high_level; 
  real IC_50;
  
  IC_50 = 10^log_IC_50;
  low_level = 10^log_low_level;
  high_level = 10^log_high_level;
  
}

model {
  vector[N] mean_y;
  vector[N] g;   // gene expression level at each concentration
  
  low_fitness ~ normal(low_fitness_mu, 0.05);
  mid_g ~ normal(mid_g_mu, 10);
  fitness_n ~ normal(fitness_n_mu, 0.03);
  
  sensor_n ~ gamma(4, 10/3);
  log_IC_50 ~ normal(1.81, 1);
  
  log_low_level ~ normal(2, 1);
  log_high_level ~ normal(3, 1);
  
  for (i in 1:N) {
    g[i] = low_level + (high_level - low_level)*(x[i]^sensor_n)/(IC_50^sensor_n + x[i]^sensor_n);
    mean_y[i] = low_fitness - low_fitness*(g[i]^fitness_n)/(mid_g^fitness_n + g[i]^fitness_n);
  }
  
  y ~ normal(mean_y, sigma);

}

generated quantities {
  real log_sensor_n;
  
  log_sensor_n = log10(sensor_n);
}
