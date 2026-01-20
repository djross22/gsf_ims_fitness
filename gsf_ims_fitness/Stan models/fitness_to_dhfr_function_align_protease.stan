// 
//

data {
  int<lower=1> N_van;      // number of different Van concentrations measured with TMP
  int<lower=1> N_sal;      // number of different Sal concentrations measured with TMP
  
  vector[N_van] log_x;     // log10 of the protease expression level 
  //vector[N_van] log_xerr;  // uncertainty of log10 of the protease expression level 
 
  array[N_van, N_sal] real y;         // normalized fitness difference with antibiotic (input data)
  array[N_van, N_sal] real yerr;      // estimated error of fitness difference
  
  vector[N_sal] log_dhfr_max;    // upper bound on log_dhfr, equal to the estimated DHFR level with zero protease
  real log_dhfr_min;             // lower bound on log_dhfr
  vector[N_sal] log_dhfr_sigma;  // uncertainty on log_dhfr bounds
  
  real log_prot_min;             // lower bound on log_ec50_prot
  real log_prot_max;             // upper bound on log_ec50_prot
  real log_prot_sigma;           // uncertainty on log_ec50_prot bounds
  
  real log_fraction_min;         // lower bound on log_fraction_inf
  real log_fraction_sigma;       // uncertainty on log_fraction_inf bounds
  
  real<lower=0> n_prot_alpha;   //prior shape for n_prot
  real<lower=0> n_prot_sigma;   //prior scale for n_prot
  
  // fitness calibration parameters:
  real low_fitness;       // fitness difference at zero function
  real mid_g;             // function level at 1/2 max fitness difference
  real fitness_n;         // cooperativity coefficient of fitness calibration curve
  real fitness_asym;      // asymmetry of fitness calibration curve
  
}

transformed data {
  real log_mid_g;
  real high_fitness;
  real logXb;
  vector[N_van] x;
  real fraction_0;  //value for the fraction_intact curve vs. protease expression at zero protease
  
  x = 10^log_x;
  
  fraction_0 = 1;
  
  high_fitness = 0;
  
  log_mid_g = log10(mid_g);
  
  logXb = log_mid_g + 1/fitness_n*log10(2^(1/fitness_asym) - 1);
}

parameters {
  ordered[N_sal] log_ec50_prot;      //log10 of the ec50 for the fraction_intact curve vs. protease expression
  //vector<upper=0>[N_sal] log_fraction_inf;   //log10 of the saturated value for the fraction_intact curve vs. protease expression (at infinite protease)
  real<upper=0> log_fraction_inf;   //log10 of the saturated value for the fraction_intact curve vs. protease expression (at infinite protease)
  ordered[N_sal] n_prot;                      //effective cooperativity for the fraction_intact curve vs. protease expression
  
  ordered[N_sal] log_initial_dhfr;     // log10 of DHFR level with zero protease
  
  real<lower=0> sigma;                // scale factor for standard deviation of noise in y
}

transformed parameters {
  vector[N_sal] ec50_prot;
  //vector[N_sal] fraction_inf;
  real fraction_inf;
  array[N_van] vector[N_sal] log_fraction_intact;   //log10 fraction of intact DHFR
  array[N_van, N_sal] real dhfr;
  array[N_van, N_sal] real mean_y;
  array[N_van, N_sal] real log_dhfr;                   // log10 of intact DHFR level
  
  ec50_prot = 10^log_ec50_prot;
  fraction_inf = 10^log_fraction_inf;
  
  for (van in 1:N_van) {
    for (sal in 1:N_sal) {
      log_fraction_intact[van, sal] = log10(fraction_0 + (fraction_inf - fraction_0)*(x[van]^n_prot[sal])/(ec50_prot[sal]^n_prot[sal] + x[van]^n_prot[sal]));
      
      log_dhfr[van, sal] = log_initial_dhfr[sal] + log_fraction_intact[van, sal];
    }
  }
  
  dhfr = 10^log_dhfr;
  
  //mean_y = low_fitness + (high_fitness - low_fitness)/(1 + 1/(dhfr^fitness_n)*10^(fitness_n*logXb))^fitness_asym;
  
  for (van in 1:N_van) {
    for (sal in 1:N_sal) {
      mean_y[van, sal] = low_fitness + (high_fitness - low_fitness)/(1 + 1/(dhfr[van, sal]^fitness_n)*10^(fitness_n*logXb))^fitness_asym;
    }
  }
  
}

model {
  // noise scale, use a prior to keep it from getting too much <> 1
  sigma ~ inv_gamma(3, 6);
  
  log_initial_dhfr ~ normal(log_dhfr_max, log_dhfr_sigma);
  
  // Prior on n_prot; <weibull> ~ n_prot_sigma*GammaFunct(1+1/n_prot_alpha)
  n_prot ~ weibull(n_prot_alpha, n_prot_sigma);
  
  // Soft lower and upper (zero) bound prior on log_fraction_inf
  target += log1m(erf((log_fraction_min - log_fraction_inf)/log_fraction_sigma));
  target += log1m(erf((log_fraction_inf)/log_fraction_sigma));
  
  for (sal in 1:N_sal) {
    // Soft lower and upper bound prior on log_ec50_prot
    target += log1m(erf((log_prot_min - log_ec50_prot[sal])/log_prot_sigma));
    target += log1m(erf((log_ec50_prot[sal] - log_prot_max)/log_prot_sigma));
    
    // Soft lower and upper (zero) bound prior on log_fraction_inf
    //target += log1m(erf((log_fraction_min - log_fraction_inf[sal])/log_fraction_sigma));
    //target += log1m(erf((log_fraction_inf[sal])/log_fraction_sigma));
    
    for (van in 1:N_van) {
      y[van, sal] ~ normal(mean_y[van, sal], sigma*yerr[van, sal]);
    }
  }
  
}

generated quantities {
  real rms_resid;
  
  rms_resid = 0;
  for (van in 1:N_van) {
    for (sal in 1:N_sal) {
      rms_resid = rms_resid + (y[van, sal] - mean_y[van, sal])^2;
    }
  }
  rms_resid = sqrt(rms_resid/(N_van*N_sal));
}
