// 
//

data {
  // With this version of the model, the number of Van concentrations can be different for the two different Sal concentrations
  // Stan doesn't handle jagged arrays, so they have to be separated
  int<lower=1> N_van_1;      // number of different Van concentrations measured with TMP at the lower Sal concentration
  int<lower=1> N_van_2;      // number of different Van concentrations measured with TMP at the higher Sal concentration
  
  vector[N_van_1] log_x_1;     // log10 of the protease expression level, lower Sal concentration
  vector[N_van_2] log_x_2;     // log10 of the protease expression level, higher Sal concentration
  //vector[N_van] log_xerr;  // uncertainty of log10 of the protease expression level 
 
  vector[N_van_1] y_1;         // normalized fitness difference with antibiotic (input data)
  vector[N_van_1] yerr_1;      // estimated error of fitness difference
  vector[N_van_2] y_2;         // normalized fitness difference with antibiotic (input data)
  vector[N_van_2] yerr_2;      // estimated error of fitness difference
  
  vector[2] log_initial_dhfr_mu;    // prior mean for upper bound on log_dhfr, equal to the estimated DHFR level with zero protease
  vector[2] log_initial_dhfr_sigma; // prior std on log_dhfr bounds
  
  real log_prot_min;             // lower bound on log_ec50_prot
  real log_prot_max;             // upper bound on log_ec50_prot
  real log_prot_sigma;           // uncertainty on log_ec50_prot bounds
  
  real log_fraction_min;         // lower bound on log_fraction_inf
  real log_fraction_sigma;       // uncertainty on log_fraction_inf bounds
  
  real<lower=0> n_prot_mu;      //prior shape for n_prot
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
  vector[N_van_1] x_1;
  vector[N_van_2] x_2;
  real fraction_0;  //value for the fraction_intact curve vs. protease expression at zero protease
  
  x_1 = 10^log_x_1;
  x_2 = 10^log_x_2;
  
  fraction_0 = 1;
  
  high_fitness = 0;
  
  log_mid_g = log10(mid_g);
  
  logXb = log_mid_g + 1/fitness_n*log10(2^(1/fitness_asym) - 1);
}

parameters {
  ordered[2] log_ec50_prot;      //log10 of the ec50 for the fraction_intact curve vs. protease expression
  real<upper=0> log_fraction_inf;   //log10 of the saturated value for the fraction_intact curve vs. protease expression (at infinite protease)
  vector<lower=0>[2] n_prot;                      //effective cooperativity for the fraction_intact curve vs. protease expression
  
  ordered[2] log_initial_dhfr;     // log10 of DHFR level with zero protease
  
  real<lower=0> sigma;                // scale factor for standard deviation of noise in y
}

transformed parameters {
  vector[2] ec50_prot;
  real fraction_inf;
  
  vector[N_van_1] log_fraction_intact_1;   //log10 fraction of intact DHFR
  vector[N_van_1] dhfr_1;
  vector[N_van_1] mean_y_1;
  vector[N_van_1] log_dhfr_1;  
  
  vector[N_van_2] log_fraction_intact_2;   //log10 fraction of intact DHFR
  vector[N_van_2] dhfr_2;
  vector[N_van_2] mean_y_2;
  vector[N_van_2] log_dhfr_2;                   // log10 of intact DHFR level
  
  ec50_prot = 10^log_ec50_prot;
  fraction_inf = 10^log_fraction_inf;
  
  // Lower Sal concentration:
  for (van in 1:N_van_1) {
    log_fraction_intact_1[van] = log10(fraction_0 + (fraction_inf - fraction_0)*(x_1[van]^n_prot[1])/(ec50_prot[1]^n_prot[1] + x_1[van]^n_prot[1]));
    
    log_dhfr_1[van] = log_initial_dhfr[1] + log_fraction_intact_1[van];
	
    dhfr_1[van] = 10^log_dhfr_1[van];
	
    mean_y_1[van] = low_fitness + (high_fitness - low_fitness)/(1 + 1/(dhfr_1[van]^fitness_n)*10^(fitness_n*logXb))^fitness_asym;
  }
  
  // Higher Sal concentration:
  for (van in 1:N_van_2) {
    log_fraction_intact_2[van] = log10(fraction_0 + (fraction_inf - fraction_0)*(x_2[van]^n_prot[2])/(ec50_prot[2]^n_prot[2] + x_2[van]^n_prot[2]));
    
    log_dhfr_2[van] = log_initial_dhfr[2] + log_fraction_intact_2[van];
    
    dhfr_2[van] = 10^log_dhfr_2[van];
    
    mean_y_2[van] = low_fitness + (high_fitness - low_fitness)/(1 + 1/(dhfr_2[van]^fitness_n)*10^(fitness_n*logXb))^fitness_asym;
  }
  
}

model {
  // noise scale, use a prior to keep it from getting too much <> 1
  sigma ~ inv_gamma(3, 6);
  
  log_initial_dhfr ~ normal(log_initial_dhfr_mu, log_initial_dhfr_sigma);
  
  // Prior on n_prot, 
  n_prot ~ normal(n_prot_mu, n_prot_sigma);
  
  // Soft lower and upper (zero) bound prior on log_fraction_inf
  target += log1m(erf((log_fraction_min - log_fraction_inf)/log_fraction_sigma));
  target += log1m(erf((log_fraction_inf)/log_fraction_sigma));
  
  for (sal in 1:2) {
    // Soft lower and upper bound prior on log_ec50_prot
    target += log1m(erf((log_prot_min - log_ec50_prot[sal])/log_prot_sigma));
    target += log1m(erf((log_ec50_prot[sal] - log_prot_max)/log_prot_sigma));
  }
  
  for (van in 1:N_van_1) {
    y_1[van] ~ normal(mean_y_1[van], sigma*yerr_1[van]);
  }
  for (van in 1:N_van_2) {
    y_2[van] ~ normal(mean_y_2[van], sigma*yerr_2[van]);
  }
  
}

generated quantities {
  real rms_resid;
  
  rms_resid = 0;
  for (van in 1:N_van_1) {
    rms_resid = rms_resid + (y_1[van] - mean_y_1[van])^2;
  }
  for (van in 1:N_van_2) {
    rms_resid = rms_resid + (y_2[van] - mean_y_2[van])^2;
  }
  rms_resid = sqrt(rms_resid/(N_van_1 + N_van_2));
}
