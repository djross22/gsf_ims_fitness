// Fit dose-response curves to basic, unconstrained Hill model for allosteric TFs
//

data {
  
#include Free_energy_model.data.shared.stan
  
#include Hill_model.data.hill.stan

  // data specific to the unconstrained Hill model
  real log_g_min;        // lower bound on log_g0 and log_ginf
  real log_g_max;        // upper bound on log_g0 and log_ginf
  
}

transformed data {

#include Free_energy_model.transformed_data_decl.shared.stan

#include Free_energy_model.transformed_data_assign.shared.stan
  
  real log_ec50_min;
  real log_ec50_max;
  
  real sensor_n_alpha;   //prior shape for n_eff
  real sensor_n_sigma;   //prior scale for n_eff
  
  log_ec50_min = -1*log(10);
  log_ec50_max = 6.5*log(10);
  
  sensor_n_alpha = 3;
  sensor_n_sigma = 1.8;
  
}

parameters {
  // In this version of the model, the data for each variant are simply fit to a Hill model, 
  //      with rep_ratio and rep_offset to handle replicate-to-replicate variability in the measurements.
  
  // log_ec50 is constrained between min and max to handle flat-line variants.
  //      The only other constraints are that the parameters must be >0, which is enforced using the log-transformed paramters.
  
  vector<lower=log_g_min, upper=log_g_max>[num_var] log_g0_var;                                                // log-transformed g0
  vector<lower=log_g_min, upper=log_g_max>[num_var] log_ginf_var;                                              // log-transformed ginf
  vector<lower=log_ec50_min, upper=log_ec50_max>[num_var] log_ec50_var;      // log(EC_50)
  
  vector<lower=0, upper=4>[num_var] n_eff_var;                                             // effective cooperativity, i.e., Hill cooeficient.
  
  real<lower=0> sigma;                        // scale factor for standard deviation of noise in log_y
  real<lower=0> offset_sigma;                 // scale factor for standard deviation of replicate variability in g_min
  
  vector[num_reps] log_rep_ratio;              // log10 of multiplicative correction factor for different replicates

  real<lower=0> rep_ratio_sigma;               // hyper-paramters for log_rep_ratio
  
  vector<lower=-3*rep_offset_scale, upper=3*rep_offset_scale>[num_reps] rep_offset;              // additional g_min shift for different replicates
  
}

transformed parameters {
  vector[num_var] g0;
  vector[num_var] ginf;
  vector[num_var] ec50;
  
  vector[N] log_mean_y;
  
  // measured values with g_min subtracted
  vector[N] y_shifted;
  
  g0 = exp(log_g0_var);
  ginf = exp(log_ginf_var);
  ec50 = exp(log_ec50_var);
  
  for (i in 1:N) {
	real dose_response;
	
    dose_response = g0[variant[i]] + (ginf[variant[i]] - g0[variant[i]])*(x[i]^n_eff_var[variant[i]])/(ec50[variant[i]]^n_eff_var[variant[i]] + x[i]^n_eff_var[variant[i]]);
    
    log_mean_y[i] = log(dose_response) + log_rep_ratio[rep[i]];
	
    // measured values with g_min and rep_offset subtracted
    y_shifted[i] = y[i] - g_min - rep_offset[rep[i]];
  }

  
}

model {
  // Prior on n_eff_var; <weibull> ~ sensor_n_sigma*GammaFunct(1+1/sensor_n_alpha)
  n_eff_var ~ weibull(sensor_n_alpha, sensor_n_sigma);
  
  // prior on scale parameter for log-normal measurement error
  sigma ~ normal(0, 1);
  
  // model of the data (dose-response curve with noise)
  y_shifted ~ lognormal(log_mean_y, sigma);
  
  // prior on scale hyper-parameter for log_rep_ratio
  rep_ratio_sigma ~ normal(0, rep_ratio_scale);
  
  // priors on log_rep_ratio
  log_rep_ratio ~ normal(0, rep_ratio_sigma);
  
  // priors on rep_offset
  rep_offset ~ normal(0, offset_sigma);

}

generated quantities {
  real y_out[num_var, 19];
  
  // WT variant is first
  real log_g0_wt;
  real log_ginf_wt;
  real log_ec50_wt;
  real n_eff_wt;
  
  log_g0_wt = log_g0_var[1];
  log_ginf_wt = log_ginf_var[1];
  log_ec50_wt = log_ec50_var[1];
  n_eff_wt = n_eff_var[1];
  
  
  
  for (variant_num in 1:num_var) {
    for (i in 1:19) {
      y_out[variant_num, i] = g0[variant_num] + (ginf[variant_num] - g0[variant_num])*(x_out[i]^n_eff_var[variant_num])/(ec50[variant_num]^n_eff_var[variant_num] + x_out[i]^n_eff_var[variant_num]);
	      
    }
	
  }
  
}
