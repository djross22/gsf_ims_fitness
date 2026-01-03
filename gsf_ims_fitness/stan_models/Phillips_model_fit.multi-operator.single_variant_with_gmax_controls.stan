// Fit dose-response curves to Phillips lab model for allosteric TFs
//    Single-variant version of the model, used for WT-only dataset
//

data {
  int<lower=1> N_contr;          // number of data points for control strains LacI deletions
  vector[N_contr] y_contr;       // gene expression (from cytometry) for control strains
  vector[N_contr] y_contr_err;   // estimated error of gene expression for control strains
  
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // gene expression (from cytometry) at each concentration
  vector[N] y_err;       // estimated error of gene expression at each concentration
  
  real y_max;             // geometric mean for prior on maximum gene expression value
  real g_max_prior_width; // geometric std for prior on maximum gene expression value
  
  real<lower=2> copy_num_prior_mean;  // geometric mean for prior on plasmid/operator copy number
  real<lower=0> copy_num_prior_width; // geometric std for prior on plasmid/operator copy number
  real<lower=1> R_prior_mean;         // geometric mean for prior on repressor dimer copy number
  real<lower=0> log_R_prior_width;    // geometric std for prior on repressor dimer copy number
  
  int variant[N];        // numerical index to indicate variants
  
  // priors on wild-type free energy parameters
  real log_k_a_wt_prior_mean;
  real log_k_a_wt_prior_std;
  real log_k_i_wt_prior_mean;
  real log_k_i_wt_prior_std;
  real delta_eps_AI_wt_prior_mean;
  real delta_eps_AI_wt_prior_std;
  real delta_eps_RA_wt_prior_mean;
  real delta_eps_RA_wt_prior_std;
  
}

transformed data {
  real hill_n;
  real N_NS;
  vector[19] x_out;
  int num_var;           // number of variants
  
  num_var = 1;
  
  hill_n = 2;
  N_NS = 3*4600000;
  
  x_out[1] = 0;
  for (i in 2:19) {
    x_out[i] = 2^(i-2);
  }
  
}

parameters {
  // In this version of the model, the base parameters belong to the first variant (the wild-type)
  //     and there is a delta_param associated with each mutation (with additive effects)
  //     plus an epistasis term associated with each variant other than the wild-type
  real log_k_a_wt;         // log10 of IPTG binding affinity to active state
  
  real log_k_i_wt;         // log10 of IPTG binding affinity to inactive state
  
  real delta_eps_AI_wt;    // free energy difference between active and inactive states
  
  real delta_eps_RA_wt;    // free energy for Active TF to operator
  
  real log_g_max;                // log10 of maximum possible gene expression
  real<lower=0> log_copy_num;    // log10 of plasmid/operator copy number
  real<lower=0> log_R;           // log10 of repressor dimer copy number
  
  real<lower=0> sigma;  // scale factor for standard deviation of noise in y
  
}

transformed parameters {
  vector[num_var] K_A;
  vector[num_var] K_I;
  vector[num_var] log_k_a_var;
  vector[num_var] log_k_i_var;
  vector[num_var] delta_eps_AI_var;
  vector[num_var] delta_eps_RA_var;
  
  real g_max;
  real N_S;
  real R;
  
  vector[N] mean_y;
  vector[N_contr] mean_y_contr;
  
  log_k_a_var[1] = log_k_a_wt;
  log_k_i_var[1] = log_k_i_wt;
  delta_eps_AI_var[1] = delta_eps_AI_wt;
  delta_eps_RA_var[1] = delta_eps_RA_wt;
  
  K_A[1] = 10^log_k_a_var[1];
  K_I[1] = 10^log_k_i_var[1];
  
  g_max = 10^log_g_max;
  N_S = 10^log_copy_num;
  R = 10^log_R;
  
  for (i in 1:N) {
    real c1;
    real c2;
    real pA;
	real xRA;
	real lam;
	real fold_change;
	
    c1 = (1 + x[i]/K_A[variant[i]])^hill_n;
    c2 = ( (1 + x[i]/K_I[variant[i]])^hill_n ) * exp(-delta_eps_AI_var[variant[i]]);
	pA = c1/(c1+c2);
	xRA = exp(-delta_eps_RA_var[variant[i]]);
	
	lam = -N_NS + pA*R - N_S*xRA + pA*R*xRA;
    lam = lam + sqrt(4*pA*R*xRA*(N_NS + N_S - pA*R) + (N_NS + N_S*xRA - pA*R*(1 + xRA))^2);
    lam = lam/(2*xRA*(N_NS + N_S - pA*R));
	
    fold_change = 1/(1 + lam*xRA);
	
    mean_y[i] = g_max*fold_change;
  }
  
  for (i in 1:N_contr) {
    mean_y_contr[i] = g_max;
  }
  
}

model {
  vector[N] log_mean_y;
  vector[N_contr] log_mean_y_contr;
  
  log_mean_y = log(mean_y);
  log_mean_y_contr = log(mean_y_contr);

  // priors on free energy params
  log_k_a_wt ~ normal(log_k_a_wt_prior_mean, log_k_a_wt_prior_std);
  log_k_i_wt ~ normal(log_k_i_wt_prior_mean, log_k_i_wt_prior_std);
  delta_eps_AI_wt ~ normal(delta_eps_AI_wt_prior_mean, delta_eps_AI_wt_prior_std);
  delta_eps_RA_wt ~ normal(delta_eps_RA_wt_prior_mean, delta_eps_RA_wt_prior_std);
  
  // prior on max output level
  log_g_max ~ normal(log10(y_max), g_max_prior_width);
  
  // priors on plasmid/operator and repressor dimer copy numbers
  log_copy_num ~ normal(log10(copy_num_prior_mean), copy_num_prior_width);
  log_R ~ normal(log10(R_prior_mean), log_R_prior_width);
  
  // prior on scale parameter for log-normal measurement error
  sigma ~ normal(0, 1);
  
  // model of the data (dose-response curve with noise)
  y ~ lognormal(log_mean_y, sigma);
  
  // model of the control strain data (constant, max output)
  y_contr ~ lognormal(log_mean_y_contr, sigma);

}

generated quantities {
  // Local variables
  real y_out[num_var, 19];
  real fc_out[num_var, 19];
  
  for (var in 1:num_var) {
    for (i in 1:19) {
      real c1;
      real c2;
      real pA;
	  real xRA;
	  real lam;
	  real fold_change;
	  
      c1 = (1 + x_out[i]/K_A[var])^hill_n;
      c2 = ( (1 + x_out[i]/K_I[var])^hill_n ) * exp(-delta_eps_AI_var[var]);
	  pA = c1/(c1+c2);
	  xRA = exp(-delta_eps_RA_var[var]);
	
	  lam = -N_NS + pA*R - N_S*xRA + pA*R*xRA;
      lam = lam + sqrt(4*pA*R*xRA*(N_NS + N_S - pA*R) + (N_NS + N_S*xRA - pA*R*(1 + xRA))^2);
      lam = lam/(2*xRA*(N_NS + N_S - pA*R));
	
      fold_change = 1/(1 + lam*xRA);
	
      y_out[var, i] = g_max*fold_change;
      fc_out[var, i] = fold_change;
    }
  }
  
}
