// Fit dose-response curves to Phillips lab model for allosteric TFs
//  This version has some of the parameters in the multi-operator model held constant.
//  It is intended to be used as a way to more rapidly generate a reasonable set of parameters to initilize the full model.
//

data {
  int<lower=1> N_contr;          // number of data points for control strains LacI deletions
  vector[N_contr] y_contr;       // gene expression (from cytometry) for control strains
  vector[N_contr] y_contr_err;   // estimated error of gene expression for control strains
  
  int rep_contr[N_contr];        // integer to indicate the measurement replicate for controls
  int<lower=1> num_contr_reps;   // number of measurement replicates for controls
  
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // gene expression (from cytometry) at each concentration
  vector[N] y_err;       // estimated error of gene expression at each concentration
  
  real y_max;             // geometric mean for prior on maximum gene expression value
  real g_max_prior_width; // geometric std for prior on maximum gene expression value
  
  real<lower=2> copy_num_prior_mean;  // geometric mean for prior on plasmid/operator copy number
  real<lower=1> R_prior_mean;         // geometric mean for prior on repressor dimer copy number
  real<lower=0> copy_num_prior_width; // geometric std for priors on plasmid/operator and repressor dimer copy numbers
  
  int rep[N];            // integer to indicate the measurement replicate
  int<lower=1> num_reps; // number of measurement replicates (for all variants)
  
  int<lower=2> num_var;  // number of variants
  int variant[N];        // numerical index to indicate variants
  int<lower=0> num_epi_var;  // number of variants with more than one mutation (only define epistasis for these)
  
  int<lower=1> num_mut;  // number of differrent mutations
  int<lower=0, upper=1> mut_code[num_var-1, num_mut];   // one-hot encoding for  presense of mutations in each variant; variant 0 (WT) has no mutations
  real<lower=0> eps_RA_prior_scale[num_mut];   // scale multiplier for width of prior on operator binding free energy term (delta_eps_RA_mut) 
  real<lower=0> RA_epi_prior_scale[num_epi_var]; // scale multiplier for width of prior on operator binding free energy epistasis (delta_eps_RA_epi) 
  
  // priors on wild-type free energy parameters
  real log_k_a_wt_prior_mean;
  real log_k_a_wt_prior_std;
  real log_k_i_wt_prior_mean;
  real log_k_i_wt_prior_std;
  real delta_eps_AI_wt_prior_mean;
  real delta_eps_AI_wt_prior_std;
  real delta_eps_RA_wt_prior_mean;
  real delta_eps_RA_wt_prior_std;
  
  real delta_prior_width; // width of prior on delta-parameters
  real epi_prior_width_2;   // width of prior on parameter epistasis
  
  real rep_ratio_scale;   // parameter to set the scale for the half-normal prior on log_rep_ratio
  real rep_offset_scale;  // parameter to set the scale for the half-normal prior on log_rep_ratio
  
  // These are parameters in the full model, set as constants in this version
  real log_g_max;                // log10 of maximum possible gene expression
  real<lower=0> log_copy_num;    // log10 of plasmid/operator copy number
  real<lower=0> log_R;           // log10 of repressor dimer copy number
  
  //real<lower=0> rep_ratio_sigma;
  //real<lower=0> rep_offset_sigma;
}

transformed data {
  real hill_n;
  real N_NS;
  vector[19] x_out;
  int num_non_epi_var;  // number of variants with less than two mutations
  
  // These are transformed parameters in the full model, set as constants in this version
  real g_max;
  real N_S;
  real R;
  
  hill_n = 2;
  N_NS = 3*4600000;
  
  x_out[1] = 0;
  for (i in 2:19) {
    x_out[i] = 2^(i-2);
  }
  
  num_non_epi_var = num_var - num_epi_var;
  
  g_max = 10^log_g_max;
  N_S = 10^log_copy_num;
  R = 10^log_R;
}

parameters {
  // In this version of the model, the base parameters belong to the first variant (the wild-type)
  //     and there is a delta_param associated with each mutation (with additive effects)
  //     plus an epistasis term associated with each variant other than the wild-type
  real log_k_a_wt;         // log10 of IPTG binding affinity to active state
  vector[num_mut] log_k_a_mut;
  vector[num_epi_var] log_k_a_epi;
  
  real log_k_i_wt;         // log10 of IPTG binding affinity to inactive state
  vector[num_mut] log_k_i_mut;
  vector[num_epi_var] log_k_i_epi;
  
  real delta_eps_AI_wt;    // free energy difference between active and inactive states
  vector[num_mut] delta_eps_AI_mut;
  vector[num_epi_var] delta_eps_AI_epi;
  
  real delta_eps_RA_wt;    // free energy for Active TF to operator
  vector[num_mut] delta_eps_RA_mut;
  vector[num_epi_var] delta_eps_RA_epi;
  
  real<lower=0> sigma;  // scale factor for standard deviation of noise in y
  
}

transformed parameters {
  vector[num_var] K_A;
  vector[num_var] K_I;
  vector[num_var] log_k_a_var;
  vector[num_var] log_k_i_var;
  vector[num_var] delta_eps_AI_var;
  vector[num_var] delta_eps_RA_var;
  
  vector[N] mean_y;
  
  log_k_a_var[1] = log_k_a_wt;
  log_k_i_var[1] = log_k_i_wt;
  delta_eps_AI_var[1] = delta_eps_AI_wt;
  delta_eps_RA_var[1] = delta_eps_RA_wt;
  
  K_A[1] = 10^log_k_a_var[1];
  K_I[1] = 10^log_k_i_var[1];
  
  for (var in 2:num_var) {
    if (var > num_non_epi_var) {
      log_k_a_var[var] = log_k_a_wt + log_k_a_epi[var-num_non_epi_var];
      log_k_i_var[var] = log_k_i_wt + log_k_i_epi[var-num_non_epi_var];
      delta_eps_AI_var[var] = delta_eps_AI_wt + delta_eps_AI_epi[var-num_non_epi_var];
      delta_eps_RA_var[var] = delta_eps_RA_wt + delta_eps_RA_epi[var-num_non_epi_var];
	}
    else {
      log_k_a_var[var] = log_k_a_wt;
      log_k_i_var[var] = log_k_i_wt;
      delta_eps_AI_var[var] = delta_eps_AI_wt;
      delta_eps_RA_var[var] = delta_eps_RA_wt;
	}
	
	for (mut in 1:num_mut) {
	  log_k_a_var[var] += mut_code[var-1, mut]*log_k_a_mut[mut];
	  log_k_i_var[var] += mut_code[var-1, mut]*log_k_i_mut[mut];
	  delta_eps_AI_var[var] += mut_code[var-1, mut]*delta_eps_AI_mut[mut];
	  delta_eps_RA_var[var] += mut_code[var-1, mut]*delta_eps_RA_mut[mut];
	}
	
	K_A[var] = 10^log_k_a_var[var];
    K_I[var] = 10^log_k_i_var[var];
  }
  
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
  
}

model {
  vector[N] log_mean_y;
  
  log_mean_y = log(mean_y);
  
  // priors on free energy params
  log_k_a_wt ~ normal(log_k_a_wt_prior_mean, log_k_a_wt_prior_std);
  log_k_i_wt ~ normal(log_k_i_wt_prior_mean, log_k_i_wt_prior_std);
  delta_eps_AI_wt ~ normal(delta_eps_AI_wt_prior_mean, delta_eps_AI_wt_prior_std);
  delta_eps_RA_wt ~ normal(delta_eps_RA_wt_prior_mean, delta_eps_RA_wt_prior_std);
  
  log_k_a_mut ~ normal(0, delta_prior_width/2.3); // factor of 1/2.3 is to compensate for use of log10 instead of ln
  log_k_a_epi ~ normal(0, epi_prior_width_2/2.3);
  
  log_k_i_mut ~ normal(0, delta_prior_width/2.3); // factor of 1/2.3 is to compensate for use of log10 instead of ln
  log_k_i_epi ~ normal(0, epi_prior_width_2/2.3);
  
  delta_eps_AI_mut ~ normal(0, delta_prior_width);
  delta_eps_AI_epi ~ normal(0, epi_prior_width_2);
  
  for (mut in 1:num_mut) {
    delta_eps_RA_mut[mut] ~ normal(0, delta_prior_width*eps_RA_prior_scale[mut]);
  }
  for (var in 1:num_epi_var) {
    delta_eps_RA_epi[var] ~ normal(0, epi_prior_width_2*RA_epi_prior_scale[var]);
  }
  
  // prior on scale parameter for log-normal measurement error
  sigma ~ normal(0, 1);
  
  // model of the data (dose-response curve with noise)
  y ~ lognormal(log_mean_y, sigma);

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
