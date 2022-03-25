// Fit dose-response curves to Phillips lab model for allosteric TFs
//     This version sets the mutational shifts and epistasis strictly to zero for delta_eps_RA

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
  real<lower=0> copy_num_prior_width; // geometric std for prior on plasmid/operator copy number
  real<lower=1> R_prior_mean;         // geometric mean for prior on repressor dimer copy number
  real<lower=0> log_R_prior_width;    // geometric std for prior on repressor dimer copy number
  
  int rep[N];            // integer to indicate the measurement replicate
  int<lower=1> num_reps; // number of measurement replicates (for all variants)
  
  int<lower=2> num_var;  // number of variants
  int variant[N];        // numerical index to indicate variants
  int<lower=0> num_epi_var;  // number of variants with more than one mutation (only define epistasis for these)
  
  int<lower=1> num_mut;  // number of differrent mutations
  int<lower=0, upper=1> mut_code[num_var-1, num_mut];   // one-hot encoding for  presense of mutations in each variant; variant 0 (WT) has no mutations
  
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
  real epi_prior_width_1;   // width of 1st mixture component of prior on parameter epistasis
  real epi_prior_width_2;   // width of 2nd mixture component of prior on parameter epistasis
  real epi_prior_phi;       // weight for 1st mixture component of prior on parameter epistasis
  
  real rep_ratio_scale;   // parameter to set the scale for the half-normal prior on log_rep_ratio
  
}

transformed data {
  real ln_10;
  real hill_n;
  real N_NS;
  vector[19] x_out;
  int num_non_epi_var;  // number of variants with less than two mutations
  real log_phi_1;
  real log_phi_2;
  
  log_phi_1 = log(epi_prior_phi);
  log_phi_2 = log(1 - epi_prior_phi);
  
  hill_n = 2;
  N_NS = 3*4600000;
  
  x_out[1] = 0;
  for (i in 2:19) {
    x_out[i] = 2^(i-2);
  }
  
  num_non_epi_var = num_var - num_epi_var;
  
  ln_10 = log(10.0);
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
  
  real log_g_max;                // log10 of maximum possible gene expression
  real<lower=0> log_copy_num;    // log10 of plasmid/operator copy number
  real<lower=0> log_R;           // log10 of repressor dimer copy number
  
  real<lower=0> sigma;  // scale factor for standard deviation of noise in y
  
  vector[num_reps] log_rep_ratio;              // log10 of multiplicative correction factor for different replicates
  vector[num_contr_reps] log_rep_ratio_contr;  // log10 of multiplicative correction factor for control replicates
  real<lower=0> rep_ratio_sigma;               // hyper-paramters for log_rep_ratio and log_rep_ratio_contr
  
}

transformed parameters {
  vector[num_var] K_A;
  vector[num_var] K_I;
  vector[num_var] log_k_a_var;
  vector[num_var] log_k_i_var;
  vector[num_var] delta_eps_AI_var;
  
  //real g_max;
  real N_S;
  real R;
  
  //vector[N] mean_y;
  //vector[N_contr] mean_y_contr;
  vector[N] log_mean_y;
  vector[N_contr] log_mean_y_contr;
  
  log_k_a_var[1] = log_k_a_wt;
  log_k_i_var[1] = log_k_i_wt;
  delta_eps_AI_var[1] = delta_eps_AI_wt;
  
  K_A[1] = 10^log_k_a_var[1];
  K_I[1] = 10^log_k_i_var[1];
  
  for (var in 2:num_var) {
    if (var > num_non_epi_var) {
      log_k_a_var[var] = log_k_a_wt + log_k_a_epi[var-num_non_epi_var];
      log_k_i_var[var] = log_k_i_wt + log_k_i_epi[var-num_non_epi_var];
      delta_eps_AI_var[var] = delta_eps_AI_wt + delta_eps_AI_epi[var-num_non_epi_var];
	}
    else {
      log_k_a_var[var] = log_k_a_wt;
      log_k_i_var[var] = log_k_i_wt;
      delta_eps_AI_var[var] = delta_eps_AI_wt;
	}
	
	for (mut in 1:num_mut) {
	  log_k_a_var[var] += mut_code[var-1, mut]*log_k_a_mut[mut];
	  log_k_i_var[var] += mut_code[var-1, mut]*log_k_i_mut[mut];
	  delta_eps_AI_var[var] += mut_code[var-1, mut]*delta_eps_AI_mut[mut];
	}
	
	K_A[var] = 10^log_k_a_var[var];
    K_I[var] = 10^log_k_i_var[var];
  }
  
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
	xRA = exp(-delta_eps_RA_wt);
	
	lam = -N_NS + pA*R - N_S*xRA + pA*R*xRA;
    lam = lam + sqrt(4*pA*R*xRA*(N_NS + N_S - pA*R) + (N_NS + N_S*xRA - pA*R*(1 + xRA))^2);
    lam = lam/(2*xRA*(N_NS + N_S - pA*R));
	
    fold_change = 1/(1 + lam*xRA);
	
    //mean_y[i] = g_max*fold_change;
    log_mean_y[i] = ln_10*log_g_max + log(fold_change) + log_rep_ratio[rep[i]];
  }
  
  for (i in 1:N_contr) {
    //mean_y_contr[i] = g_max;
    log_mean_y_contr[i] = ln_10*log_g_max + log_rep_ratio_contr[rep_contr[i]];
  }
  
}

model {
  // priors on free energy params
  log_k_a_wt ~ normal(log_k_a_wt_prior_mean, log_k_a_wt_prior_std);
  log_k_i_wt ~ normal(log_k_i_wt_prior_mean, log_k_i_wt_prior_std);
  delta_eps_AI_wt ~ normal(delta_eps_AI_wt_prior_mean, delta_eps_AI_wt_prior_std);
  delta_eps_RA_wt ~ normal(delta_eps_RA_wt_prior_mean, delta_eps_RA_wt_prior_std);
  
  log_k_a_mut ~ normal(0, delta_prior_width/ln_10); // factor of 1/ln_10 is to compensate for use of log10 instead of ln
  
  log_k_i_mut ~ normal(0, delta_prior_width/ln_10); // factor of 1/ln_10 is to compensate for use of log10 instead of ln
  
  delta_eps_AI_mut ~ normal(0, delta_prior_width);
  
  for (var in 1:num_epi_var) {
    // factor of 1/ln_10 is to compensate for use of log10 instead of ln
	target += log_sum_exp(log_phi_1 + normal_lpdf(log_k_a_epi[var] | 0, epi_prior_width_1/ln_10), log_phi_2 + normal_lpdf(log_k_a_epi[var] | 0, epi_prior_width_2/ln_10));
	target += log_sum_exp(log_phi_1 + normal_lpdf(log_k_i_epi[var] | 0, epi_prior_width_1/ln_10), log_phi_2 + normal_lpdf(log_k_i_epi[var] | 0, epi_prior_width_2/ln_10));
	
	target += log_sum_exp(log_phi_1 + normal_lpdf(delta_eps_AI_epi[var] | 0, epi_prior_width_1), log_phi_2 + normal_lpdf(delta_eps_AI_epi[var] | 0, epi_prior_width_2));
  }
  
  // prior on max output level
  log_g_max ~ normal(log10(y_max), g_max_prior_width);
  
  // priors on plasmid/operator and repressor dimer copy numbers
  log_copy_num ~ normal(log10(copy_num_prior_mean), copy_num_prior_width);
  log_R ~ normal(log10(R_prior_mean), log_R_prior_width);
  
  // prior on scale parameter for log-normal measurement error
  sigma ~ normal(0, 1);
  
  // prior on scale hyper-paramter for log_rep_ratio
  rep_ratio_sigma ~ normal(0, rep_ratio_scale);
  
  // priors on log_rep_ratio and log_rep_ratio_contr
  log_rep_ratio ~ normal(0, rep_ratio_sigma);
  log_rep_ratio_contr ~ normal(0, rep_ratio_sigma);
  
  // model of the data (dose-response curve with noise)
  y ~ lognormal(log_mean_y, sigma);
  
  // model of the control strain data (constant, max output)
  y_contr ~ lognormal(log_mean_y_contr, sigma);

}

generated quantities {
  vector[num_var] delta_eps_RA_var;
  vector[num_mut] delta_eps_RA_mut;
  vector[num_epi_var] delta_eps_RA_epi;
  real g_max;
  real y_out[num_var, 19];
  real fc_out[num_var, 19];
  
  for (n in 1:num_mut) {
    delta_eps_RA_mut[n] = 0;
  }
  for (n in 1:num_epi_var) {
    delta_eps_RA_epi[n] = 0;
  }
  
  g_max = 10^log_g_max;
  
  for (var in 1:num_var) {
    delta_eps_RA_var[var] = delta_eps_RA_wt;
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
