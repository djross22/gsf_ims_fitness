//
//

data {
  int<lower=1> N_lig;                // number of non-zero ligand concentrations for each ligand
  
  vector[N_lig] x_1;                 // non-zero ligand 1 concentrations
  vector[N_lig] x_2;                 // non-zero ligand 2 concentrations
  
  real y_0_low_tet;                  // fitness difference with zero ligand and medium tet concentration
  real y_0_low_tet_err;              // estimated error of fitness difference
  
  vector[N_lig] y_1_low_tet;         // fitness difference with ligand 1 and medium tet concentration
  vector[N_lig] y_1_low_tet_err;     // estimated error of fitness difference
  vector[N_lig] y_2_low_tet;         // fitness difference with ligand 2 and medium tet concentration
  vector[N_lig] y_2_low_tet_err;     // estimated error of fitness difference
  
  vector[N_lig] y_1_high_tet;        // fitness difference with ligand 1 and high tet concentration
  vector[N_lig] y_1_high_tet_err;    // estimated error of fitness difference
  vector[N_lig] y_2_high_tet;        // fitness difference with ligand 2 and high tet concentration
  vector[N_lig] y_2_high_tet_err;    // estimated error of fitness difference
  
  real log_g_min;                    // lower bound on log_low_level and log_high_level
  real log_g_max;                    // upper bound on log_low_level and log_high_level
  
  real low_fitness_mu_low_tet;       // fitness difference at zero gene expression, medium tet
  real mid_g_mu_low_tet;             // gene expression level at 1/2 max fitness difference, medium tet
  real fitness_n_mu_low_tet;         // cooperativity coefficient of fitness difference curve, medium tet
  
  real low_fitness_std_low_tet;       // fitness difference at zero gene expression, medium tet
  real mid_g_std_low_tet;             // gene expression level at 1/2 max fitness difference, medium tet
  real fitness_n_std_low_tet;         // cooperativity coefficient of fitness difference curve, medium tet
  
  real low_fitness_mu_high_tet;      // fitness difference at zero gene expression, high tet
  real mid_g_mu_high_tet;            // gene expression level at 1/2 max fitness difference, high tet
  real fitness_n_mu_high_tet;        // cooperativity coefficient of fitness difference curve, high tet
  
  real low_fitness_std_high_tet;      // fitness difference at zero gene expression, high tet
  real mid_g_std_high_tet;            // gene expression level at 1/2 max fitness difference, high tet
  real fitness_n_std_high_tet;        // cooperativity coefficient of fitness difference curve, high tet
  
}

transformed data {
  int N;                       // number of grid points for GP
  array[2*N_lig+1] vector[2] x_gp;   // array of 2D coordinates of the log(ligand concentrations)
  real sqrt_pi = sqrt(pi());
  real center_log_g;
  real log_spacing;
  real zero_spacing_factor;
  vector[2] log_x_zero;
  
  N = 2*N_lig+1;
  
  center_log_g = (log_g_min + log_g_max)/2;
  
  // this sets the zero-ligand coordinates on the log(x) scale at a distance of 1.5x the dilution spacing of the non=zero-ligands 
  log_spacing = log10(x_1[2]) - log10(x_1[1]);
  zero_spacing_factor = 1.5;
  log_x_zero[1] = log10(x_1[1]) - zero_spacing_factor*log_spacing;
  log_x_zero[2] = log10(x_2[1]) - zero_spacing_factor*log_spacing;
  
  // First grid point is zero-ligand
  x_gp[1][1] = log_x_zero[1];
  x_gp[1][2] = log_x_zero[2];
  
  for (i in 1:N_lig){
    // Grid points 2:N_lig+1 are the samples with non-zero ligand 1
    x_gp[i+1][1] = log10(x_1[i]);
    x_gp[i+1][2] = log_x_zero[2];
	
	// Grid points 2+N_lig:2*N_lig+1 are the samples with non-zero ligand 2
    x_gp[i+1+N_lig][1] = log_x_zero[1];
    x_gp[i+1+N_lig][2] = log10(x_2[i]);
  }
}

parameters {
  real<lower=0> sigma;            // scale factor for standard deviation of noise in fitness differnce, y
  
  real low_fitness_low_tet;       // fitness difference at zero gene expression, med tet
  real mid_g_low_tet;             // gene expression level at 1/2 max fitness difference, med tet
  real fitness_n_low_tet;         // cooperativity coefficient of fitness difference curve, med tet
  
  // parameters of the fitness vs. g curve
  real low_fitness_high_tet;      // fitness difference at zero gene expression
  real mid_g_high_tet;            // gene expression level at 1/2 max fitness difference
  real fitness_n_high_tet;        // cooperativity coefficient of fitness difference curve

  // gp params
  real<lower=0> rho;
  real<lower=0> alpha;
  vector[N] eta;

}

transformed parameters {
  real mean_y_0_low_tet;
  
  vector[N_lig] mean_y_1_low_tet;
  vector[N_lig] mean_y_1_high_tet;
  vector[N_lig] mean_y_2_low_tet;
  vector[N_lig] mean_y_2_high_tet;
  
  real g0;
  vector[N_lig] g_1;                // gene expression level at each non-zero concentration of ligand 1
  vector[N_lig] g_2;                // gene expression level at each non-zero concentration of ligand 2
  
  vector[N] log_g;         // the GP function, analogous to y in all of the Stan examples
  vector[N] constr_log_g;  // log10 gene expression, constrained to be between log_g_min and log_g_max

  {
    matrix[N, N] L_K;
    matrix[N, N] K = gp_exp_quad_cov(x_gp, alpha, rho);
    real term1;
    real term2;
    real term3;

    // diagonal elements
    for (n in 1:N)
      K[n, n] = K[n, n] + 1e-9;

    L_K = cholesky_decompose(K);

    log_g = L_K * eta;
	
    constr_log_g = log_g_min + (log_g_max - log_g_min)*inv_logit(log_g);

  }
  
  // First grid point is zero-ligand
  g0 = 10^constr_log_g[1];
  
  mean_y_0_low_tet = low_fitness_low_tet - low_fitness_low_tet*(g0^fitness_n_low_tet)/(mid_g_low_tet^fitness_n_low_tet + g0^fitness_n_low_tet);
  
  for (i in 1:N_lig) {
    // Grid points 2:N_lig+1 are the samples with non-zero ligand 1
	g_1[i] = 10^constr_log_g[i+1];
	
	// Grid points 2+N_lig:2*N_lig+1 are the samples with non-zero ligand 2
	g_2[i] = 10^constr_log_g[i+1+N_lig];
	
    mean_y_1_low_tet[i] = low_fitness_low_tet - low_fitness_low_tet*(g_1[i]^fitness_n_low_tet)/(mid_g_low_tet^fitness_n_low_tet + g_1[i]^fitness_n_low_tet);
    mean_y_2_low_tet[i] = low_fitness_low_tet - low_fitness_low_tet*(g_2[i]^fitness_n_low_tet)/(mid_g_low_tet^fitness_n_low_tet + g_2[i]^fitness_n_low_tet);
	
    mean_y_1_high_tet[i] = low_fitness_high_tet - low_fitness_high_tet*(g_1[i]^fitness_n_high_tet)/(mid_g_high_tet^fitness_n_high_tet + g_1[i]^fitness_n_high_tet);
    mean_y_2_high_tet[i] = low_fitness_high_tet - low_fitness_high_tet*(g_2[i]^fitness_n_high_tet)/(mid_g_high_tet^fitness_n_high_tet + g_2[i]^fitness_n_high_tet);
  }

  
}

model {
  // informative priors on fitness calibration params
  low_fitness_low_tet ~ normal(low_fitness_mu_low_tet, low_fitness_std_low_tet);
  mid_g_low_tet ~ normal(mid_g_mu_low_tet, mid_g_std_low_tet);
  fitness_n_low_tet ~ normal(fitness_n_mu_low_tet, fitness_n_std_low_tet);
  
  low_fitness_high_tet ~ normal(low_fitness_mu_high_tet, low_fitness_std_high_tet);
  mid_g_high_tet ~ normal(mid_g_mu_high_tet, mid_g_std_high_tet);
  fitness_n_high_tet ~ normal(fitness_n_mu_high_tet, fitness_n_std_high_tet);
  
  // noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);

  // GP
  rho ~ inv_gamma(5, 5);
  alpha ~ normal(1, 1);
  eta ~ std_normal();
  
  y_0_low_tet ~ normal(mean_y_0_low_tet, sigma*y_0_low_tet_err);
  
  y_1_low_tet ~ normal(mean_y_1_low_tet, sigma*y_1_low_tet_err);
  y_2_low_tet ~ normal(mean_y_2_low_tet, sigma*y_2_low_tet_err);
  
  y_1_high_tet ~ normal(mean_y_1_high_tet, sigma*y_1_high_tet_err);
  y_2_high_tet ~ normal(mean_y_2_high_tet, sigma*y_2_high_tet_err);

}

generated quantities {
  real rms_resid;
  real log_rho;
  real log_alpha;
  real log_sigma;
  
  vector[N_lig+1] log_g_1;    // log-gene expression level at each concentration of ligand 1, including zero
  vector[N_lig+1] log_g_2;    // log-gene expression level at each concentration of ligand 2, including zero
  vector[N_lig+1] log_g_ratio_1;    // log-ratio of gene expression at each concentration to gene expression at zero
  vector[N_lig+1] log_g_ratio_2;    // log-ratio of gene expression at each concentration to gene expression at zero

  vector[N_lig+1] dlog_g_1; // derivative of the gp along the ligand-1 direction
  vector[N_lig+1] dlog_g_2; // derivative of the gp along the ligand-2 direction
  
  real mean_y_0_high_tet;
  
  vector[N_lig+1] y_1_out_low_tet;
  vector[N_lig+1] y_1_out_high_tet;
  vector[N_lig+1] y_2_out_low_tet;
  vector[N_lig+1] y_2_out_high_tet;
  
  mean_y_0_high_tet = low_fitness_high_tet - low_fitness_high_tet*(g0^fitness_n_high_tet)/(mid_g_high_tet^fitness_n_high_tet + g0^fitness_n_high_tet);
  
  y_1_out_low_tet[1] = mean_y_0_low_tet;
  y_2_out_low_tet[1] = mean_y_0_low_tet;
  y_1_out_high_tet[1] = mean_y_0_high_tet;
  y_2_out_high_tet[1] = mean_y_0_high_tet;
  
  rms_resid = sqrt(distance(y_1_low_tet, mean_y_1_low_tet)^2 + distance(y_2_low_tet, mean_y_2_low_tet)^2 + distance(y_1_high_tet, mean_y_1_high_tet)^2 + distance(y_2_high_tet, mean_y_2_high_tet)^2)/sqrt(4*N_lig + 1);
  
  log_rho = log10(rho);
  log_alpha = log10(alpha);
  log_sigma = log10(sigma);
  
  log_g_1[1] = log10(g0);
  log_g_2[1] = log10(g0);
  for (i in 1:N_lig) {
    log_g_1[i+1] = log10(g_1[i]);
    log_g_2[i+1] = log10(g_2[i]);
	
	y_1_out_low_tet[i+1] = mean_y_1_low_tet[i];
    y_2_out_low_tet[i+1] = mean_y_2_low_tet[i];
    y_1_out_high_tet[i+1] = mean_y_1_high_tet[i];
    y_2_out_high_tet[i+1] = mean_y_2_high_tet[i];
  }

  // derivative calculation: end points based on difference with next point toward the middle
  //     non-end points are averaaged
  dlog_g_1[1] = (log_g_1[2] - log_g_1[1]) / (zero_spacing_factor*log_spacing);
  dlog_g_2[1] = (log_g_2[2] - log_g_2[1]) / (zero_spacing_factor*log_spacing);
  
  dlog_g_1[2] = (log_g_1[3] - log_g_1[1]) / ((1+zero_spacing_factor)*log_spacing);
  dlog_g_2[2] = (log_g_2[3] - log_g_2[1]) / ((1+zero_spacing_factor)*log_spacing);
  
  for (i in 3:N_lig) {
    dlog_g_1[i] = (log_g_1[i+1] - log_g_1[i-1]) / (2*log_spacing);
    dlog_g_2[i] = (log_g_2[i+1] - log_g_2[i-1]) / (2*log_spacing);
  
  dlog_g_1[N_lig+1] = (log_g_1[N_lig+1] - log_g_1[N_lig]) / (log_spacing);
  dlog_g_2[N_lig+1] = (log_g_2[N_lig+1] - log_g_2[N_lig]) / (log_spacing);
  }
  
  log_g_ratio_1 = log_g_1 - log_g_1[1];
  log_g_ratio_2 = log_g_2 - log_g_2[1];
  
}
