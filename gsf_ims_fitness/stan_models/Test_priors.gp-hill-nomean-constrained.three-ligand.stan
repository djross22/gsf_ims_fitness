//
//

data {
  int<lower=1> N_lig;                // number of non-zero ligand concentrations for each ligand
  
  vector[N_lig] x_1;                 // non-zero ligand 1 concentrations
  vector[N_lig] x_2;                 // non-zero ligand 2 concentrations
  vector[N_lig] x_3;                 // non-zero ligand 3 concentrations
  
  real log_g_min;                    // lower bound on log_low_level and log_high_level
  real log_g_max;                    // upper bound on log_low_level and log_high_level
  
  real high_fitness_mu;      // fitness difference at +infinite gene expression, with antibiotic
  real mid_g_mu;             // gene expression level at 1/2 max fitness difference, with antibiotic
  real fitness_n_mu;         // cooperativity coefficient of fitness difference curve, with antibiotic
  
  real high_fitness_std;       // fitness difference at +infinite gene expression, with antibiotic
  real mid_g_std;             // gene expression level at 1/2 max fitness difference, with antibiotic
  real fitness_n_std;         // cooperativity coefficient of fitness difference curve, with antibiotic
  
}

transformed data {
  int N;                       // number of grid points for GP
  vector[3] x_gp[3*N_lig+1];   // array of 3D coordinates of the log(ligand concentrations)
  real sqrt_pi = sqrt(pi());
  real center_log_g;
  real log_spacing;
  real zero_spacing_factor;
  vector[3] log_x_zero;
  real low_fitness;
  
  low_fitness = 0;
  
  N = 3*N_lig+1;
  
  center_log_g = (log_g_min + log_g_max)/2;
  
  // this sets the zero-ligand coordinates on the log(x) scale at a distance of 1.5x the dilution spacing of the non=zero-ligands 
  log_spacing = log10(x_1[2]) - log10(x_1[1]);
  zero_spacing_factor = 1.5;
  log_x_zero[1] = log10(x_1[1]) - zero_spacing_factor*log_spacing;
  log_x_zero[2] = log10(x_2[1]) - zero_spacing_factor*log_spacing;
  log_x_zero[3] = log10(x_3[1]) - zero_spacing_factor*log_spacing;
  
  // First grid point is zero-ligand
  x_gp[1][1] = log_x_zero[1];
  x_gp[1][2] = log_x_zero[2];
  x_gp[1][3] = log_x_zero[3];
  
  for (i in 1:N_lig){
    // Grid points 2:N_lig+1 are the samples with non-zero ligand 1
    x_gp[i+1][1] = log10(x_1[i]);
    x_gp[i+1][2] = log_x_zero[2];
    x_gp[i+1][3] = log_x_zero[3];
	
	// Grid points 2+N_lig:2*N_lig+1 are the samples with non-zero ligand 2
    x_gp[i+1+N_lig][1] = log_x_zero[1];
    x_gp[i+1+N_lig][2] = log10(x_2[i]);
    x_gp[i+1+N_lig][3] = log_x_zero[3];
	
	// Grid points 2*N_lig+2:3*N_lig+1 are the samples with non-zero ligand 3
    x_gp[i+1+2*N_lig][1] = log_x_zero[1];
    x_gp[i+1+2*N_lig][2] = log_x_zero[2];
    x_gp[i+1+2*N_lig][3] = log10(x_3[i]);
  }
}

parameters {
  real<lower=0> sigma;            // scale factor for standard deviation of noise in fitness differnce, y
  
  real high_fitness;       // fitness difference at +infinite gene expression, with antibiotic
  real mid_g;             // gene expression level at 1/2 max fitness difference, with antibiotic
  real fitness_n;         // cooperativity coefficient of fitness difference curve, with antibiotic

  // gp params
  real<lower=0> rho;
  real<lower=0> alpha;
  vector[N] eta;

}

transformed parameters {
  real mean_y_0;
  
  vector[N_lig] mean_y_1;
  vector[N_lig] mean_y_2;
  vector[N_lig] mean_y_3;
  
  real g0;
  vector[N_lig] g_1;                // gene expression level at each non-zero concentration of ligand 1
  vector[N_lig] g_2;                // gene expression level at each non-zero concentration of ligand 2
  vector[N_lig] g_3;                // gene expression level at each non-zero concentration of ligand 3
  
  vector[N] log_g;         // the GP function, analogous to y in all of the Stan examples
  vector[N] constr_log_g;  // log10 gene expression, constrained to be between log_g_min and log_g_max

  {
    matrix[N, N] L_K;
    matrix[N, N] K = cov_exp_quad(x_gp, alpha, rho);
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
  
  mean_y_0 = low_fitness + (high_fitness - low_fitness)*(g0^fitness_n)/(mid_g^fitness_n + g0^fitness_n);
  
  for (i in 1:N_lig) {
    // Grid points 2:N_lig+1 are the samples with non-zero ligand 1
	g_1[i] = 10^constr_log_g[i+1];
	
	// Grid points 2+N_lig:2*N_lig+1 are the samples with non-zero ligand 2
	g_2[i] = 10^constr_log_g[i+1+N_lig];
	
	// Grid points 2*N_lig+2:3*N_lig+1 are the samples with non-zero ligand 3
	g_3[i] = 10^constr_log_g[i+1+2*N_lig];
	
    mean_y_1[i] = low_fitness + (high_fitness - low_fitness)*(g_1[i]^fitness_n)/(mid_g^fitness_n + g_1[i]^fitness_n);
    mean_y_2[i] = low_fitness + (high_fitness - low_fitness)*(g_2[i]^fitness_n)/(mid_g^fitness_n + g_2[i]^fitness_n);
    mean_y_3[i] = low_fitness + (high_fitness - low_fitness)*(g_3[i]^fitness_n)/(mid_g^fitness_n + g_3[i]^fitness_n);
  }

  
}

model {
  // informative priors on fitness calibration params
  high_fitness ~ normal(high_fitness_mu, high_fitness_std);
  mid_g ~ normal(mid_g_mu, mid_g_std);
  fitness_n ~ normal(fitness_n_mu, fitness_n_std);
  
  // noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);

  // GP
  rho ~ inv_gamma(5, 5);
  alpha ~ normal(1, 1);
  eta ~ std_normal();

}

generated quantities {
  real log_rho;
  real log_alpha;
  real log_sigma;
  
  vector[N_lig+1] log_g_1;    // log-gene expression level at each concentration of ligand 1, including zero
  vector[N_lig+1] log_g_2;    // log-gene expression level at each concentration of ligand 2, including zero
  vector[N_lig+1] log_g_3;    // log-gene expression level at each concentration of ligand 3, including zero

  vector[N_lig+1] dlog_g_1; // derivative of the gp along the ligand-1 direction
  vector[N_lig+1] dlog_g_2; // derivative of the gp along the ligand-2 direction
  vector[N_lig+1] dlog_g_3; // derivative of the gp along the ligand-3 direction
  
  
  vector[N_lig+1] y_1_out;
  vector[N_lig+1] y_2_out;
  vector[N_lig+1] y_3_out;
  
  y_1_out[1] = mean_y_0;
  y_2_out[1] = mean_y_0;
  y_3_out[1] = mean_y_0;
    
  log_rho = log10(rho);
  log_alpha = log10(alpha);
  log_sigma = log10(sigma);
  
  log_g_1[1] = log10(g0);
  log_g_2[1] = log10(g0);
  log_g_3[1] = log10(g0);
  for (i in 1:N_lig) {
    log_g_1[i+1] = log10(g_1[i]);
    log_g_2[i+1] = log10(g_2[i]);
    log_g_3[i+1] = log10(g_3[i]);
	
	y_1_out[i+1] = mean_y_1[i];
    y_2_out[i+1] = mean_y_2[i];
    y_3_out[i+1] = mean_y_3[i];
  }

  // derivative calculation: end points based on difference with next point toward the middle
  //     non-end points are averaaged
  dlog_g_1[1] = (log_g_1[2] - log_g_1[1]) / (zero_spacing_factor*log_spacing);
  dlog_g_2[1] = (log_g_2[2] - log_g_2[1]) / (zero_spacing_factor*log_spacing);
  dlog_g_3[1] = (log_g_3[2] - log_g_3[1]) / (zero_spacing_factor*log_spacing);
  
  dlog_g_1[2] = (log_g_1[3] - log_g_1[1]) / ((1+zero_spacing_factor)*log_spacing);
  dlog_g_2[2] = (log_g_2[3] - log_g_2[1]) / ((1+zero_spacing_factor)*log_spacing);
  dlog_g_3[2] = (log_g_3[3] - log_g_3[1]) / ((1+zero_spacing_factor)*log_spacing);
  
  for (i in 3:N_lig) {
    dlog_g_1[i] = (log_g_1[i+1] - log_g_1[i-1]) / (2*log_spacing);
    dlog_g_2[i] = (log_g_2[i+1] - log_g_2[i-1]) / (2*log_spacing);
    dlog_g_3[i] = (log_g_3[i+1] - log_g_3[i-1]) / (2*log_spacing);
  
  dlog_g_1[N_lig+1] = (log_g_1[N_lig+1] - log_g_1[N_lig]) / (log_spacing);
  dlog_g_2[N_lig+1] = (log_g_2[N_lig+1] - log_g_2[N_lig]) / (log_spacing);
  dlog_g_3[N_lig+1] = (log_g_3[N_lig+1] - log_g_3[N_lig]) / (log_spacing);
  }
}
