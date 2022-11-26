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
  
  real log_g_min;        // lower bound on log_low_level and log_high_level
  real log_g_max;        // upper bound on log_low_level and log_high_level
  
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
  vector[2] x_gp[2*N_lig+1];   // array of 2D coordinates of the log(ligand concentrations)
  real sqrt_pi = sqrt(pi());
  real low_constr;
  real high_constr;
  real sig_constr;
  real center_log_g;
  vector[2] log_x_zero;
  
  low_constr = log_g_min + 0.5;
  high_constr = log_g_max - 0.5;
  sig_constr = 0.9;
  
  center_log_g = (log_g_min + log_g_max)/2;
  
  // this sets the zero-ligand coordinates on the log(x) scale at a distance of 1.5x the dilution spacing of the non=zero-ligands 
  log_x_zero[1] = log10(x_1[1]) - 1.5*(log10(x_1[2]) - log10(x_1[1]));
  log_x_zero[2] = log10(x_2[1]) - 1.5*(log10(x_2[2]) - log10(x_2[1]));
  
  x_gp[1][1] = log_x_zero[1];
  x_gp[1][2] = log_x_zero[2];
  
  for (i in 1:N_lig){
    x_gp[i+1][1] = log10(x_1[i]);
    x_gp[i+1][2] = log_x_zero[2];
	
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
  vector[N] mean_y;
  vector[N] g;             // GP approx to gene expression
  vector[N] log_g;         // the GP function, analogous to y in all of the Stan examples
  vector[N] constr_log_g;  // log10 gene expression, constrained to be between 1 and 4

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

    log_g = center_log_g + L_K * eta;
    for (i in 1:N){
      //constr_log_g[i] = 1.5*erf(sqrt_pi/3*(log_g[i] - 2.5)) + 2.5;
      
      term1 = sig_constr/sqrt_pi*(exp(-((low_constr - log_g[i])^2/sig_constr^2)) - exp(-((high_constr - log_g[i])^2/sig_constr^2)));
      term2 = (log_g[i] - high_constr)*erf((high_constr - log_g[i])/sig_constr);
      term3 = (low_constr - log_g[i])*erf((low_constr - log_g[i])/sig_constr);
      
      constr_log_g[i] = (high_constr + low_constr + term1 + term2 + term3)/2;
      
      g[i] = 10^constr_log_g[i];
    }

  }
  
  for (i in 1:N) {
    mean_y[i] = low_fitness_high_tet - low_fitness_high_tet*(g[i]^fitness_n_high_tet)/(mid_g_high_tet^fitness_n_high_tet + g[i]^fitness_n_high_tet);
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

  // observations
  y ~ normal(mean_y, sigma*y_err);

}

generated quantities {
  real rms_resid;
  real log_rho;
  real log_alpha;
  real log_sigma;
  vector[N] dlog_g; // derivative of the gp
  
  rms_resid = distance(y, mean_y)/sqrt(N);
  
  log_rho = log10(rho);
  log_alpha = log10(alpha);
  log_sigma = log10(sigma);

  // derivative calculation
  {
    matrix[N, N] dK;
    matrix[N, N] ddK;
    vector[N] df_pred_mu;
    vector[N] K_div_f;
    matrix[N, N] cov_df_pred;
    matrix[N, N] nug_pred;
    matrix[N, N] v_pred;
    real lsInv = 1./rho/rho;
    real diff;
    matrix[N, N] L_K;
    matrix[N, N] K = cov_exp_quad(x_gp, alpha, rho);

    nug_pred = diag_matrix(rep_vector(1e-8, N));
    // diagonal elements
    for (n in 1:N)
      K[n, n] = K[n, n] + 1e-9;

    L_K = cholesky_decompose(K);

    dK = cov_exp_quad(x_gp, alpha, rho);
    ddK = cov_exp_quad(x_gp, alpha, rho);
    for (i in 1:N){
      for (j in 1:N){
        diff = x_gp[i] - x_gp[j];
        dK[i,j] = dK[i,j] * (-lsInv * diff);
        ddK[i,j] = ddK[i,j] * (1.-lsInv*diff*diff) * lsInv;
      }
    }

    K_div_f = mdivide_left_tri_low(L_K, constr_log_g);
    K_div_f = mdivide_right_tri_low(K_div_f', L_K)';

    df_pred_mu = (dK * K_div_f);

    v_pred = mdivide_left_tri_low(L_K, dK');
    cov_df_pred = ddK - v_pred' * v_pred;

    dlog_g = multi_normal_rng(df_pred_mu, cov_df_pred + nug_pred);

  }
}
