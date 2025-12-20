//
//

data {
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // fitness difference at each concentration
  vector[N] y_err;       // estimated error of fitness difference at each concentration
  real log_g_min;        // lower bound on log_low_level and log_high_level
  real log_g_max;        // upper bound on log_low_level and log_high_level
  
  real low_fitness_mu;      // fitness difference at zero gene expression
  real mid_g_mu;            // gene expression level at 1/2 max fitness difference
  real fitness_n_mu;        // cooperativity coefficient of fitness difference curve
  
}

transformed data {
  real x_gp[N];
  real sqrt_pi = sqrt(pi());
  real center_log_g;
  
  center_log_g = (log_g_min + log_g_max)/2;
  
  for (i in 1:N){
    x_gp[i] = log10(x[i] + 0.25);
  }
}

parameters {
  real<lower=0> sigma;            // scale factor for standard deviation of noise in fitness differnce, y
  
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

    log_g = L_K * eta;
	
    constr_log_g = log_g_min + (log_g_max - log_g_min)*inv_logit(log_g);

  }
  
  for (i in 1:N) {
    g[i] = 10^constr_log_g[i];
    
    mean_y[i] = low_fitness_high_tet - low_fitness_high_tet*(g[i]^fitness_n_high_tet)/(mid_g_high_tet^fitness_n_high_tet + g[i]^fitness_n_high_tet);
  }

  
}

model {
  real neg_low_fitness;
  
  neg_low_fitness = -1*low_fitness_high_tet;
  
  // fitness calibration params
  //low_fitness_high_tet ~ student_t(8, low_fitness_mu, 0.1);
  neg_low_fitness ~ exp_mod_normal(-1*low_fitness_mu-0.04, 0.03, 14); 
  
  
  //mid_g_high_tet ~ normal(mid_g_mu, 27); // use with PTY1
  //fitness_n_high_tet ~ normal(fitness_n_mu, 0.22); // use with pTY1
  mid_g_high_tet ~ normal(mid_g_mu, 499); // use with pVER
  fitness_n_high_tet ~ normal(fitness_n_mu, 0.29); // use with pVER
  
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
  vector[N] log_g_ratio; // log-g-ratio from the gp
  
  rms_resid = distance(y, mean_y)/sqrt(N);
  
  log_rho = log10(rho);
  log_alpha = log10(alpha);
  log_sigma = log10(sigma);
  
  log_g_ratio = constr_log_g - constr_log_g[1];

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
