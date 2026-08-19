// 
//

data {
  int<lower=1> N_antibiotic;  // number of non-zero antibiotic concentrations
  int<lower=1> N;             // total number of data points across all non-zero antibiotic concentrations
  array[N_antibiotic] int s;        // array of the number of data points for each non-zero antibiotic concentration
  
  vector[N] x;           // ligand concentrations across all non-zero antibiotic concentrations
  vector[N] y;           // normalized fitness difference datapoints across all non-zero antibiotic concentrations
  vector[N] y_err;       // estimated error of y
  
  real log_g_min;                    // lower bound on log_g
  real log_g_max;                    // upper bound on log_g
  
  // prior means (mu) and standard deviations for fitness calibration parameters:
  vector[N_antibiotic] low_fitness_mu;       // fitness difference at zero function
  vector[N_antibiotic] log_mid_g_mu;             // function level at 1/2 max fitness difference
  vector[N_antibiotic] fitness_n_mu;         // cooperativity coefficient of fitness calibration curve
  
  vector[N_antibiotic] low_fitness_std;      // fitness difference at zero function
  vector[N_antibiotic] log_mid_g_std;            // function level at 1/2 max fitness difference
  vector[N_antibiotic] fitness_n_std;        // cooperativity coefficient of fitness calibration curve
  
  vector[1] log_x_zero;    // log10 of the x value to use in place of x=0 for the GP model, vector is used instead of real for consistency with multi-ligand models
  
}

transformed data {
  array[N] real x_gp;
  
  for (i in 1:N) {
    if (x[i]==0) {
      x_gp[i] = log_x_zero[1];
	} else {
	  x_gp[i] = log10(x[i]);
	}
  }
  
}

parameters {
  // GP params:
  real<lower=0> rho;
  real<lower=0> alpha;
  vector[N] eta;
  
  real<lower=0> sigma;            // scale factor for standard deviation of noise in y
  
  vector[N_antibiotic] low_fitness;       // fitness difference at zero function
  vector[N_antibiotic] log_mid_g;         // log10 of gene expression level at 1/2 max fitness difference
  vector[N_antibiotic] fitness_n;         // cooperativity coefficient of fitness calibration curve
}

transformed parameters {
  vector[N_antibiotic] mid_g;
  
  vector[N] log_g;         // the GP function, analogous to y in all of the Stan examples
  vector[N] constr_log_g;  // log10 gene expression, constrained to be between 1 and 4
  
  vector[N] g;
  vector[N] mean_y;
  
  mid_g = 10^log_mid_g;
  
  {
    int pos;
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
    // g is the dose-response curve:
	g = 10^constr_log_g;
	
    // y is the fitness curve derived from the dose-response:
    pos = 1;
    for (k in 1:N_antibiotic) {
      for (p in pos:pos+s[k]-1) {
        mean_y[p] = low_fitness[k] - low_fitness[k]*(g[p]^fitness_n[k])/(mid_g[k]^fitness_n[k] + g[p]^fitness_n[k]);
      }
      //mean_y[pos:pos+s[k]-1] = low_fitness[k] - low_fitness[k]*(g[pos:pos+s[k]-1]^fitness_n[k])/(mid_g[k]^fitness_n[k] + g[pos:pos+s[k]-1]^fitness_n[k]);
      
      pos = pos + s[k];
    }
  }
  
}

model {
  // Priors on fitness calibration curve parameters:
  low_fitness ~ normal(low_fitness_mu, low_fitness_std);
  log_mid_g ~ normal(log_mid_g_mu, log_mid_g_std);
  fitness_n ~ normal(fitness_n_mu, fitness_n_std);

  // Priors on GP:
  rho ~ inv_gamma(5, 5);
  alpha ~ normal(1, 1);
  eta ~ std_normal();
  
  // prior noise scale, prior to keep it from getting too much < 1
  sigma ~ inv_gamma(3, 6);
  
  y ~ normal(mean_y, sigma*y_err);
  
}

generated quantities {
  real rms_resid;
  real log_rho;
  real log_alpha;
  real log_sigma;
  vector[N] dlog_g;      // derivative of the gp
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
