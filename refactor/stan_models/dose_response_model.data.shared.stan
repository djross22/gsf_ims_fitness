
//data {

  // Input data variables that are shared across all models 
  int<lower=1> N;        // number of data points
  vector[N] x;           // inducer concentration
  vector[N] y;           // gene expression (from cytometry) at each concentration
  vector[N] y_err;       // estimated error of gene expression at each concentration
  
  int rep[N];            // integer to indicate the measurement replicate
  int<lower=1> num_reps; // number of measurement replicates (for all variants)
  
  int<lower=1> N_contr;          // number of data points for control strains LacI deletions
  vector[N_contr] y_contr;       // gene expression (from cytometry) for control strains
  vector[N_contr] y_contr_err;   // estimated error of gene expression for control strains
  int rep_contr[N_contr];        // integer to indicate the measurement replicate for controls
  int<lower=1> num_contr_reps;   // number of measurement replicates for controls
  
  real y_max;             // geometric mean for prior on maximum gene expression value
  real g_max_prior_width; // geometric std for prior on maximum gene expression value
  
  int<lower=2> num_var;  // number of variants
  int variant[N];        // numerical index to indicate variants
  
  int<lower=1> num_mut;  // number of differrent mutations
  int<lower=0, upper=1> mut_code[num_var-1, num_mut];   // one-hot encoding for  presence of mutations in each variant; variant 0 (WT) has no mutations
  
  real rep_ratio_scale;   // parameter to set the scale for the half-normal prior on log_rep_ratio
  
  // prior for non-fluorescent control level
  real g_min_prior_mu;
  real g_min_prior_std;
  
  real rep_offset_scale;  // hyperparameter parameter to set the scale for the half-normal prior on offset_sigma
  
//}

