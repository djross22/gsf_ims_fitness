
//data {
  
  // Input data variables for multi-operator model priors
  real<lower=2> copy_num_prior_mean;  // geometric mean for prior on plasmid/operator copy number
  real<lower=0> copy_num_prior_width; // geometric std for prior on plasmid/operator copy number
  real<lower=1> R_prior_mean;         // geometric mean for prior on repressor dimer copy number
  real<lower=0> log_R_prior_width;    // geometric std for prior on repressor dimer copy number
  
//}

