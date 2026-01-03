
//functions {

  real fold_change_fnct(real x_in, real k_a, real k_i, real eps_ai, real eps_ra, real h_n, real n_ns, real r, real n_s){
    real c1;
    real c2;
    real c3;
	
    c1 = (1 + x_in/k_a)^h_n;
    c2 = ( (1 + x_in/k_i)^h_n ) * exp(-eps_ai);
    c3 = r/n_ns * exp(-eps_ra);
	
	return 1/(1 + (c1/(c1+c2))*c3);
  }
  
//}

