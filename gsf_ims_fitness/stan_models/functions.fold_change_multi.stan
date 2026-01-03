
//functions {

  real fold_change_fnct(real x_in, real k_a, real k_i, real eps_ai, real eps_ra, real h_n, real n_ns, real r, real n_s){
    real c1;
    real c2;
    real pA;
	real xRA;
	real lam;
	
    c1 = (1 + x_in/k_a)^h_n;
    c2 = ( (1 + x_in/k_i)^h_n ) * exp(-eps_ai);
	pA = c1/(c1+c2);
	xRA = exp(-eps_ra);
	
	lam = -n_ns + pA*r - n_s*xRA + pA*r*xRA;
    lam = lam + sqrt(4*pA*r*xRA*(n_ns + n_s - pA*r) + (n_ns + n_s*xRA - pA*r*(1 + xRA))^2);
    lam = lam/(2*xRA*(n_ns + n_s - pA*r));
	
    return 1/(1 + lam*xRA);
  }
  
//}

