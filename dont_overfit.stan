functions {
  
}

data {
  
  // Define the training set
  int <lower = 1> N;
  int <lower = 1> n_pred;
  
  int y [N];
  matrix [N, n_pred] X;
  
  // Define the test set
  int <lower = 1> N_pp;
  matrix [N_pp, n_pred] X_pp;
  
  // Specify prior parameters outside of Stan to allow for iterative approach
  real prior_mu;
  real <lower = 0> prior_sigma;

}

transformed data {
  
}

parameters {
  
  // Co-efficients on the log-odds scale
  vector [n_pred] beta;
  
}

transformed parameters{
  
}

model {

// Likelihood model : Logistic regression
target += bernoulli_logit_lpmf(y | X * beta);

// Prior model: Consistent for all co-efficients on log-odds scle
target += normal_lpdf(beta | prior_mu, prior_sigma);
  
}

generated quantities {
  
  // Generate log likelihoods for option of WAIC or LOO-CV evaluation
  vector [N] log_lik;
  
  // Use model to sample from posterior predictive distribution, using test set
  vector [N_pp] post_pred_y;

  for (n in 1:N){

    log_lik[n] = bernoulli_logit_lpmf(y[n] | X[n] * beta);

  }
  
  for (n_pp in 1:N_pp){
    
    post_pred_y[n_pp] = inv_logit(X_pp[n_pp] * beta);
    
  }
  
}