library(tidyverse)

setwd("~/Kaggle")

# Read in training data
do_train_val <- read.csv(file = 'dontOverfit/train.csv')

#########################################################
#
#     Data Pre-Processing
#
#########################################################

library(rsample)
# Generate a valiadtion set
train_val_split <- initial_split(data = do_train_val, strata = target, prop = 4/5)
do_train <- training(x = train_val_split); do_val <- testing(x = train_val_split)

# Pre-processing function that generates the necessary inputs for Stan
gen_Stan_data_list <- function(df_train, df_val, test){
  
  model_data <- model.frame(formula = target ~ ., data = df_train %>% select(-c(id)))
  data_matrix <- model.matrix(target ~ ., data = model_data)
  
  if(test == TRUE) {
    
    post_pred_matrix <- as.matrix(x = df_val %>%
                                    select(-c(id)))
    
  } else {
    
    post_pred_matrix <- as.matrix(x = df_val %>%
                                    select(-c(id, target)))
    
  }
  
  
  
  n_pred <- dim(data_matrix)[2]; N_pp <- dim(post_pred_matrix)[1]
  
  int_df <- data.frame(x = double(length = N_pp)); colnames(int_df) <- '(Intercept)'; int_df$`(Intercept)` <- 1
  formatted_post_pred_matrix <- cbind(int_df, post_pred_matrix)
  
  # Creating a Stan data list
  df_model_data <- list(N = nrow(df_train), 
                        y = df_train$target,
                        id = df_train$id, 
                        n_pred = n_pred,
                        X = data_matrix,
                        N_pp = N_pp,
                        X_pp = formatted_post_pred_matrix,
                        prior_mu = 0,
                        prior_sigma = 1/2)

  return(df_model_data)
    
}

do_model_data <- gen_Stan_data_list(df_train = do_train, 
                                    df_val = do_val, 
                                    test = FALSE)

#########################################################
#
#     Tuning hyperparameter: Prior variance on coefficients
#
#########################################################

library(rstan); library(ggmcmc); library(loo)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

n_chains <- parallel::detectCores(); n_warmup <- 5e3; n_draws <- n_warmup + 5e3

# Initialise an empty results dataframe
hyp_param_df <- data.frame(prior_sigma = double(), 
                           val_fp_rate = double(), val_fn_rate = double(), 
                           error = double())

for(s in c(0.25, 0.5, 1, 1.5, 2.0, 2.5)){
  
  do_model_data$prior_sigma <- s
  
  # Fit the model
  do_model <- stan(file = 'dontOverfit/dont_overfit.stan', data = do_model_data,
                   pars = c('post_pred_y'),
                   chains = n_chains, iter = n_draws, warmup = n_warmup, seed = 1008,
                   control = list(adapt_delta = 0.9))
  
  # Extract predictions
  predictions <- cbind(do_val %>% select(id, target),
                       (summary(do_model))$summary %>%
                         as.data.frame() %>%
                         dplyr::filter(mean >=0) %>% 
                         mutate(pred_class = round(mean)) %>% 
                         select(pred_class))
  
  # Calculate a false positive error rate
  fp_rate <- nrow(predictions %>% dplyr::filter(round(pred_class) == 1 & target == 0)) / 
     nrow(predictions %>% dplyr::filter(target == 0))
  
  # Calculate a false positive error rate 
  fn_rate <- nrow(predictions %>% dplyr::filter(round(pred_class) == 0 & target == 1)) / 
    nrow(predictions %>% dplyr::filter(target == 1))
  
  # Calculate a mean absolute error
  error <- mean(abs(predictions$target - predictions$pred_class))
  
  hyp_param_df <- rbind(hyp_param_df, 
                        data.frame(prior_sigma = s, 
                                   val_fp_rate = fp_rate, 
                                   val_fn_rate = fn_rate, 
                                   error = error))
  
  print(s)
  
}

hyp_param_df

# Save the results of the hyperparameter tuning
write.csv(x = hyp_param_df, file = 'hyp_param_df_orig.csv')

# Plot results
hyp_param_df %>% 
  tidyr::pivot_longer(cols = c(val_fp_rate, val_fn_rate, error), 
                      names_to = 'metric', values_to = 'value') %>%
  dplyr::filter(grepl(pattern = 'val_', x = metric)) %>% 
  ggplot(mapping = aes(x = log10(prior_sigma), y = value))+
    geom_line(mapping = aes(col = metric))+
    geom_point(shape = 1)+
    DomDF::theme_ddf_light()

#########################################################
#
#     Fitting final model
#
#########################################################

# Read in test set
do_test <- read.csv(file = 'dontOverfit/test.csv'); 

n_rows_test <- nrow(do_test)

# Use pre-processing function to generate Stan input data
do_model_data_final <- gen_Stan_data_list(df_train = do_train_val,
                                          test = do_test, 
                                          test = TRUE)

# Fit final model
do_model_final <- stan(file = 'dontOverfit/dont_overfit.stan', data = do_model_data_final,
                       pars = c('post_pred_y'),
                       chains = n_chains, iter = n_draws, warmup = n_warmup, seed = 1008,
                       control = list(adapt_delta = 0.9))

# Create results data frame
predictions_final <- cbind(do_test %>% select(id), 
                           ((summary(do_model_final))$summary %>%
                             as.data.frame() %>% 
                             dplyr::filter(mean >= 0) %>%
                             rename(pred_class = round(mean)) %>%
                             select(pred_class))) 

# Create Kaggle submission file  
write.csv(x = predictions_final, 
          file = 'dontOverfit/test_predictions.csv', 
          row.names = FALSE)
