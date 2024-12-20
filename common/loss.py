import torch
import torch.nn as nn
import numpy as np

# Linear annealing schedule
def linear_annealing(epoch, total_epochs, start_beta=0.0, end_beta=1.0):
    anneal_fraction = min(epoch / float(total_epochs), 1.0)
    return start_beta + anneal_fraction * (end_beta - start_beta)

def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2))* (y-mu)**2

def elbo(y_pred, y, mu, log_var):
    # likelihood of observing y given Variational mu and sigma
    likelihood = ll_gaussian(y, mu, log_var)
    
    # prior probability of y_pred
    log_prior = ll_gaussian(y_pred, 0, torch.log(torch.tensor(1.)))
    
    # variational probability of y_pred
    log_p_q = ll_gaussian(y_pred, mu, log_var)
    
    # by taking the mean we approximate the expectation
    return (likelihood + log_prior - log_p_q).mean()
    
# Loss function for VAE
def vae_loss(x, x_reconstructed, mu, logvar): 
    reconstruction_loss = nn.MSELoss(reduction='sum')(x_reconstructed, x)

    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return reconstruction_loss, kl_divergence
