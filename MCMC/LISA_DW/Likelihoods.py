import numpy as np

def Chi2_function(Obs, Exp, std):
    
    Chi2 = (Obs-Exp)**2 / std**2
    
    return Chi2

def Likelihood(hOm_th, hOm_exp, perc, dofs):
    
    chi2          = Chi2_function(Obs = hOm_th, Exp = hOm_exp, std = hOm_exp*perc)  
    LogLikelihood = (1/dofs) * ( np.sum((chi2)/(-2)) )
    
    if LogLikelihood != LogLikelihood:
        return -np.inf
    
    else:
        return LogLikelihood
        
