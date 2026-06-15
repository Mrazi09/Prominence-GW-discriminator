#!/usr/bin/env pYthon
# coding: utf-8

import numpy as np
import emcee
from Aux_functions import *
from Likelihoods import *
from Input_card import *
import os
import shutil
from threading import Lock
from scipy import optimize
import sys
import time
import math
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.interpolate import interp1d
import pandas as pd

# LISA Obs time and frequency range
T_years       = 4

Data_LISA     = pd.read_csv('./GW_Data/LISA.csv',names = ['fpeak', 'Ompeak'],sep=';',decimal=',')
Data_LISA     = Data_LISA.loc[Data_LISA['Ompeak'] < 1e-6]

f_LISA        = np.geomspace(Data_LISA['fpeak'].min(), Data_LISA['fpeak'].max(), 5_000)
Spectrum_LISA = interp1d(np.array(Data_LISA['fpeak']), np.log10(np.array(Data_LISA['Ompeak'])), kind = 'cubic')
Om_LISA       = np.array([Spectrum_LISA(i) for i in f_LISA])

# Base signal
gstar = 106.75
vw    = 1
betaH = 100
alpha = 4.56e-2
Tp    = 1889.61
kSW   = 1.0
kBC   = 1e-10
kMHD  = 1e-10
args_PT1       = [gstar, vw, betaH, alpha, Tp, kSW, kBC, kMHD]
hOm_SignalPT   = hOm_FOPT(f_LISA, args_PT1)
hOm_back       = hOm_ExtGW(f_LISA,[-12.38, 0.66]) + hOm_GalGW(f_LISA,[-7.85,7/3])
hOm_total_FOPT = hOm_back + hOm_SignalPT

def Log_signal(x,burn_index):
    LTotal = 0

    x = 10**x
    βH, α, Tp, log10_h2Ext, alphaExt, log10_h2Gal, alphaGal = x
    log10_h2Ext = - log10_h2Ext
    log10_h2Gal = - log10_h2Gal

    if βH < 1:
        return -np.inf
        
    if log10_h2Ext <= -12.38-0.17:
        return -np.inf
    if log10_h2Ext >= -12.38+0.17:
        return np.inf
        
    if alphaExt <= 0.66-0.34:
        return -np.inf
    if alphaExt >= 0.66+0.34:
        return np.inf
        
    if log10_h2Gal <= -7.85-0.21:
        return -np.inf
    if log10_h2Gal >= -7.85+0.21:
        return np.inf
        
    if alphaGal <= (7/3)-0.055:
        return -np.inf
    if alphaGal >= (7/3)+0.015:
        return np.inf
    
    # Varying signal (FOPT)
    args_PT1      = [106.75, 1.0, βH, α, Tp, 1, 0, 1e-10]
    hOm_SignalPT  = hOm_FOPT(f_LISA, args_PT1)
    hOm_back      = hOm_ExtGW(f_LISA,[log10_h2Ext, alphaExt]) + hOm_GalGW(f_LISA,[log10_h2Gal,alphaGal])  
    hOm_total_var = hOm_back + hOm_SignalPT   

    # Prepare signals
    Spec_FOPT = interp1d(f_LISA, np.log10(hOm_total_FOPT), kind = 'cubic')  
    Spec_DW   = interp1d(f_LISA, np.log10(hOm_total_var), kind = 'cubic')  
    Spec_SKA  = interp1d(np.array(Data_LISA['fpeak']), np.log10(np.array(Data_LISA['Ompeak'])), kind = 'cubic')
    
    # Likelihood based on the SNR of signals
    # h1/h1
    OmegaGW_int  = 10**Spec_FOPT(f_LISA)
    OmegaGW_sens = 10**Spec_SKA(f_LISA)
    Integrand    = simpson(pow((OmegaGW_int/OmegaGW_sens),2), x=f_LISA)
    rhosq_FOPT   = T_years*Integrand
    
    # h2/h2
    OmegaGW_int  = 10**Spec_DW(f_LISA)
    OmegaGW_sens = 10**Spec_SKA(f_LISA)
    Integrand    = simpson(pow((OmegaGW_int/OmegaGW_sens),2), x=f_LISA)
    rhosq_DW     = T_years*Integrand
    
    # h1/h2
    OmegaGW_int1 = 10**Spec_DW(f_LISA)
    OmegaGW_int2 = 10**Spec_FOPT(f_LISA)
    OmegaGW_sens = 10**Spec_SKA(f_LISA)
    Integrand    = simpson(((OmegaGW_int1*OmegaGW_int2)/pow(OmegaGW_sens,2)), x=f_LISA)
    rhosq_MIX    = T_years*Integrand
    
    # Overlap
    O_12 = rhosq_MIX/(np.sqrt(rhosq_FOPT*rhosq_DW))
    
    # Chi2
    chi2 = (rhosq_FOPT + rhosq_DW - 2*np.sqrt(rhosq_FOPT)*np.sqrt(rhosq_DW)*O_12)

    # Likelihood 
    LTotal = -2*chi2
    
    if LTotal != LTotal:
        return -np.inf
        
    if burn_index == 11:
        
        # Limit the arrays to the point where the signals intersect the sensitivity curve
        f_indices_FOPT = np.argwhere(np.diff(np.sign(np.log10(hOm_total_var) - Om_LISA))).flatten() 
        hOm_total_var  = hOm_total_var[f_indices_FOPT[0]:f_indices_FOPT[1]]

        # Get prominences (first signal)
        peaks1, _      = find_peaks(np.log10(hOm_total_var))                  # Find the peaks
        prominences1   = peak_prominences(np.log10(hOm_total_var), peaks1)[0] # Compute prominences. First element is the prominence, second/third the left/right bases
        prominences1   = -np.sort(-prominences1)                              # sort prominences (first largest and second smallest)    
    
        
        df = pd.DataFrame({'MCMC Likelihood':[LTotal], 'βH':[βH], 'α':[α], 'Tp':[Tp], 'E':[6.83104864410821e+26],
                           'Vbias':[1.942575876544e+19],'log10_h2Ext':[log10_h2Ext], 'alphaExt':[alphaExt],
                           'log10_h2Gal':[log10_h2Gal], 'alphaGal':[alphaGal], 'b':[7/3],'c':[1.65],
                           'prom':[prominences1], 'SNR':[np.sqrt(rhosq_DW)]})
        
        df.to_csv('./{}/LISA_FOPT.csv'.format(Data_p), sep=',', index=False, mode='a', header=False)
        
    return LTotal

#Alter a variable inside in the moves argument in EnsembleSampler
def run_MCMC(p0, lnprob, ndim, nwalkers, nburnin, nrun, h5name, scale, reset=True, pool_obj=None, burn_in=True):
    burn_index = 10
    initial = [np.array(p0) + 1e-2* np.random.randn(ndim) for i in range(nwalkers)]         
    
    backend = emcee.backends.HDFBackend(h5name)
    if reset:
        backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, pool=pool_obj, a = scale, args=[burn_index])

    if burn_in:
        print("Running burn-in...")
        print(len(initial))
        initial, _ , _  = sampler.run_mcmc(initial, nburnin, progress=True, store=False)
        sampler.reset()
    print("Running production...")
    burn_index = 11
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, pool=pool_obj, a = scale, args=[burn_index])
    pos, prob, state  = sampler.run_mcmc(initial, nrun, progress=True, store=False)

    return sampler





    
