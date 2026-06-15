#!/usr/bin/env python
# coding: utf-8
import numpy as np
import random
import os
import os.path
import sys
import pandas as pd
import emcee
from multiprocessing import Pool
import math
import tarfile

import time
import scipy
from scipy.integrate import simpson
from scipy.stats import unitary_group
from scipy.stats import ortho_group
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

from Aux_functions import *
from Solve_LISA import *
from Input_card import *
from Likelihoods import *

#printing for arrays. To offer a better visualization in the notebook
np.set_printoptions(linewidth=np.inf)

#Surpress warnings. Not recommended. Only use after code has been properly debugged
import warnings
warnings.filterwarnings("ignore")
import ast

os.popen('cp ./Data_header/LISA_FOPT.csv ./{}'.format(Data_p)) 

βH          = np.log10( 100       )
α           = np.log10( 4.56e-2   )
Tp          = np.log10( 1889.61   )
log10_h2Ext = np.log10( 12.38     )
alphaExt    = np.log10( 0.66      )
log10_h2Gal = np.log10( 7.85      )
alphaGal    = np.log10( 7/3       )

xMC = [βH] + [α] + [Tp] + [log10_h2Ext] + [alphaExt] + [log10_h2Gal] + [alphaGal]
Dim=len(xMC)

start = "_"
end = "_"
print("============================================================")
print("========== BEGGINING MCMC RUN --> MODEL: CONFORMAL =========")
print("============================================================")
print("")
print("Initializing MCMC run")
print("The following options are considered:")
print("")
print("---> Parallelisation option: {}".format(run_Pool))
print("---> Burn in: {}".format(nBurn))
print("---> Compress .h5 files: {}".format(tar_data))
if nBurn == True:
    print("---> Number of iterations for burn: {}".format(itBurn))
print("---> Number of interations per MC run: {}".format(nMCMC))
print("---> Number of MC batch runs: {}".format(nBatches))
print("")
print("The following paths are considered:")
print("")
print("Path to the run folder: {}".format(Path))
print("Path to the folder where the data is saved {}".format(Data_p))
print("Code will begin in 20 seconds. If the options are wrong, please")
print("press CTRL+C to stop the code")
print("")
time.sleep(1)

for i in range(0,nBatches):
    
    if i == 0:
        if (run_Pool == True and nBurn == True):
            with Pool() as pool:
                sampler = run_MCMC(p0 = xMC, lnprob = Log_signal, ndim = Dim, nwalkers = Dim*2, nburnin = itBurn, nrun=nMCMC,
                                   h5name = '{D}results_{it}.h5'.format(D=Data_p, it=i), 
                                   reset=True, burn_in=True, pool_obj=pool, scale = 10.0)

        if (run_Pool == True and nBurn == False):
            with Pool() as pool:
                sampler = run_MCMC(p0 = xMC, lnprob = Log_signal, ndim = Dim, nwalkers = Dim*2, nrun=nMCMC, nburnin = 0,
                                   h5name = '{D}results_{it}.h5'.format(D=Data_p, it=i), 
                                   reset=True, burn_in=False, pool_obj=pool, scale = 10.0)

        if (run_Pool == False and nBurn == True):
            sampler = run_MCMC(p0 = xMC, lnprob = Log_signal, ndim = Dim, nwalkers = Dim*2, nburnin = itBurn, nrun=nMCMC,
                                   h5name = '{D}results_{it}.h5'.format(D=Data_p, it=i),
                                   reset=True,  burn_in=True, scale = 10.0)

        if (run_Pool == False and nBurn == False):
            sampler = run_MCMC(p0 = xMC, lnprob = Log_signal, ndim = Dim, nwalkers = Dim*2, nrun=nMCMC, nburnin = 0,
                                   h5name = '{D}results_{it}.h5'.format(D=Data_p, it=i),
                                   reset=True, burn_in=False, scale = 10.0)

        if tar_data == True:
            make_tarfile("{D}results_{it}.tar.gz".format(D=Data_p, it=i), "{D}results_{it}.h5".format(D=Data_p, it=i))
                
    if i > 0:
        reader = emcee.backends.HDFBackend('{D}results_{it}.h5'.format(D=Data_p, it=i-1))
        samples = reader.get_chain(flat=True)
        blobs = reader.get_blobs(flat=True)
        flatlnprobability = reader.get_log_prob(flat=True)
        
        p_max = samples[np.argmax(flatlnprobability)]

        print("\n")
        print("=========================================================================")
        print("In the previous batch, the best point was:")

        βH          = 10**p_max[0]
        α           = 10**p_max[1]
        log10_h2Ext = 10**p_max[3]
        alphaExt    = 10**p_max[4]
        log10_h2Gal = 10**p_max[5]
        alphaGal    = 10**p_max[6]
        
            
        print("log = {}".format(flatlnprobability.max()))
        print("")
        print("βH =", βH)
        print("α =", α)
        print("log10_h2Ext =", log10_h2Ext)
        print("alphaExt =", alphaExt)
        print("log10_h2Gal =", log10_h2Gal)
        print("alphaGal =", alphaGal)

        if tar_data == True:
            os.remove("{D}results_{it}.h5".format(D=Data_p, it=i-1))
        
        if (run_Pool == True and nBurn == True):
            with Pool() as pool:
                sampler = run_MCMC(p0 = p_max, lnprob = Log_signal, ndim = Dim, nwalkers = Dim*2, nburnin = itBurn, nrun=nMCMC,
                                   h5name = '{D}results_{it}.h5'.format(D=Data_p, it=i),
                                   reset=True, burn_in=True, pool_obj=pool, scale = 10.0)

        if (run_Pool == True and nBurn == False):
            with Pool() as pool:
                sampler = run_MCMC(p0 = p_max, lnprob = Log_signal, ndim = Dim, nwalkers = Dim*2, nrun=nMCMC, nburnin = 0,
                                   h5name = '{D}results_{it}.h5'.format(D=Data_p, it=i),
                                   reset=True, burn_in=False, pool_obj=pool, scale = 10.0)

        if (run_Pool == False and nBurn == True):
            sampler = run_MCMC(p0 = p_max, lnprob = Log_signal, ndim = Dim, nwalkers = Dim*2, nburnin = itBurn, nrun=nMCMC,
                                   h5name = '{D}results_{it}.h5'.format(D=Data_p, it=i), 
                                   reset=True, burn_in=True, scale = 10.0)

        if (run_Pool == False and nBurn == False):
            sampler = run_MCMC(p0 = p_max, lnprob = Log_signal, ndim = Dim, nwalkers = Dim*2, nrun=nMCMC, nburnin = 0,
                                   h5name = '{D}results_{it}.h5'.format(D=Data_p, it=i),
                                   reset=True, burn_in=False, scale = 10.0)        

        if tar_data == True:
            make_tarfile("{D}results_{it}.tar.gz".format(D=Data_p, it=i), "{D}results_{it}.h5".format(D=Data_p, it=i))
                
print("")
print("")
print("###########################################################")
print("###########################################################")
print("###                                                     ###")
print("### Code is finished. Thank you for choosing Vault-Tec! ###")
print("###                                                     ###")
print("###########################################################")
print("###########################################################")    

