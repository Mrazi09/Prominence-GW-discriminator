#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import random
import os.path
import tarfile
import glob
import pandas as pd
import emcee
from scipy.integrate import simpson

def hOm_ExtGW(f, args_model):
    
    """
    Extra-galatic background
    """

    log10_h2Ext, alpha = args_model
    
    hOmExt = 10**log10_h2Ext
    
    return hOmExt*(f/(1e-3))**(alpha)

def hOm_GalGW(f, args_model):

    """
    Galatic background
    """

    log10_h2Gal, ns = args_model
    
    Tobs   = 4
    year   = 1
    a1     = -0.15
    b1     = -2.72
    ak     = -0.37
    bk     = -2.49
    v      = 1.56
    f2     = 6.7e-4
    hOmGal = 10**(log10_h2Gal)
    f1     = 10**( a1*np.log10(Tobs/year) + b1 )
    fknee  = 10**( ak*np.log10(Tobs/year) + bk )
    hOm    = (f**3/2)*(f/1)**(-ns)*(1 + np.tanh( (fknee - f)/f2 ) )*np.exp( -(f/f1)**v )*hOmGal
    
    return hOm


def hOm_SMBHBs(f, args_model):

    """
    # Supermassive black hole binaries at PTAs
    """
    
    gw_bhb = args_model
    
    # Values used by PTArcade/Nanograv collab.
    Tspan  = 505861299.1401643
    fyr    = 3.168808781402895e-08
    H_0_Hz = 2.1842852410855023e-18
    h      = 0.674
    
    h2Omega = (2*np.pi**2*(10**gw_bhb[0])**2)/(3*H_0_Hz**2) * \
              (f/(fyr**(1)))**(5-gw_bhb[1]) * (fyr**(2))

    return h2Omega

def hOm_FOPT(f, args_model):

    gstar, ξw, βH, α, Tp, κsw, κbc, κmhd = args_model
    
    # Written based on formulas from LISA Cosmology working group (2403.03723)
    
    # -------------------------------------
    # -------- SOUND WAVE SPECTRUM --------
    # -------------------------------------

    n1_sw, n2_sw, n3_sw, a1_sw, a2_sw = 3, 1, -3, 2, 4
    Γ           = 4/3 # adiabatic index. 4/3 for radiation fluid
    cs          = 1/np.sqrt(3)
    ξshell      = np.abs(ξw - cs)
    Δw          = ξshell/max(ξw, cs)
    Hstar_Rstar = (8*np.pi)**(1/3) * max(ξw, cs) * βH**(-1)
    Trh_V       = Tp #Tp*(1 + α)**(1/4)
    H0star      = (1.65 * 10**(-5)) * (gstar/100) ** (1/6) * (Trh_V/100) # Factor to include redshifting to present day (i.e. already has the h^2 prefactor included)
    FGW0        = (1.64 * 10**(-5)) * (100/gstar)**(1/3)                 # Factor to include redshifting to present day (i.e. already has the h^2 prefactor included)
    Asw         = 0.11
    vJ          = (np.sqrt((2/3)*α + α**2) + np.sqrt(1/3))/ (1 + α)

    K           = (0.6*κsw*α)/(1 + α)
    vfsq        = Γ**(-1)*K
    Hstar_τsh   = Hstar_Rstar/np.sqrt(vfsq)
    Hstar_τstar = min(Hstar_τsh, 1)

    # Spectral shape
    def S_SW(f, args):
        n1,n2,n3,a1,a2,f1,f2 = args
        return (f/f1)**n1 * ( 1 + (f/f1)**a1)**((-n1+n2)/(a1))  * ( 1 + (f/f2)**a2 )**((-n2 + n3)/a2)
        
    f1_sw = 0.2 * H0star * (Hstar_Rstar)**(-1)
    f2_sw = 0.5 * H0star * Δw**(-1) * (Hstar_Rstar)**(-1)
    Ωint  = FGW0 * Asw * K**2 * Hstar_τstar * Hstar_Rstar 

    #Integrate spectral shape to get normalisation factor (use a better f)
    ff1 = np.geomspace(1e-10,1000,100_000)
    Sf1  = S_SW(ff1, [n1_sw,n2_sw,n3_sw,a1_sw,a2_sw,f1_sw,f2_sw] )
    lnf1 = np.log(ff1)
    N   = 1/simpson(Sf1, x=lnf1)
    
    # Sound wave amplitude and its peak
    Sf  = S_SW(f, [n1_sw,n2_sw,n3_sw,a1_sw,a2_sw,f1_sw,f2_sw] )
    ΩSW      = N * Ωint * Sf
    
    # -----------------------------------------------
    # ---------- BUBBLE COLLISION SPECTRUM ----------
    # -----------------------------------------------

    n1_bc, n2_bc, a1_bc = 2.4, -2.4, 1.2
    Astr   = 0.05

    Ktilde = κbc*(α/(1 + α))
    Ωp     = FGW0 * Astr * Ktilde**2 * βH**(-2)
    fp     = 0.11 * H0star * βH
    
    # Spectral shape
    def S_BC(f, args):
        n1,n2,a1,fpeak = args
        den  = (n1 - n2)**((n1-n2)/a1)
        num  = ( -n2*(f/fpeak)**(-n1*a1/(n1-n2)) + n1*(f/fpeak)**(-n2*a1/(n1-n2)) )**((n1-n2)/a1)
        return den/num
    
    # Bubble collision amplitude and its peak
    ΩBC      = Ωp * S_BC(f, [n1_bc, n2_bc, a1_bc, fp])
    
    # -----------------------------------------------
    # ------------- TURBULENCE SPECTRUM -------------
    # -----------------------------------------------
    
    n1_mhd, n2_mhd, n3_mhd, a1_mhd, a2_mhd = 3, 1, -8/3, 4, 2.15
    Nn = 2.
    A  = 0.085
    Ωs = κmhd*K # κmhd corresponds to ε in 2403.03723
    vA = (3/4)*Ωs
    f1 = (np.sqrt((3/4)*Ωs))/(2*Nn) * H0star * (Hstar_Rstar)**(-1)
    f2 = 2.2 * H0star * (Hstar_Rstar)**(-1)
    
    # Spectral shape
    def S_MHD(f, args):
        n1,n2,n3,a1,a2,f1,f2 = args
        return (f/f1)**n1 * ( 1 + (f/f1)**a1)**((-n1+n2)/(a1))  * ( 1 + (f/f2)**a2 )**((-n2 + n3)/a2)
    
    #Integrate spectral shape to get normalisation factor
    ff1  = np.geomspace(1e-10,1000,100_000)  # Should be between -inf to +inf. Putting big/small numbers is numerically, the same thing
    Sf1  = S_MHD(ff1, [n1_mhd,n2_mhd,n3_mhd,a1_mhd,a2_mhd,f1,f2])
    lnf1 = np.log(ff1)
    N    = 1/simpson(Sf1, x=lnf1)
    
    # Magnetohydrodinamics contribution
    ΩMHD = N * ((3 * A * vA * Ωs**2 * Hstar_Rstar**2)/(4 * np.pi**2 * Nn)) * FGW0 * Sf
    
    # Get peak amplitude and frequency from full spectra
    hOm_Total = ΩSW + ΩBC + ΩMHD
    
    return hOm_Total

def hOm_DomainWall(f, args_model):

    E, Vbias, b, c = args_model
    
    a         = 3
    eGW       = 0.7 # 0.7±0.4
    gstar     = 106.75 # SM High-T DOFs
    A         = 0.8 # 0.8 ± 0.1/ area parameter
    Cann      = 5 # Assuming Z2 breaking
    hOm_peak  = 1.49e-10 * (eGW/0.7) * (A/0.8)**4 * (10.75/gstar)**(1/3) * (E**(1/3)/1e7)**(12) * (1e7/Vbias)**2
    f_peak    = 5.93e-6 * (5/Cann)**(1/2)*(0.8/A)**(1/2)*(1e7/E**(1/3))**(3/2)*(Vbias/1e7)**(1/2)
    S         = lambda x: ((a+b)**c)/(b*x**(-a/c) + a*x**(b/c))**c
    
    hOm_Total = hOm_peak*S(f/f_peak)
    
    return hOm_Total

def hOm_PTA(f, gw_bhb):

    Tspan  = 505861299.1401643
    fyr    = 3.168808781402895e-08
    H_0_Hz = 2.1842852410855023e-18
    h      = 0.674
    
    h2Omega = (2*np.pi**2*(10**gw_bhb[0])**2)/(3*H_0_Hz**2) * \
              (f/(fyr**(1)))**(5-gw_bhb[1]) * (fyr**(2))

    return h2Omega



    
