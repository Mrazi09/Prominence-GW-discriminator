################################################
################################################
#                                              #
#                  INPUT CARD                  #
#                                              #
################################################
################################################


# ========== MCMC run options ==========

run_Pool = True      #Activate parallelization. Recommended to be true, unless you like your scripts to run slower
nBurn    = True      #Run burn. Recommended, especially when running multiply batches.
itBurn   = 10        #Choose number of iterations for burn in
tar_data = True      #Tar .h5 files. STRONGLY recommended to be set to true. .h5 files occupy a lot of space
nMCMC    = 2_000_000 #Number of iterations per Monte Carlo run
nBatches = 1         #Batches of MCMC runs (Note: The total number of iterations is nMCMC*nBatches)

# ========== Relevant paths ==========

Path    =  "./"       #Main path. Where the various python codes are located
Data_p  =  "./Data/" #Data path. Where the data is saved

