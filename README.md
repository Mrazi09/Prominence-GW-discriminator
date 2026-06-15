# Prominence: GW discriminator
Auxiliary code for the paper: "*Prominence*: A discriminator of gravitational wave signals" by João Gonçalves, Danny Marfatia and António P. Morais. The original paper is available at [arXiv:2509.04384.](https://arxiv.org/abs/2509.04384)

**GW_Data**: ```.csv``` files containing the sensitivity curves of different GW experiments. For NANOGrav, the periodogram dataset is also included;

**Plots**: Folder where plots are saved (plots are generated using the provided ```jupyter``` notebook;

**Signal_Data**: ```.csv``` files containing example signals (2 for DWs and 2 for FOPTs).  

**prominence-sens.ipynb**: ```jupyter``` notebook for caculating the prominence PDFs and p-values.  As an example, the calculation is done for Figure 5 of the paper.

**MCMC**: Contains scripts implementing the Markov Chain Monte Carlo algorithm using the ```emcee``` package. These scripts are designed to minimize the $\chi^2$ statistic defined in Section 4.4 of the paper.
