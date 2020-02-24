# About this repository
This repository contains the code and the data for the paper "Combining interdependent climate model outputs in CMIP5: A spatial Bayesian approach".

# Install relevant software and packages

The code is in Python, and the Jupyter notebook is used to analyze the data. Users can manually install the Jupyter notebook, Python, and the following packages.
  * Cython
  * jupyterlab
  * matplotlib
  * numpy
  * python
  * scipy

However, an easier way is to use conda, which is highly recommended. See https://docs.conda.io/en/latest/miniconda.html for conda installation. Once conda is available, run the following commands in a Terminal, then all the required software and packages will be automatically downloaded and installed.

```bash
conda env create --prefix ./env --file environment.yml
source activate ./env
jupyter lab
```

# How to run the python code
1. Some computationally-intensive routines are written in .pyx files using BLAS interfaces directly and multiprocessing, which need to be compiled to C files. 

    1.1. Open a Terminal. Enter the directory "src".

    1.2. Open the file "setup.py". Change the values of "extra_compile_args" and "extra_link_args" according to the operation system. Currently the flag "-fopenmp" is used for OpemMP multiprocessing parallism.

    1.3. In the command shell, type "cythonize -i *.pyx".
    
2. Enter the directory "CMIP5".
    
    2.1. "**runApplicationMCMC.ipynb**" is the python jupyter notebook for running the MCMC with the CMIP5 data. User can choose the scenarios in the first code chunk.
    * **region**: either "CNA" or "EAS"
    * **RCP**: either "RCP45" or "RCP85"
    * **model**: either "Full" or "SSB". "Full" is our model and "SSB" is the model proposed by Sansom, P. G., D. B. Stephenson, and T. J. Bracegirdle (2017). *On constraining projections of future climate using observations and simulations from multiple climate models.* 
    * **nSave**, results are saved after multiplications of this number of iterations is reached
    * **nThin**, the number of thinning, i.e., only one iteration is saved for every nThin iterations
    * **nChain**, the number of saved MCMC iterations 
    * **nBurn**, the number of iterations for burn-in
    
    2.2. "**showResultsInCNA.ipynb**" and "**showResultsInEAS.ipynb**" are notebooks for analyzing the MCMC inference results. If the user does not want to wait for a new execution of MCMC, step 2.1 can be skipped, and some precomputed results are stored in the directory "results". These two notebooks draw figures shown in Section 4.1 in the paper.
    
    2.3. "**showSummary.ipynb**" is the notebook summarizing the MCMC results and draw figures shown in Sections 4.2 and 4.3 in the paper.