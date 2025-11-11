# INDITEK-2.0

TThis is the second version of INDITEK model, a global model of diversification (#genera My^-1) of marine invertebrates in the Phanerozoic (from 541 Ma to present).

This version focuses on diversification on the continental platform, where most diversity emerges, in order to speed up the model to simulations of ~30 seconds.
The model speed is key to allow applying a Bayesian Markov chain Monte Carlo (MCMC) framework. With the new MCMC module, we can run thousands of simulations in multiple chains to infer the probability of the model parameter values. 
This is done according to the model fit to present day global observations.
The aim of this upgraded version is to be able to explore the probability of different hypotheses explaining diversification in the deep time.

This work was funded by national research grant PID2023-152076NB-I00  (INDICIOS project) from the Spanish government.
The model is further explained in the publication: INDITEK-2.0: A Bayesian inverse eco-evolutionary modelling framework for reconstructing Phanerozoic biodiversity; https://doi.org/10.1101/2025.08.22.671786

This version is written in Python 3.14.

# How to run the model:

The main module of INDITEK 2.0 is **indicios_7param.py** which estimates probabilistically the parameters of the model according to a Metropolis-Hastings (M-H) MCMC algorithm. 
In indicios_7param.py you can modify the priors: range of tolerance for the proposed parameter values in the M-H iterations (out of these bounds we reject the proposal), and the mean and standard deviation of the prior parameter distributions. 
This priors are inferred from previous knowledge (e.g., literature). You can also change the extinction pattern to apply and compare the proof of concept to the output.


To run the model, you need to have in the same folder the functions (.py) and the folder data. The folder data contains:

- Point_ages_xyz.mat: floor age data from the plate-tectonic/paleo-elevation model.
- Point_foodtemp.mat: food-temp data from the cGenie earth-system model.
- landShelfOceanMask.mat: 0-2 mask to distinguish land-shelf-ocean grids.
- LonDeg.mat: degrees of longitud according to the latitude with distance equivalent to 1 degree at the equator. This is used to search for nearest neighbours (NN) coastal points in the square to account for immigration.
- rhoExt.csv: mass extinction patterns to input in the model.
- observed_D.npz: The proof of concept data, the pattern diversity nowadays.
- indices_points.npz: The position of the points that we follow backwards across time slices in the three hotspots (in the publication: Mediterranean, Caribbean and Pacific). This allows to track how the diversity at a certain location has evolved along time.

# Model functions
The module principal_proof.py runs the following sequence of functions with their corresponding outputs:

- rhonet.py: calculates diversification rate (rho) and effective carrying capacity (Keff), furthermore it calculates the index with the point time slices that suffer a mass extinction, (ext_index)
- alphadiv.py: computes diversity in the model particles → D_shelf and rho_shelf_eff
- gridMean.py: calculates the mean diversity in 0.5ºx0.5º grids
- inditek_model_proof.py: compares the mean diversity with the proof diversity and calculates the Residual Sum of Squares Error.(RSME)

Then, to run the Metropolis-Hastings algorithm, you need to run the following scripts:

- metropolis_7param.py: Run the metropolis-hastings algorithm, calling principal_proof.py for the model output, and estimate the most probable parameters.
- indicios_7param.py: Load the data, call the metropolis function and save the results.

# Figures 2 to 5:

The script visualization.py plots the main results of the manuscript: (1) MCMC Chain Trajectories for parameter inference. (2) Proof-of-concept study used to validate the Bayesian inverse modelling framework of INDITEK-2.0. (3) The diversity maps of the calibrated model and of the proof model. It need to be in the  same folder of INDITEK*.npz generated with indicios_7param.py. 

Explanations of all the functions are written inside them. For any further doubt, do not hesitate to contact me: (Email)


