# Inditek 2.0
The second version of indicios, a global model of diversification (#genera My^-1) of marine invertebrates in the Phanerozoic (from 541 Ma to present).

This version is written in Python 3.14.

# How to run the model:

The main module of INDITEK 2.0 is indicios_7param.py. To estimate probabilistically the parameters of the model, run indicios_7param.py. In inditek_main you can manually change the range of tolerance for the proposed parameter values in the M-H iterations (out of these bounds we reject the proposal), and the mean and standard deviation of the parameters distribution. You can also change the extinction pattern to apply and compare the proof of concept to the output.


To run the model, you need to have in the same folder the functions (.py) and the folder data. The folder data contains:

- Point_ages_xyzKocsisScotese_400.mat: floor age data from the plate-tectonic/paleo-elevation model.
- Point_foodtemp_paleoconfKocsisScotese_option2_GenieV4.mat: food-temp data from the cGenie earth-system model.
- landShelfOceanMask_ContMargMaskKocsisScotese.mat: 0-2 mask to distinguish land-shelf-ocean grids.
- LonDeg.mat: degrees of longitud according to the latitude with distance equivalent to 1 degree at the equator. This is used to search for nearest neighbours (NN) coastal points in the square to account for immigration.
- rhoExtOriginal_b.csv: mass extinction patterns to input in the model.
- datos_proof_2.npz: The proof of concept data, the pattern diversity nowadays.
- indices_points.npz: The index with the points in the three hotspots: mediterranean, caribbean and pacific. To track how the different hotspots have evolved during the time.

# Model functions
The module inditek_main runs the following sequence of functions with their corresponding outputs:

- rhonet.py: calculates diversification rate (rho) and effective carrying capacity (Keff), furthermore it calculates the index with the point time slices that suffer a mass extinction, (ext_index)
- alphadiv.py: computes diversity in the model particles → D_shelf and rho_shelf_eff
- gridMean.py: calculates the mean diversity in 0.5ºx0.5º grids
- inditek_model_proof.py: compares the mean diversity with the proof diversity and calculates the Residual Sum of Squares Error.(RSME)
- principal_proof.py: Run the previous functions and return the diversity and the RSME
- metropolis_7param.py: Run the metropolis-hastings algorithm, estimating the most probable parameters.
- indicios_7param.py: Load the data, call the metropolis function and save the results.

# Figures 2 to 5:

The script visualization.py plots the main results of the manuscript: (1) MCMC Chain Trajectories for parameter inference. (2) Proof-of-concept study used to validate the Bayesian inverse modelling framework of INDITEK-2.0. (3) The diversity maps of the calibrated model and of the proof model. It need to be in the  same folder of INDITEK*.npz generated with indicios_7param.py. 

Explanations of all the functions are written inside them. For any further doubt, do not hesitate to contact me: (Email)


