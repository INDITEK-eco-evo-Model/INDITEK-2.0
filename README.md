# INDITEK-2.0

This is version 2.0 of INDITEK, a global model of marine invertebrates diversification ( measured in #genera My^-1)  throughout the Phanerozoic eon (from 541 Ma to present). In this version, we implement a Bayesian parameterization module that relies on fitting present-day patterns simulated by INDITEK to empirical observations.

This version focuses on diversification on the continental platform, where the majority of marine diversity is concentrated, in order to reduce computational costs and achieve simulation times of ~30 seconds.  This substantial increase in speed is crucial, as it enables the implementation of a Bayesian Markov chain Monte Carlo (MCMC) framework. With this new MCMC module, we can run thousands of simulations across multiple chains to infer the posterior distributions of model parameters based on their fit to present-day global observations. Ultimately, this framework enables us to explore and test different hypotheses explaining biodiversity dynamics through deep time.

**Funding and Citation**

This work was funded by national research grant PID2023-152076NB-I00  (INDICIOS project) from the Spanish government.
For further details, please refer to our publication: 
**INDITEK-2.0: A Bayesian inverse eco-evolutionary modelling framework for reconstructing Phanerozoic biodiversity**
DOI: 10.1101/2025.08.22.671786

Compatibility: Written in Python 3.9.18 and tested with 3.8.12. 

## Installation

Install the required dependencies by running the following command in the project root:

```bash
pip install -r requirements.txt
```

## Data dictionary

To run the model, the Python scripts (`.py` and `ipynb`) and the `data`, `output_data`, `ìmages` and `tool_for_images` folders must be located in the same dictionary. The `data` folder contains:

- `Point_ages_xyz.mat`: Seafloor age data from the plate-tectonic/paleo-elevation model.
- `Point_foodtemp.mat`: Food and temperature data from the cGenie earth-system model.
- `landShelfOceanMask.mat`: A 0-2 mask used to distinguish land, shelf and ocean grids.
- `LonDeg.mat`: Degrees of longitud as a function of latitude (with a distance equivalent to 1º at the equator). Used to identify active nearest neighbours (NN) within a restricted area, mimicking immigration to newly submerged continental platform.
- `rhoExt.csv`: Mass extinction patterns inputted in the model.
- `observed_D.npz`: Observed present-day biodiversity patterns (in this case proof-of-concept simulated data) created with `synthetic_data.py` script.

The other folders are used for:
- `images`: Contains all the images generated for the manuscript.
- `output_data`: Contains the main outputs of the model scripts.
- `tool_for_images`: Auxiliary data necessary for the `final_visualization` script.

## Running the model:

The main execution module is **`inditek_parallelMCMC.py`**. This script estimates model parameters probabilistically using a Metropolis-Hastings (M-H) MCMC algorithm. 

**Configuration**
Inside `inditek_paralelMCMC.py`, you can modify the set-up to run the MCMC framework by defining
- `num_chains`: The number of parallel MCMC chains. 
- `nsamples`: The number of iterations.
- `model`:  The specific experiment you want to run.

**Available Experiments**
  
- **`proof`**: The proof-of-concept experiment.  
  
- **`open`**: The proof-of-concept experiment using a broader initial parameter range.  
  
- **`expo`** (Exponential): Removes the carrying capacity constraint.  
  
- **`temp`** (No temperature dependence):  Decouples speciation from temperature.  
   
- **`food`** (No food dependence): Removes the influence of marine export production on the speciation rate.

**Modifying Priors**

You can also adjust the prior distributions of the parameters (Kmax, Kmin, spec_max, spec_min, Q10),  including tolerance bounds (proposal outside these bounds are rejected) `ran_bound` and the range of the initial window of the parameters `ran_initial`, as well as the mean `mu` and the standard deviation `sigma` of parameters distributions. These priors are inferred from existing literature.

## Model Architecture and Workflow

To execute the M-H MCMC algorithm, run **`inditek_parallelMCMC.py`**. This script loads the data, sets the priors, prepares the parallel chains, and calls **`inditek_MCMC.py`** to run the algorithm:

**1. MCMC Framework** 
 
- **`inditek_parallelMCMC.py`**: Initializes the execution and saves the final results in `inditekMCMCoutput_{nsamples}_{model}.npz`.
- **`inditek_MCMC.py`**: Runs the M-H MCMC algorithm, evaluating the proposed parameters based on the model-observation fit.

**2. Core Diversification Model**
At each iteration, `inditek_MCMC.py` calls `inditek_main_2.py`, which executes the following sequence:

- **`inditek_rhonet_2.py`**: Calculates the diversification rate (_rho_) and effective carrying capacity (_Keff_). It also recordds time slices affected by mass extinctions (_ext_index_)
- **`inditek_alphadiv_2.py`**: Computes diversity in the model particles → *D_shelf* and *rho_shelf_eff*
- **`inditek_gridding_alphadiv.py`**: Calculates _D_, the mean diversity in 0.5ºx0.5º grid cells above the global continental shelves.
- **`inditek_model_proof.py`**: Compares simulated diversity (_D_) with the empirical data (_observed_D_) and calculates the Residual Sum of Squares Error (RSS)

## Outputs

Results are saved in `inditekMCMCoutput_{nsamples}_{model}.npz`, containing:

- **`params_proposed_history`**: Proposed parameter values for each MCMC iteration.
- **`params_accepted_history`**: Accepted parameter values by the MCMC algorithm at each iteration
- **`rss_proposed_history`**: RSS for proposed parameters at each iteration
- **`rss_accepted_history`**: RSS for proposed parameters at each iteration
- **`acceptance_history`**: Binary flag (1 = accepted, 0 = rejected) at each iteration.
- **`sigma_new`**: Updated sigma value at each iteration.
- **`D`**: Simulated diversity computed with the proposed parameters per grid cell (saved every _n_D_ iterations).
- **`residuals`**: Residuals  per grid cell (saved every _n_D_ iterations).



## Visualization

The script `final_visualization.py` generates the main manuscript figures: 
- **Figure 2:** Markov chain trajectories for model parameters 
- **Figure 3:** Recovery of true eco-evolutionary parameters using Bayesian inversion.
- **Figure 4** Parameter posterior distributions and RSS across model configurations.
Supplementary figures:
- **Figures S1-S4** MCMC chains for each different experiment 
- **Figure S5:** Global diversity map for each different experiment.

Further explanation can be found inside each function.

