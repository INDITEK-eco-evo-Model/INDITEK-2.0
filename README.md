# INDITEK-2.0

This is the second version of INDITEK, a global model of marine invertebrates diversification ( measured in #genera My^-1)  in the Phanerozoic eon (from 541 Ma to present).

This version focuses on diversification on the continental platform, where the majority of marine diversity emerges, in order to speed up the model to simulations of ~30 seconds. This drastic increase in speed is cruciaal, as it allows the implementation of a Bayesian Markov chain Monte Carlo (MCMC) framework. With this new MCMC module, we can run thousands of simulations across multiple chains to infer the probability distributions of the model parameters based on their fit to present-day global observations. Ultimately, this framework enables us tp explore and test different hypotheses explaining biodiversity dynamics in deep time.

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

To run the model, the Python scripts (`.py` and `ipynb`) and the `data` folder must be located in the same dictionary. The `data` folder contains:

- `Point_ages_xyz.mat`: Seafloor age data from the plate-tectonic/paleo-elevation model.
- `Point_foodtemp.mat`: Food and temperature data from the cGenie earth-system model.
- `landShelfOceanMask.mat`: A 0-2 mask used to distinguish land-shelf-ocean grids.
- `LonDeg.mat`: Degrees of longitud according to latitude (with a distance equivalent to 1º at the equator). Used to find active nearest neighbours (NN) in a restricted area, mimicking immigration to newly submerged continental platform.
- `rhoExt.csv`: Mass extinction patterns inputted in the model.
- `observed_D.npz`: The proof of concept data, the pattern diversity nowadays.

## Running the model:

The main execution module is **`indicios_7param.py`**. This script estimates the model parameters probabilistically using a Metropolis-Hastings (M-H) MCMC algorithm. 

**Configuration**
Inside `indicios_7param.py`, you can modify the set-up to run the MCMC framework by defining
- `num_chains`: The number of parallel MCMC chains. 
- `nsamples`: The number of iterations.
- `model`:  The specific experiment you want to run.

**Available Experiments**
  
- **`proof`**: The proof-of-concept experiment.  
  
- **`open`**: The proof-of-concept experiment using a broader initial parameter range.  
  
- **`expo`** (Exponential): The exponential growth experiment, which removes the carrying capacity constraint.  
  
- **`temp`** (No temperature dependence):  The experiment that decouples speciation from temperature.  
   
- **`food`** (No food dependence): The experiment that removes the influence of marine export production on the speciation rate.

**Modifying Priors**

You can also adjust the prior distributions: the tolerance bounds (proposal outside these bounds are rejected), the mean and the standard deviation. These priors are inferred from existing literature. Additionally, you can modify the applied mass extinction pattern to compare the proof-of-concept against the output.

## Model Architecture and Workflow

To execute the M-H MCMC algorithm, run **`indicios_7param.py`**. This script loads the data, sets the priors, prepares the parallel chains, and calls **`metropolis_7param.py`** to run the algorithm:

**1. MCMC Framework** 
 
- **`indicios_7param.py`**: Initializes the execution and saves the final results in `inditekMCMCoutput_{nsamples}_{model}.npz`.
- **`metropolis_7param.py`**: Runs the M-H MCMC algorithm, evaluating the proposed parameters based on the model-observation fit.

**2. Core Diversification Model**
At each iteration, `metropolis_7param.py` calls `principal_proof.py`, which executes the following sequence:

- **`rhonet.py`**: Calculates the diversification rate (_rho_) and effective carrying capacity (_Keff_). It also recordds the time slices affected by mass extinctions (_ext_index_)
- **`alphadiv.py`**: Computes diversity in the model particles → *D_shelf* and *rho_shelf_eff*
- **`gridMean.py`**: calculates *D*, the mean diversity in 0.5ºx0.5º grids
- **`inditek_model_proof.py`**: compares *D* with *observed_D* and calculates the Residual Sum of Squares Error (RSME)

The final data are saved in INDITEK_MCMCoutput.npz that contains the following variables:

- **`Params_proposed_history`**: Stores the proposed parameter values for each iteration of the MCMC method.
- **`Params_accepted_history`**: Stores the parameter values that were accepted by the MCMC model after comparison with the previously accepted ones.
- **`rss_proposed_history`**: Records the resulting RSS of the model using the proposed parameters.
- **`rss_accepted_history`**: Records the RSS of the model using the accepted parameters.
- **`acceptance_history`**: Equals 1 if the proposed parameters were accepted, and 0 otherwise.
- **`sigma_new`**: Shows the current value of sigma after it is updated in each iteration.
- **`D`**: Represents the diversity calculated with the proposed parameters, saved every n_D iterations.



# Figures 2 to 5:

The script **visualization.py** plots the main images of the manuscript: (2) Markov chain trajectories for model parameters (3) Recovery of true eco-evolutionary parameters using Bayesian inversion. (4) Parameter posterior distributions and residual sum of squares across model configurations. It also plots the supplementary figures: (S1-S4) McMC chains for each different Experiment abd (S5) Global diversity map for each different experiment.

Further explanation can be found inside each function.

