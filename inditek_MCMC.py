import numpy as np
from inditek_main_2 import inditek_main
import scipy.io
import time
import mat73

# Charge the data (just for tests)
'''
##############################################################################################
# uncomment JUST FOR TESTING PURPOSES TO LOAD THE DATA independently from the parallelization in the main function indicios_7param.py
###############################################################################################

data_point_ages=scipy.io.loadmat('Point_ages_xyzKocsisScotese_400.mat')
shelf_lonlatAge=data_point_ages['shelf_lonlatAge']
Point_timeslices=data_point_ages['Point_timeslices']

data_mask=mat73.loadmat('landShelfOceanMask_ContMargMaskKocsisScotese.mat')
landShelfOcean_Lat=data_mask['landShelfOcean_Lat']
landShelfOcean_Lon=data_mask['landShelfOcean_Lon']
landShelfOceanMask=data_mask['landShelfOceanMask']
landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)

data_food_temp=scipy.io.loadmat('Point_foodtemp_v241023.mat')

food_shelf=data_food_temp['food_shelf']
temp_shelf=data_food_temp['temp_shelf']

data_LonDeg=scipy.io.loadmat('LonDeg.mat')
#print(data_LonDeg.keys())

LonDeg=data_LonDeg['LonDeg']
num_chains=2
nsamples=3
nparams=7

data=np.load("indicios.npz")
params_current=data["params_current"]
mu=data["mu"]
sigma=data["sigma"]
sigma_prop=data["sigma_prop"]
ran=data["ran"]

########################################################
#END OF LOADING DATA
#########################################################
'''
def inditek_MCMC(params_current, food_shelf, temp_shelf, Point_timeslices, shelf_lonlatAge, nsamples, nparams, proof, landShelfOceanMask, landShelfOcean_Lat, landShelfOcean_Lon, LonDeg, mu, sigma, ran, sigma_prop, n_D, model, active_params):


    gaus=np.array([])
    #Saves the fix parameters
    ext_intercept=0 #  intercept of the extinction as a function of latitude, which is fixed to 0 in all experiments (lambda is thus treated as net diversification, same as in Cermeño et al 2022). It can be changed in future implementations.
    ext_slope=0 #  slope of the extinction as a function of latitude, which is fixed to 0 in all experiments (lambda is thus treated as net diversification, same as in Cermeño et al 2022). It can be changed in future implementations.
    if model!="temp":
        gaus=np.array([4]) # position of parameter with gaussian prior distributions (for now only Q10:temperature dependency supported by bibliography) 
    kfood = 0.5 # half-saturation constant for food limitation[POC mol * m-2 yr-1]
    lonWindow=2.5 # distance in degrees to search for particles from which diversity is "migrated" into the new coastal particles (newly submerged or artificially created by the paleotectonic model). 
    latWindow=2.5 # same but for latitude.
    ext_pattern=4 # big five imposed extinctions according to fossil curves: Zaffos curve = 1, % Alroy curve = 2, % Sepkoski curve = 3, % average of all curves = 4

    # Storage for Diagnostics (see definitions in indicios_7param.py)
    output={
        "params_proposed_history": np.zeros([nsamples,nparams]),
        "params_accepted_history": np.zeros([nsamples+1,nparams]),
        "rss_proposed_history": np.zeros([nsamples,1]),
        "rss_accepted_history": np.zeros([nsamples,1]),
        "acceptance_history": np.zeros([nsamples,1]),
        "D": np.zeros([int(nsamples/n_D)+1,2978]),
        "residuals": np.zeros([int(nsamples/n_D)+1,2978]),
        "sigma_prop": np.zeros([nsamples,nparams]),
        "AR_parameter": np.zeros(nparams),
        }


    # Save the initial parameters in the output dictionary
    output["params_proposed_history"][0,:]=params_current # Calculated in inditek_7param.py as params_current=initial_theta(iChain,:)

    # Initial RSS Calculation (before the loop)
    # Calculates the initial RSS (Residual Sum of Squares) for the current parameters
    # It also saves the current global diversity value in D and the residuals for each grid cell in the residuals variable

    [rss_current,D, residuals]=inditek_main(kfood, params_current[1], food_shelf, temp_shelf, ext_pattern, params_current[0], 
                                                  params_current[3], params_current[2], params_current[4], ext_intercept, ext_slope, shelf_lonlatAge, Point_timeslices, 
                                                  latWindow,lonWindow,LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, proof, model)
    
    # --- PRIOR AND INITIAL LOG-POSTERIOR SETUP ---

    # Prior probability for current parameters:
    # - Uniform priors contribute 0 to log-prior (since log(1) = 0)
    # - Gaussian priors (e.g., for Q10 parameters) are handled separately

    temp=np.zeros(nparams) # Force those with a uniform distribution to a probability of 1 along the range (log(1)=0;)
    # Assign Gaussian prior contributions (only for parameters with Gaussian prior)
    if gaus.size>0:
        temp[gaus]=mu[gaus]

    # Computethe log(prior), log(likelihood) and log(posterior) for current parameters to compare them to the proposed ones in the loop
    log_prior_current=-0.5*sum(temp)
    log_likelihood_current=-0.5*rss_current #based on the assumption of normally distributed (gaussian) errors
    log_posterior_current =log_prior_current+log_likelihood_current

    # --- METROPOLIS-HASTINGS SAMPLING SETUP ---

    # Number of iterations used to adapt proposal variance (sigma_prop)
    n_AR=50
    # Matrix to track when each parameter is updated (stores iteration indices)
    change_params=np.full([nparams,n_AR], np.nan)

    #Index to track if the parameter has been updated in each iteration
    #It is initialized to NaN to ensure correct behavior in the first iteration
    index2=np.nan
    # Track acceptance rate per parameter (number of accepted proposals)
    AR_parameter=np.zeros(nparams)
    # Track how many times each parameter is proposed
    new_parameter=np.zeros(nparams)
    
    # --- MAIN MCMC LOOP ---

    #Initialize the loop for the number of iterations defined in nsamples

    for iter in range(1,nsamples):

         # Randomly select one active parameter to perturb (uniform probability of selection among active parameters)
        index1= np.random.randint(min(active_params), max(active_params)+1)
        new_parameter[index1]+=1

        
        # If a different parameter than in previous iteration is selected,
        # record this change in the tracking structure

        if index1!=index2:
            
            idx = np.isnan(change_params[index1]).argmax() #1st empty slot
            change_params[index1][idx] = iter


        # Initialize the proposed parameters with the current ones
        params_proposed=params_current.copy()

        # Adaptive proposal variance: 
        # every n_AR valid updates of a parameter, adjust its sigma_prop according to its acceptance rate (AR) in the last n_AR iterations.
        
        if change_params[index1][np.isnan(change_params[index1])==0].size%n_AR==0 and index2!=index1:
            
            # Calculate the acceptance rate (AR) of the parameter that has changed as the sum of the acceptance history divided by the number of iterations
            # Get the iteraction index where the parameter was updated
            correct_index = change_params[index1][np.isnan(change_params[index1])==0].astype(int)
            # Compute acceptance rate over those iterations
            AR=np.nanmean(output["acceptance_history"][correct_index])
            # Reset the change tracking for that parameter
            change_params[index1]=np.full([n_AR], np.nan)
            # Adapt proposal standard deviation (sigma_prop) for that parameter toward the target acceptance rate of 0.4
            sigma_prop[index1]=max(sigma_prop[index1]*AR/(0.4),1e-12)

        # --- PROPOSE NEW PARAMETER VALUE --
        # Perturb the selected parameter (propose a value randomly) according to a search window represented by a Gaussian with a standard deviation equal to sigma_prop
        A=np.random.randn(nparams,1)
        params_proposed[index1]=params_current[index1]+A[index1,0]*sigma_prop[index1]

        # --- CHECK PARAMETER BOUNDS ---
        # Ensure that all active parameters remain within predefined limits
        limit_inf = [fila[0] for fila in ran if not np.any(np.isnan(fila))]
        limit_sup = [fila[1] for fila in ran if not np.any(np.isnan(fila)) ]
        if np.all(params_proposed[active_params] <= limit_sup) and np.all(params_proposed[active_params] >=limit_inf):
            
            # --- MODEL EVALUATION ---
            # Run the diversification model with proposed parameters and compute RSS (Residual Sum of Squares) for the proposed parameters

            [rss_proposed,D, residuals]=inditek_main(kfood, params_proposed[1],
                                                            food_shelf, temp_shelf, ext_pattern, params_proposed[0], params_proposed[3], params_proposed[2], params_proposed[4], ext_intercept, ext_slope, 
                                                            shelf_lonlatAge, Point_timeslices, latWindow,lonWindow,LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, proof, model) 
            
            # --- COMPUTE POSTERIOR FOR PROPOSED PARAMETERS ---
            temp=np.zeros([nparams,1])
            if gaus.size>0:
                temp[gaus]=((params_current[gaus] - mu[gaus]) / sigma[gaus])**2
            log_prior_proposed=-0.5*sum(temp)
            
            log_likelihood_proposed=-(1/2)*rss_proposed
            
            log_posterior_proposed =log_prior_proposed+log_likelihood_proposed
           
            # --- STORE CURRENT ITERATION DATA ---
            # Save the results in the output dictionary

            output["params_proposed_history"][iter, 0:nparams]=params_proposed
            output["params_accepted_history"][iter, 0:nparams]=params_current
            output["rss_proposed_history"][iter]=rss_proposed
            output["rss_accepted_history"][iter]=rss_current
            output["sigma_prop"][iter]=sigma_prop

            # Additionally save, every n_D iterations, the diversity and residuals in the output dictionary
            if iter % n_D == 0:
                output["D"][int(iter/n_D),:]=D
                output["residuals"][int(iter/n_D),:]=residuals

            # --- METROPOLIS-HASTINGS ACCEPTANCE STEP ---
            # Calculate the acceptance probability according to the ratio between the likelihood of proposed vs. current 
            # (log_posterior_proposed-log_posterior_current) and compare it to a random number u (0-1):
           
            u=np.random.rand()
            if np.log(u)<log_posterior_proposed-log_posterior_current:

                
                # Accept proposal
                acceptance_tagmark=1 # Mark as accepted
                AR_parameter[index1]+=1
                
                # UPDATE (current) parameter values for NEXT ITERATION
                params_current=params_proposed.copy()
                rss_current=rss_proposed
                log_posterior_current=log_posterior_proposed.copy()
            else:
                # Reject proposal
                acceptance_tagmark=0 # Mark as not accepted
                # DO NOT UPDATE params_current as params_proposed FOR NEXT ITERATION

            output["acceptance_history"][iter]=acceptance_tagmark

        else:
            # for Out-of-bounds proposals: SKIP the evaluation procedure  

            output["params_proposed_history"][iter, 0:nparams]=params_proposed
            output["params_accepted_history"][iter, 0:nparams]=params_current
            output["rss_proposed_history"][iter]=np.nan
            output["rss_accepted_history"][iter]=rss_current
            output["sigma_prop"][iter]=sigma_prop
            output["acceptance_history"][iter]=0

            if iter % n_D == 0:
                output["D"][int(iter/n_D),:]=D
                output["residuals"][int(iter/n_D),:]=residuals

        # Store current parameter index for next interation
        index2=index1

        #print("#########################################################################")
        #print("ONE ITERATION DONE")
        #print("############################################################################")
    # Save final accepted parameters after the last iteration
    output["params_accepted_history"][iter+1, 0:nparams]=params_current

################################################################
#save the output dictionary to a .npz file for testing purposes
################################################################

    # np.savez("final_data_metropolis.npz", params_proposed_history=output["params_proposed_history"], params_accepted_history=output["params_accepted_history"],
    # rss_proposed_history=output["rss_proposed_history"], rss_accepted_history=output["rss_accepted_history"],
    # acceptance_history=output["acceptance_history"], log_posterior_diff_history=output["log_posterior_diff_history"])


    #Return the output dictionary with the results of the Metropolis-Hastings algorithm
    return output







