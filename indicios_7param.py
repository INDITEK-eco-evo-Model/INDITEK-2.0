import numpy as np
from latin_hypercube import latin_hypercube_sampling_ichains
from joblib import Parallel, delayed
import time
from metropolis_7param import inditek_metropolis
import mat73
import scipy.io

#Calls the MCMC function (Metropolis) in parallel and returns its output
def run_chain(iChain):
    
    params_current=np.transpose(initial_theta[iChain,:])
    output=inditek_metropolis(params_current, food_shelf, temp_shelf, Point_timeslices, shelf_lonlatAge, nsamples, nparams, proof, landShelfOceanMask, landShelfOcean_Lat, landShelfOcean_Lon, LonDeg, mu, sigma, ran_bounded, sigma_prop, n_D, model, active_params)

    return (
        output["params_proposed_history"], output["params_accepted_history"],
        output["rss_proposed_history"], output["rss_accepted_history"],
        output["acceptance_history"], output["sigma_prop"],
        output["D"],output["residuals"]
    )

start=time.time()

#MH-MCMC SETUP:
num_chains=1#number of chains
nsamples=3#number of iterations
n_D=1#how many iterations to store the D and residuals values (e.g., if n_D=10, it stores the D and residuals values every 10 iterations)
model='proof'

#############################################################
#Define the parameters for each model to be tested (open, proof, expo, food, temp)
#############################################################

models=["proof", "open", "expo", "food", "temp"]

ind_model=models.index(model)

dicts_models=[
    {"model":"proof", 
    "mu": np.array([161,19,0.035,0.002,2]), 
    "active_params": [0,1,2, 3, 4],
    "sigma": np.array([8,1.9,0.00035,0.0003,0.08]), 
    "ran_initial": np.array([[128.8,193.2],[15.2,22.8],[0.028,0.042],[0.0016,0.0024],[1.6,2.4]]), 
    "ran_bounded":[[50,1000],[1,150],[0.001,0.1],[0.0001,0.05],[1,3.5]]},

    {"model":"open", 
    "mu": np.array([161,19,0.035,0.002,2]), 
    "active_params": [0,1,2, 3, 4],
    "sigma": np.array([8,1.9,0.00035,0.00001,0.08]), 
    "ran_initial": np.array([[50,500],[0,50],[0.01,0.1],[0,0.01],[1.2,2.8]]), 
    "ran_bounded":[[50,1000],[1,150],[0.001,0.1],[0.0001,0.05],[1,3.5]]},

    {"model":"expo", 
    "mu": np.array([np.nan,np.nan,0.035,0.002,2]), 
    "active_params": [2, 3, 4],
    "sigma": np.array([np.nan,np.nan,0.00035,0.00001,0.08]), 
    "ran_initial": np.array([[0.028,0.042],[0.0016,0.0024],[1.6,2.4]]), 
    "ran_initial_open": np.array([[0.01,0.1],[0,0.01],[1.2,2.8]]),
    "ran_bounded":[np.nan,np.nan,[0.001,0.1],[0.0001,0.05],[1,3.5]]},

    {"model":"food", 
    "mu":np.array([161,19,0.035,0.002,2]), 
    "active_params": [0,1,2, 3, 4],
    "sigma":np.array([8,1.9,0.00035,0.00001,0.08]), 
    "ran_initial":np.array([[128.8,193.2],[15.2,22.8],[0.028,0.042],[0.0016,0.0024],[1.6,2.4]]),
    "ran_initial_open":np.array([[50,500],[0,50],[0.01,0.1],[0,0.01],[1.2,2.8]]), 
    "ran_bounded":[[50,1000],[1,150],[0.001,0.1],[0.0001,0.05],[1,3.5]]},

    {"model":"temp", 
    "mu": np.array([161,19,0.035,0.002,np.nan]), 
    "active_params": [0,1,2, 3],
    "sigma":np.array([8,1.9,0.00035,0.00001,np.nan]), 
    "ran_initial":np.array([[128.8,193.2],[15.2,22.8],[0.028,0.042],[0.0016,0.0024]]),
    "ran_initial_open":np.array([[50,500],[0,50],[0.01,0.1],[0,0.01]]), 
    "ran_bounded":[[50,1000],[1,150],[0.001,0.1],[0.0001,0.015],np.nan]},
]
#Mean of parameters distributions (mu):
#
#Kmax_mu = 200; Maximum carrying capacity (maximum number of genera at the point with the greatest food available in the time series)  
#Kmin_mu = 10; Minimum carrying capacity
#spec_max_mu = 0.15; Maximum speciation rate according to FoodlimxTemplim  
#spec_min_mu = 0.005;  Minimum speciation rate 
#Q10_mu = 2; Parameter defining the thermal limitation for speciation  


mu=dicts_models[ind_model]["mu"]

#Standard deviation of parameters distributions (sigma):  
#
#Kmax_std = 80  
#Kmin_std = 4
#spec_max_std = 0.05  
#spec_min_std = 0.002
#Q10_std = 0.2  

sigma=dicts_models[ind_model]["sigma"]

#Selection of the active parameters, the ones used during that experiment
active_params=dicts_models[ind_model]["active_params"]




########################## Parameter distribution of the search window to define the proposal:  

#Starting point of the chains (theta at time 1) found with latinhypercube to efficiently cover the param. distributions:  
#
#Kmax_theta = [250,500];    
#spec_max_theta = [0.1,1.5];  
#Q10_theta = [1,3];  
#ext_intercept_shelf_theta = [0],0.2];  
#ext_slope_theta = NAN;

if model=='proof' or model=='open':
    ran_initial=dicts_models[ind_model]["ran_initial"]
else:
    ran_initial=dicts_models[ind_model]["ran_initial_open"]
pre_initial_theta=latin_hypercube_sampling_ichains(np.array(ran_initial[:,0]), np.array(ran_initial[:,1]), num_chains)
initial_theta=np.full((num_chains,5),np.nan)
initial_theta[:,active_params]=pre_initial_theta


c=1
sigma_prop=c*sigma

#Range of parameters ([min,max])= range of tolerance for the proposed parameter values in the M-H iterations (out of these bounds we reject the proposal)
#
#Kmax_range = [200,1000];  
#Kmin_range =[1,150];
#spec_max_range = [0.05,Inf];  
#spec_min_range = [0.0001,0.05]
#Q10_range = [1,4];  
#ext_intercept_shelf_range = [0,Inf];  

ran_bounded=dicts_models[ind_model]["ran_bounded"]

nparams=len(mu)
print("Number of params:", nparams)
print("Number of iterations:", nsamples)
print("Number of chains:", num_chains)

#Pre-allocate variables to store results
#params_proposed_history: Stores the values of all proposed parameters
#params_accepted_history: Stores the values of the accepted parameters, if a proposal is rejected, it
# retains the previously accepted value
#rss_proposed history: Stores the RSS (Residual Sum of Squares) proposed during each iteration
#rss_accepted_history: Stores only the accepted RSS, following the same logic as with params_accepted_history
#D: Stores the diversity value of each grid cell (2978 in total) during each iteration
#residuals: Stores the residual value of each grid cell during each iteration
#sigma_new: Stores the variance of the parameter
#AR_parameter: Stores the acceptance rate of each parameter during each iteration

params_proposed_history = np.zeros([nsamples,nparams, num_chains])
params_accepted_history = np.zeros([nsamples+1,nparams, num_chains])
rss_proposed_history = np.zeros([nsamples,num_chains])
rss_accepted_history = np.zeros([nsamples,num_chains])
acceptance_history = np.zeros([nsamples,num_chains])
D = np.zeros([int(nsamples/n_D)+1, 2978, num_chains])
residuals = np.zeros([int(nsamples/n_D)+1, 2978, num_chains])
sigma_new=np.zeros([nsamples,nparams, num_chains])
AR_parameter=np.zeros([nparams,num_chains])

####################################################### Load input variables 

data_food_temp=scipy.io.loadmat('data/Point_foodtemp.mat')

food_shelf=data_food_temp['food_shelf']
temp_shelf=data_food_temp['temp_shelf']

data_point_ages=scipy.io.loadmat('data/Point_ages_xyz.mat')
Point_timeslices=data_point_ages['Point_timeslices'].astype(int)
shelf_lonlatAge=data_point_ages['shelf_lonlatAge']

data_LonDeg=scipy.io.loadmat('data/LonDeg.mat')

LonDeg=data_LonDeg['LonDeg']

data_Mask=mat73.loadmat('data/landShelfOceanMask.mat')

landShelfOcean_Lat=data_Mask['landShelfOcean_Lat']
landShelfOcean_Lon=data_Mask['landShelfOcean_Lon']
landShelfOceanMask=data_Mask['landShelfOceanMask']
landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)

data_proof=np.load("data/observed_D.npz")
proof=data_proof[ "proof"]


#########################################################Start the parallel computation

results= Parallel(n_jobs=num_chains)(delayed(run_chain)(i) for i in range(num_chains))

#Stores the values for each variable obtained from the metropolis script
for iChain, result in enumerate(results):

    params_proposed_history[:, :, iChain] = result[0]
    params_accepted_history[:, :, iChain] = result[1]
    rss_proposed_history[:, iChain] = result[2].flatten()
    rss_accepted_history[:, iChain] = result[3].flatten()
    acceptance_history[:, iChain] = result[4].flatten()
    sigma_new[:,:,iChain]=result[5]
    D[:,:,iChain]=result[6]
    residuals[:,:,iChain]=result[7]

#Save all variables to the final .npz file
np.savez(f"probita/inditekMCMCoutput_{nsamples}_{model}.npz", params_proposed_history=params_proposed_history, params_accepted_history=params_accepted_history, rss_proposed_history=rss_proposed_history, rss_accepted_history=rss_accepted_history, acceptance_history=acceptance_history, D=D, residuals=residuals, sigma_new=sigma_new, AR_parameter=AR_parameter)

end=time.time()
print('{:.4f} s'.format(end-start)) 
