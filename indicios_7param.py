import numpy as np
from latin_hypercube import latin_hypercube_sampling_ichains
from joblib import Parallel, delayed
import time
from metropolis_sigma import inditek_metropolis
import mat73
import scipy.io


def run_chain(iChain):
    
    params_current=np.transpose(initial_theta[iChain,:])
    #output=inditek_metropolis(LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, d_obis,se_obis, idx_obis, shelf_lonlatAge, Point_timeslices, food_shelf, temp_shelf, initial_theta[iChain,:], mu, sigma, sigma_prop, ran,  nsamples, nparams)
    #output=inditek_metropolis(       initial_theta[iChain,:],      )
    output=inditek_metropolis(params_current, food_shelf, temp_shelf, Point_timeslices, shelf_lonlatAge, nsamples, nparams, proof, landShelfOceanMask, landShelfOcean_Lat, landShelfOcean_Lon, LonDeg, mu, sigma, ran_bounded, sigma_prop, n_D, model, active_params)

    return (
        output["params_proposed_history"], output["params_accepted_history"],
        output["rss_proposed_history"], output["rss_accepted_history"],
        output["acceptance_history"], output["sigma_prop"],
        output["D"]
    )

start=time.time()

#MH-MCMC SETUP:
num_chains=6
nsamples=5002
n_D=50
model='proof'

###########################################################3
#Define the parameters of each different model to test (open, proof, expo, food, temp)
#############################################################

models=["open", "proof", "expo", "food", "temp"]

ind_model=models.index(model)

dicts_models=[
    {"model":"open", 
    "mu": np.array([161,19,0.035,0.002,1.75]), 
    "active_params": [0,1,2, 3, 4],
    "sigma": np.array([8,1.9,0.00035,0.0001,0.08]), 
    "ran_initial": np.array([[50,500],[0,50],[0.01,0.1],[0,0.01],[1.2,3.8]]), 
    "ran_bounded":[[0,1000],[1,150],[0.001,np.inf],[0.0001,0.05],[1,4]]},

    {"model":"proof", 
    "mu": np.array([161,19,0.035,0.002,1.75]), 
    "active_params": [0,1,2, 3, 4],
    "sigma": np.array([8,1.9,0.00035,0.0003,0.08]), 
    "ran_initial": np.array([[96.6,225.4],[11.4,26.6],[0.021,0.049],[0.0012,0.0028],[1.1,2.45]]), 
    "ran_bounded":[[50,1000],[1,150],[0.001,np.inf],[0.0001,0.05],[1,4]]},

    {"model":"expo", 
    "mu": np.array([np.nan,np.nan,0.035,0.002,1.75]), 
    "active_params": [2, 3, 4],
    "sigma": np.array([np.nan,np.nan,0.00035,0.00001,0.08]), 
    "ran_initial": np.array([[0.028,0.042],[0.0016,0.0024],[1.4,2.1]]), 
    "ran_bounded":[np.nan,np.nan,[0.001,np.inf],[0.0001,0.05],[1,4]]},

    {"model":"food", 
    "mu":np.array([161,19,0.035,0.002,1.75]), 
    "active_params": [0,1,2, 3, 4],
    "sigma":np.array([8,1.9,0.00035,0.00001,0.08]), 
    "ran_initial":np.array([[128.8,193.2],[15.2,22.8],[0.028,0.042],[0.0016,0.0024],[1.4,2.1]]), 
    "ran_bounded":[[50,1000],[1,150],[0.001,np.inf],[0.0001,0.05],[1,4]]},

    {"model":"temp", 
    "mu": np.array([161,19,0.035,0.002,np.nan]), 
    "active_params": [0,1,2, 3],
    "sigma":np.array([8,1.9,0.00035,0.00001,np.nan]), 
    "ran_initial":np.array([[128.8,193.2],[15.2,22.8],[0.028,0.042],[0.0016,0.0024]]), 
    "ran_bounded":[[50,1000],[1,150],[0.001,np.inf],[0.0001,0.05],np.nan]},
]
#Mean of parameters distributions (mu):
#
#Kmax_mu = 200; % Maximum carrying capacity (maximum number of genera in a point with the greatest food available in the time series)  
#Kmin_mu = 10; Minimum carrying capacity
#spec_max_mu = 0.15; %greatest speciation rate according to FoodlimxTemplim  
#spec_min_mu = 0.005; 
#Q10_mu = 2; % parameter defining the thermal limitation for speciation  
#ext_intercept_shelf_mu = 0.01; % background extinction in the tropics  
#ext_slope_mu = 0; %slope of extinction rate according to absolute latitude from 20º


mu=dicts_models[ind_model]["mu"]

#Standard deviation of parameters (sigma):  

#Kmax_std = 80  
#Kmin_std = 4
#spec_max_std = 0.05  
#spec_min_std = 0.002
#Q10_std = 0.2  
#ext_intercept_shelf_std = 0.005  
#ext_slope_std = NAN

sigma=dicts_models[ind_model]["sigma"]
active_params=dicts_models[ind_model]["active_params"]

#which parameters to consider with gaussian distribution instead of uniform distribution (no a priori, only bounds: range)

#gaus=np.array([4]) # for now we only believe that Q10 should fall around the value 2 



########################## Parameter distribution of the search-window to defign the proposal:  

#Starting point of the chains (theta at time 1) found with latinhypercube to efficiently cover the param.distributions:  
#
#Kmax_theta = [250,500];    
#spec_max_theta = [0.1,1.5];  
#Q10_theta = [1,3];  
#ext_intercept_shelf_theta = [0],0.2];  
#ext_slope_theta = NAN;

ran_initial=dicts_models[ind_model]["ran_initial"]
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
print("nparams:", nparams)
#Pre-allocate variables to store results


params_proposed_history = np.zeros([nsamples,nparams, num_chains])
params_accepted_history = np.zeros([nsamples+1,nparams, num_chains])
rss_proposed_history = np.zeros([nsamples,num_chains])
rss_accepted_history = np.zeros([nsamples,num_chains])
acceptance_history = np.zeros([nsamples,num_chains])
D = np.zeros([int(nsamples/n_D)+1, 2978, num_chains])
residuals = np.zeros([int(nsamples/n_D)+1, 2978, num_chains])
sigma_new=np.zeros([nsamples,nparams, num_chains])
AR_parameter=np.zeros([nparams,num_chains])

####################################################### 
# Load input variables 
#######################################################

data_food_temp=scipy.io.loadmat('data/Point_foodtemp.mat')

#print(data_food_temp.keys())
#food_ocean=data['food_ocean']
food_shelf=data_food_temp['food_shelf']
#temp_ocean=data['temp_ocean']
temp_shelf=data_food_temp['temp_shelf']

data_point_ages=scipy.io.loadmat('data/Point_ages_xyz.mat')#
#print(data_point_ages.keys())
#
Point_timeslices=data_point_ages['Point_timeslices'].astype(int)
#Point_timeslices=Point_timeslices[0]
shelf_lonlatAge=data_point_ages['shelf_lonlatAge']

data_LonDeg=scipy.io.loadmat('data/LonDeg.mat')
#print(data_LonDeg.keys())

LonDeg=data_LonDeg['LonDeg']

data_Mask=mat73.loadmat('data/landShelfOceanMask.mat')
#print(data_Mask.keys())

landShelfOcean_Lat=data_Mask['landShelfOcean_Lat']
landShelfOcean_Lon=data_Mask['landShelfOcean_Lon']
landShelfOceanMask=data_Mask['landShelfOceanMask']
landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)

data_proof=np.load("proof_of_concept.npz")
proof=data_proof[ "proof"]


#########################################################Start the parallel computation



results= Parallel(n_jobs=num_chains)(delayed(run_chain)(i) for i in range(num_chains))


#Save the results of the parallel computation


for iChain, result in enumerate(results):

    params_proposed_history[:, :, iChain] = result[0]
    params_accepted_history[:, :, iChain] = result[1]
    rss_proposed_history[:, iChain] = result[2].flatten()
    rss_accepted_history[:, iChain] = result[3].flatten()
    acceptance_history[:, iChain] = result[4].flatten()
    sigma_new[:,:,iChain]=result[5]
    D[:,:,iChain]=result[6]

np.savez(f"datos_indicios_{nsamples}_{model}.npz", params_proposed_history=params_proposed_history, params_accepted_history=params_accepted_history, rss_proposed_history=rss_proposed_history, rss_accepted_history=rss_accepted_history, acceptance_history=acceptance_history, D=D, residuals=residuals, sigma_new=sigma_new, AR_parameter=AR_parameter)

#Finally, to measure the time it costs for the simulation
end=time.time()
print('{:.4f} s'.format(end-start)) 
