import scipy.io
import mat73
import numpy as np
import pandas as pd
import time
from inditek_rhonet_2 import inditek_rhonet
from inditek_alphadiv_2 import inditek_alphadiv
from alphadiv_expo import alphadiv_expo
from inditek_gridding_alphadiv_2 import inditek_gridding_alphadiv
from inditek_model_proof import inditek_model_proof

start_time = time.time()


#def principal(Kmax_mean, spec_max_mean, Q10_mean, ext_intercept_shelf_mean,ext_slope_mean):
def inditek_main(kfood, Kmin, food_shelf, temp_shelf, ext_pattern, Kmax_mean, spec_min_mean, spec_max_mean, Q10_mean, ext_intercept_shelf_mean,ext_slope_mean, shelf_lonlatAge, Point_timeslices, latWindow,lonWindow,LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, proof, model):

        #############################################################################################
        # JUST FOR TESTING PURPOSES [161, 19, 0.035, 0.002, 1.75]
        ###############################################################################################
        '''
        model="proof"
        #CHOOSE model parameters:
        kfood = 0.5 #[POC mol * m-2 yr-1] #1
        spec_min_mean = 0.00214#0.0097 #[MA-1]   #0.1
        spec_max_mean = 0.0348#0.02 #[MA-1]   #la sacas
        Q10_mean = 2.0036 #n.u.   #la sacas 
        Kmax_mean=162.44# Carrying capacity of #genera at maximum food availability #la sacas  
        Kmin=18.8 # Carrying capacity of #genera at minimum food availability #10
        ext_slope_mean=0 #la sacas
        ext_intercept_shelf_mean=0 #la sacas
        
        latWindow=2.5 #2.5
        lonWindow=2.5 #2.5
        
        ext_pattern=4 #3
        
        data_food_temp=scipy.io.loadmat('data_input/Point_foodtemp.mat')
        
        #food_ocean=data_food_temp['food_ocean']
        food_shelf=data_food_temp['food_shelf']
        #temp_ocean=data_food_temp['temp_ocean']
        temp_shelf=data_food_temp['temp_shelf']
        
        data_point_ages=scipy.io.loadmat('data_input/Point_ages_xyz.mat')
        
        Point_timeslices=data_point_ages['Point_timeslices'].astype(int)
        #Point_timeslices=Point_timeslices[0]
        shelf_lonlatAge=data_point_ages['shelf_lonlatAge']
        
        data_LonDeg=scipy.io.loadmat('data_input/LonDeg.mat')
        #print(data_LonDeg.keys())
        
        LonDeg=data_LonDeg['LonDeg']
        
        data_Mask=mat73.loadmat('data_input/landShelfOceanMask.mat')
        #print(data_Mask.keys())
        
        landShelfOcean_Lat=data_Mask['landShelfOcean_Lat']
        landShelfOcean_Lon=data_Mask['landShelfOcean_Lon']
        landShelfOceanMask=data_Mask['landShelfOceanMask']
        landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)
        
        data_proof=np.load("data_input/proof_of_concept.npz")
        proof=data_proof["proof"]
        '''
        ############################################
        #END OF LOADING DATA
        ############################################
        # Call inditek_rhonet_2 function to calculate, for points defining the continental shelves, the rho_shelf (net diversification rate) and K_shelf (carrying capacity) matrices.
        [rho_shelf,K_shelf, ext_index]=inditek_rhonet(kfood,Kmin,food_shelf,temp_shelf,ext_pattern,Kmax_mean,spec_min_mean,spec_max_mean, Q10_mean,ext_intercept_shelf_mean,ext_slope_mean,shelf_lonlatAge,Point_timeslices[0], model)

        # Call inditek_alphadiv function to calculate D_shelf (diversity in each palaeogeographic point at each time slice) matrix.
        # If the model is "expo", it calls the alphadiv_expo function that does not include the effect of K on diversity.
        if model=="expo":
            D_shelf=alphadiv_expo(Point_timeslices,shelf_lonlatAge,rho_shelf,latWindow,lonWindow,LonDeg, ext_index)
        else:
            [rho_shelf_eff,D_shelf]=inditek_alphadiv(Point_timeslices,shelf_lonlatAge,rho_shelf,K_shelf,latWindow,lonWindow,LonDeg, ext_index)
               
        # Call inditek_gridding_alphadiv function to compute D: mean diversity of points falling in each grid cell covering the continental shelves.
        [X,Y,D]=inditek_gridding_alphadiv(D_shelf,shelf_lonlatAge,landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask)
        
        
               
        # Model-Observation comparison: Calculates RSS (Residual Sum of Squares) 
        # This function will be modified in future updates to accomodate for real observations: OBIS data.
        [rss, residuals]=inditek_model_proof(D,proof) 

        # Clean NaN values (empty grid cells) from D 
        D=D[~np.isnan(D)]

        # Return RSS, D (model diversity for the active points averge by grid cell) and the residuals (difference between model and observed diversity by grid cell).
        return [rss, D, residuals]
     #elapsed_time = time.time() - start_time
#
     #print(f"La función tardó {elapsed_time:.4f} segundos.")






