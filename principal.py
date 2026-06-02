import scipy.io
import mat73
import numpy as np
import pandas as pd
import time
from rhonet import rhonet_evo
from alphadiv_huer import alphadiv
from alphadiv_expo import alphadiv_expo
from gridMean import inditek_gridMean_alphadiv
from inditek_model_proof import inditek_model_proof

start_time = time.time()


#def principal(Kmax_mean, spec_max_mean, Q10_mean, ext_intercept_shelf_mean,ext_slope_mean):
def principal(kfood, Kmin, food_shelf, temp_shelf, ext_pattern, Kmax_mean, spec_min_mean, spec_max_mean, Q10_mean, ext_intercept_shelf_mean,ext_slope_mean, shelf_lonlatAge, Point_timeslices, latWindow,lonWindow,LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, proof, model):

#############################################################################################
# JUST FOR TESTING PURPOSES
###############################################################################################
        '''
        #CHOOSE model parameters:
        kfood = 0.5 #[POC mol * m-2 yr-1] #1
        spec_min_mean = 0.002 #[MA-1]   #0.1
        spec_max_mean = 0.035 #[MA-1]   #la sacas
        Q10_mean = 1.75 #n.u.   #la sacas 
        Kmax_mean=161# Carrying capacity of #genera at maximum food availability #la sacas  
        Kmin=19 # Carrying capacity of #genera at minimum food availability #10
        ext_slope_mean=0 #la sacas
        ext_intercept_shelf_mean=5 #la sacas
        
        latWindow=2.5 #2.5
        lonWindow=2.5 #2.5
        
        ext_pattern=4 #3
        
        data_food_temp=scipy.io.loadmat('Point_foodtemp.mat')
        
        food_ocean=data['food_ocean']
        food_shelf=data_food_temp['food_shelf']
        temp_ocean=data['temp_ocean']
        temp_shelf=data_food_temp['temp_shelf']
        
        data_point_ages=scipy.io.loadmat('Point_ages_xyz.mat')
        
        Point_timeslices=data_point_ages['Point_timeslices']
        Point_timeslices=Point_timeslices[0]
        shelf_lonlatAge=data_point_ages['shelf_lonlatAge']
        
        data_LonDeg=scipy.io.loadmat('LonDeg.mat')
        #print(data_LonDeg.keys())
        
        LonDeg=data_LonDeg['LonDeg']
        
        data_Mask=mat73.loadmat('landShelfOceanMask.mat')
        #print(data_Mask.keys())
        
        landShelfOcean_Lat=data_Mask['landShelfOcean_Lat']
        landShelfOcean_Lon=data_Mask['landShelfOcean_Lon']
        landShelfOceanMask=data_Mask['landShelfOceanMask']
        landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)
        
        data_proof=np.load("proof_no_noise.npz")
        proof=data_proof["proof"]
        data_obis=np.load("datos_obis.npz")
        
        mean_obis=data_obis["mean_obis"]
        std_obis=data_obis["obis_error"]
        ids_obis=data_obis["index"]
        ############################################
        #END OF LOADING DATA
        ############################################
        '''
        #Calls the rhonet_evo function to calculate the rho_shelf and K_shelf matrices.
        [rho_shelf,K_shelf, ext_index]=rhonet_evo(kfood,Kmin,food_shelf,temp_shelf,ext_pattern,Kmax_mean,spec_min_mean,spec_max_mean, Q10_mean,ext_intercept_shelf_mean,ext_slope_mean,shelf_lonlatAge,Point_timeslices[0], model)

     #print("YA")
       #Calls the alphadiv function to calculate the D_shelf matrix. If the model is "expo", it calls the alphadiv_expo function that does not include the effect of K on diversity.
        if model=="expo":
            D_shelf=alphadiv_expo(Point_timeslices,shelf_lonlatAge,rho_shelf,latWindow,lonWindow,LonDeg, ext_index)
        else:
            [rho_shelf_eff,D_shelf]=alphadiv(Point_timeslices,shelf_lonlatAge,rho_shelf,K_shelf,latWindow,lonWindow,LonDeg, ext_index)
               #Calls the inditek_gridMean_alphadiv function to calculate the grid that covers the earth surface and the mean of the diversity in each grid cell.
        [X,Y,D]=inditek_gridMean_alphadiv(D_shelf,shelf_lonlatAge,landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask)
        D_nan=D[~np.isnan(D)]
               #Calculates the rss (Residual Sum of Squares) comparing the model diversity with the observed diversity.
               #This function will be changed soon to include the new data from OBIS.
        [rss, residuals]=inditek_model_proof(D,proof) 

        return [rss, D_nan, D_shelf, residuals]
     #elapsed_time = time.time() - start_time
#
     #print(f"La función tardó {elapsed_time:.4f} segundos.")






