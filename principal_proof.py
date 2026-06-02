import scipy.io
import mat73
import numpy as np
import pandas as pd
import time
from rhonet import rhonet_evo
from alphadiv import alphadiv 
from gridMean import inditek_gridMean_alphadiv
from inditek_model_proof import inditek_model_proof

start_time = time.time()

def principal(kfood, Kmin, food_shelf, temp_shelf, ext_pattern, Kmax_mean, spec_min_mean, spec_max_mean, Q10_mean, ext_intercept_shelf_mean,ext_slope_mean, shelf_lonlatAge, Point_timeslices, latWindow,lonWindow,LonDeg, landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask, proof):
#
#############################################################################################
# JUST FOR TESTING PURPOSES
################################################################################################
##
     ##CHOOSE model parameters:
     #kfood = 0.5 #[POC mol * m-2 yr-1] #1
     #spec_min_mean = 0.00139 #[MA-1]   #0.1
     #spec_max_mean = 0.03499 #[MA-1]   #la sacas
     #Q10_mean = 1.69 #n.u.   #la sacas 
     ### Carrying capacity of #genera at maximum food availability #la sacas  
     #Kmin=18.87# Carrying capacity of #genera at minimum food availability #10
     #ext_slope_mean=0 #la sacas
     #ext_intercept_shelf_mean=0 #la sacas
     ##
     #latWindow=2.5 #2.5
     #lonWindow=2.5 #2.5
     ##
     #ext_pattern=4 #3
     ##
     #Kmax_mean=161.14
     ##
     #data_food_temp=scipy.io.loadmat('data/Point_foodtemp.mat')
     ##
     ###food_ocean=data['food_ocean']
     #food_shelf=data_food_temp['food_shelf']
     ###temp_ocean=data['temp_ocean']
     #temp_shelf=data_food_temp['temp_shelf']
     ##
     #data_point_ages=scipy.io.loadmat('data/Point_ages_xyz.mat')#
     ###print(data_point_ages.keys())
     ###
     #Point_timeslices=data_point_ages['Point_timeslices'].astype(int)
     ###Point_timeslices=Point_timeslices[0]
     ###Point_timeslices=Point_timeslices[0]
     #shelf_lonlatAge=data_point_ages['shelf_lonlatAge']
     ##
     #data_LonDeg=scipy.io.loadmat('data/LonDeg.mat')
     ###print(data_LonDeg.keys())
     ##
     #LonDeg=data_LonDeg['LonDeg']
     ##
     #data_Mask=mat73.loadmat('data/landShelfOceanMask.mat')
     ###print(data_Mask.keys())
     ##
     #landShelfOcean_Lat=data_Mask['landShelfOcean_Lat']
     #landShelfOcean_Lon=data_Mask['landShelfOcean_Lon']
     #landShelfOceanMask=data_Mask['landShelfOceanMask']
     #landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)
     ##
     ##data_obis=np.load("datos_obis.npz")
     ###
     ##mean_obis=data_obis["mean_obis"]
     ##std_obis=data_obis["obis_error"]
     ##ids_obis=data_obis["index"]
     ##
     #data_proof=np.load("data/observed_D.npz")
     #proof=data_proof[ "proof"]
     ##indices=np.load("index_points.npz")
     ##indices_pac=indices["indices_pac"]
     ##indices_med=indices["indices_med"]
     ##indices_car=indices["indices_car"]
     ##############################################
     ###END OF LOADING DATA
     ##############################################

     #Calls the rhonet_evo function to calculate the rho_shelf and K_shelf matrices.

     [rho_shelf,K_shelf, ext_index]=rhonet_evo(kfood,Kmin,food_shelf,temp_shelf,ext_pattern,Kmax_mean,spec_min_mean,spec_max_mean, Q10_mean,ext_intercept_shelf_mean,ext_slope_mean,shelf_lonlatAge,Point_timeslices)
                     
     #Calls the alphadiv function to calculate the D_shelf matrix.

     [rho_shelf_eff,D_shelf]=alphadiv(Point_timeslices,shelf_lonlatAge,rho_shelf,K_shelf,latWindow,lonWindow,LonDeg, ext_index)

     #Calls the inditek_gridMean_alphadiv function to calculate the grid that covers the earth surface and the mean of the diversity in each grid cell.
     [X, Y, D]=inditek_gridMean_alphadiv(D_shelf,shelf_lonlatAge,landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask)

     D_nan=D[~np.isnan(D)]

     #Calculates the rss (Residual Sum of Squares) comparing the model diversity with the observed diversity.
     #This function will be changed soon to include the new data from OBIS.

     rss=inditek_model_proof(D,proof)





     return rss, D_nan
     #elapsed_time = time.time() - start_time


     #print(f"La función tardó {elapsed_time:.4f} segundos.")

#np.savez("principal_div.npz", D_shelf=D_shelf)






