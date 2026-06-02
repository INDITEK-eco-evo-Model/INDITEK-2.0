import scipy.io
from rhonet import rhonet_evo
from alphadiv_huer import alphadiv
from gridMean import inditek_gridMean_alphadiv
import matplotlib.pyplot as plt
import mat73
import numpy as np


########################################
#If you want to plot the map from the proof_of_concept
########################################

#def plot_robinson(X, Y, Dinterp, title='Interpolated Data (Robinson)'):
#    fig = plt.figure(figsize=(12, 6))
#    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
#    ax.set_global()
#    #ax.coastlines()
#    ax.gridlines(draw_labels=False, color='gray', linewidth=0.5)
#
#    # Pseudocolor plot
#    mesh = ax.pcolormesh(X, Y, Dinterp, transform=ccrs.PlateCarree(), cmap='viridis', shading='auto')
#    plt.colorbar(mesh, orientation='horizontal', pad=0.05, label='Interpolated Value')
#
#    plt.title(title)
#    plt.savefig(f'alphadiv/{title}.png', dpi=300, bbox_inches='tight')
#    plt.show()

#It saves all the constants
kfood = 0.5
lonWindow = 2.5
latWindow = 2.5
ext_pattern = 4
ext_intercept_shelf_mean=0
ext_slope_mean=0

#Kmax_mean = 161
#Kmin = 19
#spec_min_mean = 0.002
#spec_max_mean = 0.035
#Q10_mean 1.75

#It saves all the parameters with the reference value
params_proposed = [161, 19, 0.035, 0.002, 2]

####################################################################
# Load the input data
####################################################################
data_food_temp=scipy.io.loadmat('data/Point_foodtemp.mat')

food_shelf=data_food_temp['food_shelf']
temp_shelf=data_food_temp['temp_shelf']


data_LonDeg=scipy.io.loadmat('data/LonDeg.mat')

LonDeg=data_LonDeg['LonDeg']

data_point_ages=scipy.io.loadmat('data/Point_ages_xyz')
Point_timeslices=data_point_ages['Point_timeslices'].astype(int)
shelf_lonlatAge=data_point_ages['shelf_lonlatAge']

data_Mask=mat73.loadmat('data/landShelfOceanMask.mat')

landShelfOcean_Lat=data_Mask['landShelfOcean_Lat']
landShelfOcean_Lon=data_Mask['landShelfOcean_Lon']
landShelfOceanMask=data_Mask['landShelfOceanMask']
landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)

#It calls the model, as it will serve as a reference, this is the proof model
model="proof"

#Calls the rhonet_evo function to calculate the rho_shelf (net diversification rate) and K_shelf (carrying capacity) matrices.
[rho_shelf,K_shelf, ext_index]=rhonet_evo(kfood,params_proposed[1],food_shelf,temp_shelf,ext_pattern,params_proposed[0],params_proposed[3],params_proposed[2], params_proposed[4],params_proposed[3],params_proposed[4],shelf_lonlatAge,Point_timeslices[0], model)

#Calls the alphadiv function to calculate the D (current diversity) and D_shelf(diversity through the years) matrix. 
[rho_shelf_eff,D_shelf]=alphadiv(Point_timeslices,shelf_lonlatAge,rho_shelf,K_shelf,latWindow,lonWindow,LonDeg, ext_index)

#If you want to plot the map, you call the function to plot it here
#for i in range(D.shape[2]):
#    Dinterp = D[:, :, i]
#    title = f'Capa {i} - Tiempo {Point_timeslices[0][i]} Ma'  # Ajusta según tus datos
#    plot_robinson(X, Y, Dinterp, title)

#Calls the inditek_gridMean_alphadiv function to calculate the grid that covers the earth surface and the mean of the diversity in each grid cell.

[X, Y, proof]=inditek_gridMean_alphadiv(D_shelf,shelf_lonlatAge,landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask)

#Here the random Gaussian error is created
mu=0
sigma=2

random_values = np.random.normal(loc=mu, scale=sigma, size=proof.shape)

#This error is added to the proof's data.
proof=proof+random_values

proof = np.clip(proof,1,None)

np.savez("data/proof_of_concept.npz", proof=proof)