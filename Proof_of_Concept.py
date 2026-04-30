import scipy.io
from rhonet import rhonet_evo
from alphadiv_huer import alphadiv
from gridMean import inditek_gridMean_alphadiv
import matplotlib.pyplot as plt
import mat73
import numpy as np


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

kfood = 0.5
Kmin=19
lonWindow = 2.5
latWindow = 2.5
spec_min_mean = 0.002
ext_pattern = 4

#Kmax_mean = 161
#Kmin = 19
#spec_min_mean = 0.002
#spec_max_mean = 0.035
#Q10_mean 1.75

params_proposed = [161, 0.035, 2, 0, 0]

####################################################################
# Load the input data
####################################################################
data_food_temp=scipy.io.loadmat('data_input/Point_foodtemp.mat')

#print(data_food_temp.keys())
#food_ocean=data['food_ocean']
food_shelf=data_food_temp['food_shelf']
#temp_ocean=data['temp_ocean']
temp_shelf=data_food_temp['temp_shelf']


data_LonDeg=scipy.io.loadmat('data_input/LonDeg.mat')
#print(data_LonDeg.keys())

LonDeg=data_LonDeg['LonDeg']

data_point_ages=scipy.io.loadmat('data_input/Point_ages_xyz')#
#print(data_point_ages.keys())
#
Point_timeslices=data_point_ages['Point_timeslices'].astype(int)
#Point_timeslices = Point_timeslices[0]
shelf_lonlatAge=data_point_ages['shelf_lonlatAge']

data_Mask=mat73.loadmat('data_input/landShelfOceanMask.mat')
#print(data_Mask.keys())

landShelfOcean_Lat=data_Mask['landShelfOcean_Lat']
landShelfOcean_Lon=data_Mask['landShelfOcean_Lon']
landShelfOceanMask=data_Mask['landShelfOceanMask']
landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)

model="proof"

[rho_shelf,K_shelf, ext_index]=rhonet_evo(kfood,Kmin,food_shelf,temp_shelf,ext_pattern,params_proposed[0],spec_min_mean,params_proposed[1], params_proposed[2],params_proposed[3],params_proposed[4],shelf_lonlatAge,Point_timeslices[0], model)

[rho_shelf_eff,D_shelf]=alphadiv(Point_timeslices,shelf_lonlatAge,rho_shelf,K_shelf,latWindow,lonWindow,LonDeg, ext_index)

#[D,X,Y]=inditek_gridding_alphadiv(Point_timeslices[0],ext_pattern,rho_shelf,D_shelf,rho_shelf_eff,shelf_lonlatAge,landShelfOceanMask,landShelfOcean_Lat,landShelfOcean_Lon,params_proposed[0])

#print("D shape:", D.shape)
#for i in range(D.shape[2]):
#    Dinterp = D[:, :, i]
#    title = f'Capa {i} - Tiempo {Point_timeslices[0][i]} Ma'  # Ajusta según tus datos
#    plot_robinson(X, Y, Dinterp, title)

[X, Y, proof]=inditek_gridMean_alphadiv(D_shelf,shelf_lonlatAge,landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask)

mu=0
sigma=2

random_values = np.random.normal(loc=mu, scale=sigma, size=proof.shape)

proof=proof+random_values

proof = np.clip(proof,1,None)

np.savez("probita.npz", proof=proof)