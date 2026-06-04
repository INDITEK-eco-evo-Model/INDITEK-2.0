import numpy as np
import scipy.io
import mat73

###########################################
#USE FOR TESTING PURPOSES
###########################################

#data=np.load("datos_finales.npz")
#D_shelf = data["D_shelf"]
##print(D_shelf.shape)
#
#data=scipy.io.loadmat('Point_ages_xyzKocsisScotese_400.mat')
#shelf_lonlatAge=data['shelf_lonlatAge']
#
#data=mat73.loadmat('landShelfOceanMask_ContMargMaskKocsisScotese.mat')
#landShelfOcean_Lat=data['landShelfOcean_Lat']
#landShelfOcean_Lon=data['landShelfOcean_Lon']
#landShelfOceanMask=data['landShelfOceanMask']
#landShelfOceanMask = np.flip(landShelfOceanMask, axis=2)

##############################################

def inditek_gridMean_alphadiv(D_shelf,shelf_lonlatAge,landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask):


    #Create a 2D grid of latitude and longitude values using meshgrid. 

    [X,Y]=np.meshgrid(landShelfOcean_Lon,landShelfOcean_Lat)#


    #Edges of the latitude and longitude bins
    lat_edges=np.arange(-90,90.5,0.5)
    lon_edges=np.arange(-180,180.5,0.5)

    #Mask of the land shelf ocean (LSO) to ignore the land and ocean areas in the grid
    LSOmask=np.transpose(landShelfOceanMask[:,:,landShelfOceanMask.shape[2]-1])

    #Create the lat and lon arrays for the last time slice of the shelf_lonlatAge array. 
    lat=shelf_lonlatAge[:,shelf_lonlatAge.shape[1]-1,1]#
    lon=shelf_lonlatAge[:,shelf_lonlatAge.shape[1]-1,0]#
    d=D_shelf[:,D_shelf.shape[1]-1]#

    
    


    #Select and ignore the latitudes and longitudes that are not NaN (inactive points without diversity at 0Myr)
    lat=lat[np.isnan(d)==0]
    lon=lon[np.isnan(d)==0]  
    d=d[np.isnan(d)==0]

    


    #Linearize the latitudinal and longitudinal points to iterate over them in a loop.
    lat_idx=np.digitize(lat, lat_edges) - 1
    lon_idx=np.digitize(lon,lon_edges) - 1

    grid_idx= lat_idx * X.shape[1] + lon_idx

    #Create a 2D array of zeros with the same shape as the grid to store diversity values.
    D=np.zeros(X.shape)
    count=np.zeros(X.shape)

    #Iterate through all elements of the grid and add diversity valuesw to the corresponding grid cell.
    #The count array tracks the number of values added to each grid cell to calculate its mean diversity
    for i in range(len(grid_idx)):
        if LSOmask.flat[grid_idx[i]]==1:
            D.flat[grid_idx[i]] += d[i]
            count.flat[grid_idx[i]]+=1


    #Divide to calculate the mean diversity value for each cell, only for active points
    valid = count > 0
    D[valid] = D[valid] / count[valid]




    #Select only the points that are not 1 in the LSOmask (excluding land/ocean) and where diversity is greater than 0
    D[LSOmask!=1]=np.nan
    D[D==0]=np.nan

    return X, Y, D





