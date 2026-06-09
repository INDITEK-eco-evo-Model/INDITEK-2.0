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

def inditek_gridding_alphadiv(D_shelf,shelf_lonlatAge,landShelfOcean_Lat,landShelfOcean_Lon, landShelfOceanMask):


    '''
    Converts point-based alpha diversity values into a regular lat-lon grid

    PARAMETERS
    -----------
    D_shelf: array (n_points x n_time)
        Diversity values for each spatial point through time.
    shelf_lonlatAge : array (n_pointsx n_time x 2)
        Longitude and latitude for each point through time
    landShelfOcean_Lat / landShelfOcean_Lon _ 1D arrays
        Latitude and Longitude coordinates defining the target grid.
    landShelfOceanMask : 3D array (lat x lon x time)
        Mask defining valid shelf areas (1=shelf, 0=land, 2 = ocean)

    RETURNS
    -----------
    X, Y: 2D arrays
        Meshgrid of longitude and latitude
    D : 2D array
        Gridded alpha diversity values (mean per cell)
    '''

    # X and Y represent the spatial grid where diversity will be mapped. 
    # They are created using the meshgrid function, which generates coordinate matrices from the latitude and longitude vectors.
    [X,Y]=np.meshgrid(landShelfOcean_Lon,landShelfOcean_Lat)#


    # Define a global grid at 0.5º resolution to assign the diversity values to the corresponding grid cells
    lat_edges=np.arange(-90,90.5,0.5)
    lon_edges=np.arange(-180,180.5,0.5)

    #Extract mask for the most recent time slice 
    # Mask of the land shelf ocean (LSO) to ignore the land and ocean areas in the grid
    LSOmask=np.transpose(landShelfOceanMask[:,:,landShelfOceanMask.shape[2]-1])

    # Extract point coordinates and diversity at last time slice. 
    lat=shelf_lonlatAge[:,shelf_lonlatAge.shape[1]-1,1]#
    lon=shelf_lonlatAge[:,shelf_lonlatAge.shape[1]-1,0]#
    d=D_shelf[:,D_shelf.shape[1]-1]#

    
    


    # Only keep points that have valid diversity values (not NaN) to avoid issues during gridding.
    lat=lat[np.isnan(d)==0]
    lon=lon[np.isnan(d)==0]  
    d=d[np.isnan(d)==0]

    


    # Assign each point to a grid cell based on its latitude and longitude using the digitize function, 
    # which return the indices of the bins to which each value belongs 
    lat_idx=np.digitize(lat, lat_edges) - 1
    lon_idx=np.digitize(lon,lon_edges) - 1
    # Convert 2D indices to 1D index for flattened arrays
    grid_idx= lat_idx * X.shape[1] + lon_idx

    # Initialize 2D array of zeros with the grid shape to store diversity values.
    D=np.zeros(X.shape)
    count=np.zeros(X.shape)

    # Iterate through all elements of the grid and compute mean point diversity valuesw to the corresponding grid cell.
    #The count array tracks the number of values added to each grid cell to calculate its mean diversity
    for i in range(len(grid_idx)):
        if LSOmask.flat[grid_idx[i]]==1:
            D.flat[grid_idx[i]] += d[i]
            count.flat[grid_idx[i]]+=1


    # Compute mean diversity per grid cell by dividing the total diversity by the count of active points in each cell
    valid = count > 0
    D[valid] = D[valid] / count[valid]




    # Remove non-shelf areas and grids with all points having 0 values (innactive)
    D[LSOmask!=1]=np.nan
    D[D==0]=np.nan

    return X, Y, D





