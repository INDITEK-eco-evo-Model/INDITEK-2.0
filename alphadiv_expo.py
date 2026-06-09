from scipy.io import loadmat
import numpy as np
from haversine_distance import haversine_distance
 

def dist_fun(shelf_lonlatAge, pos, step, point_pos, lim):
    '''
    This function calculates the active nearest neighbours at time step within a given area (pos= active points within the search area of NN)
    '''

    neighbor_lonlat = shelf_lonlatAge[pos, step, 0:2] 

    # Calculate the distance to all the active points within the area defined by lonMask & latMask
    #  (being lim the position of the active points within the area)

    dist_pos=haversine_distance(point_pos, neighbor_lonlat)# Remove the points that have a distance of 0 (the point itself)


    lim=lim[dist_pos!=0]
    dist_pos=dist_pos[dist_pos!=0]##Remove the points that have a distance of 0 (the point itself)
    
    dist=dist_pos
    #Select the nearest point from all points in the area
    lim=lim[dist==min(dist)]

    return lim
    





def alphadiv_expo(Point_timeslices,shelf_lonlatAge,rho_shelf,latWindow,lonWindow,LonDeg, ext_index):

    '''
    Calculate the diversity following the differential equation:  ∂D/ ∂t = D ρ (1-D/K)

    PARAMETERS
    -----------
    Point_timeslices: array (1xn_time)
        Age of each time slice
    shelf_lonlatAge : array (n_pointsx n_time x 2)
        Longitude and latitude for each point through time
    rho_shelf: array (n_points x n_time)
        Net diversification rate values for each active point and time slice
    K_shelf: array (n_points x n_time)
        Carrying capacity values for each active point and time slice
    lonWindow: constant value to implement recolonization of newly submerged points from NN
        distance in degrees to search for particles from which diversity is "migrated" into the new coastal particles
    latWindow: constant value
        same but for latitude
    LonDeg: array (361x2)
        Degrees of longitud as a function of latitude (with a distance equivalent to 1º at the equator) to correct the size of the seach window for recolonization
    ext_index: array
        Index of mass extinction events (resolved to 1 Myr)


    RETURNS
    -----------
    D_shelf: array (n_points x n_time)
        Diversity values for each spatial point through time.
    '''

    pt=Point_timeslices# Position of 82 time slices in the 542 Myr (starting from 0 Ma (million years ago)+1=position 1) 
    pt=np.fliplr(pt).flatten() # Flip from 542 MA to 0  to accumulate diversity forward

    Point_timeslices=Point_timeslices[0] # flat the vector

    D0 = 1 # initialise diversity at time 541 MA with #1 genus at every active point within that 1sr time frame
    # Initialize diversity matrix
    D_shelf=np.full([shelf_lonlatAge.shape[0],542], np.nan) # (n_pointsxn_timeslices)


    count=-1 # time frame resolved (MA) (there are 82 timeframes out of 542MA defined by the Point_timeslices)
    step=0 # 82 time frames (steps in the loop)
    # 82 time frames (steps in the loop)
    # ts2: next timeframe after ts
    # in order to fill the gap between both at each loop 
    # (the model accumulates diversity every Myr and points can activate at any time within the gap from ts to ts2)
    ts2=Point_timeslices[0]+1


    for ts in Point_timeslices:# current Point_timeslice

        count += (ts2-ts)# Update the count variable to track the 82 resolved time slices

        #Get ages of points in the time slice (time submerged until the step time slice)
        ageS = shelf_lonlatAge[:, step, 2]

        # active point positions from shelf data (points that exist=are submerged, within the time gap (ageS>0))
        posS=np.where(np.logical_and(~np.isnan(ageS), ageS>0))[0]


        # Initialize diversity for the first timeframe (ts == Point_timeslices[0])
        if ts == Point_timeslices[0]:

            D_shelf[posS, count] = D0 # Seed the coastal platform with 1 genus everywhere (to every active point) at time 541 Ma
        
        else:

            deltaAgeS = ageS[posS] - shelf_lonlatAge[posS, step - 1, 2] # (age at time ts) - (age at time ts-1) to get the Myr that the point is active during the time gap)

        ############## Different kinds of points are treated a bit different to diversify:
        # (the paleotectonic model provides points of different nature, e.g., occasionally points made be added for density reasons)

        # #1# Handle newly inundated shelf points ##

        # Points that didn't exist or were not inundated in time t-1 and are now active

            pos1S = posS[np.logical_and(np.isnan(deltaAgeS), ageS[posS] <= ts2 - ts)] # Select the points that didn't exist at time t-1
            pos1S=np.concatenate((pos1S,posS[np.logical_and(shelf_lonlatAge[posS,step-1,2]==0,ageS[posS]<=ts2-ts)])) # Select the points that were 0 years old at time step -1

            if pos1S.size > 0:# If there are points of this type, loop through them 1 by 1 to find its nearest neighbour (active NN) from which being recolonized, mimicking inmigration after newly inundation

                for k in range(len(pos1S)):# Iterate over all points of this type


                    point_lonlat = shelf_lonlatAge[pos1S[k], step, [0,1]] # point location
                    
                    # Find the latitude band of the point

                    lon_diff = np.abs(np.abs(point_lonlat[ 1]) - LonDeg[:, 0])  
                    f_diff = np.argmin(lon_diff) 

                    # Normalize the search window size in degrees according to the latitude to maintain a constant window size in km
                    lon=lonWindow * LonDeg[f_diff,1]

                    # Find points within the spatial window to initialize diversity, mimicking recolonization

                    lonMask = abs(shelf_lonlatAge[posS, step, 0] - point_lonlat[0]) <= lon
                    latMask = abs(shelf_lonlatAge[posS, step, 1] - point_lonlat[1]) <= latWindow

                    lim=np.where(lonMask & latMask)[0]

                    #Select the index of the active points
                    f=np.where(D_shelf[posS[lim],count2]>0)[0]
                    
                    lim = lim[f]

                    # Find and import NN diversity within the selected points
                    if f.size > 0:
                        #Call dist_fun to calculate the distance to all points in the area and select the nearest neighboor
                        lim=dist_fun(shelf_lonlatAge, posS[lim], step, point_lonlat, lim)
                        
                        # The diversity of the point of interest is the average of the diversity of the nearest points
                        #  (excluding the point itself) bounded by the carring capacity of the point (mimicking local extinction)
                        d=np.nanmean(D_shelf[posS[lim], count2])#The diversity of the point of interest is the average of the diversity of the nearest points (excluding the point itself)

                        #####Apply differential equation to calculate diversity within the gap (count2 Myr)


                        #The diversity is calculated using the logistic equation
                        d=max(D0,d+rho_shelf[pos1S[k],count2+1]*d)#bounded by D0
                        D_shelf[pos1S[k],count2+1]=d
                    else:
                        D_shelf[pos1S[k], count2+1] = 0 # points for which we haven't found active NN (orphans)
                
                # Define the position of the orphans to recolonize from the newly colonized points (iterative procedure until all points are recolonized or are too isolated)
                orphans=pos1S[D_shelf[pos1S, count2+1]==0]

                change=True# Initialize a flag = True to track changes; if no changes occur, flag = False

                while_element=0
                
                # Iteratively search for neighbors of neighbors until all points have received diversity 
                # or are forced to be D0 because they are too far from any neighbor
                while orphans.size > 0 and change:

                    change=False
                    while_element+=1

                    
                    # Search for the nearest colonized point from the previous iteration to transmit diversity
                    for p_idx in orphans:

                        new_colonized = np.array(pos1S[D_shelf[pos1S, count2+1] > 0])

                        point_lonlat = shelf_lonlatAge[p_idx, step, [0,1]]
                        
                        # If there are any points it selects those within the search window
                        if new_colonized.size>0:
                            c_coords = shelf_lonlatAge[new_colonized, step, 0:2]#Coordinates of the new colonized point
                            # search for them within the search window
                            lim=np.where(np.logical_and(np.abs(c_coords[:,0]-point_lonlat[0])<=2.5*LonDeg[f_diff,1], np.abs(c_coords[:,1]-point_lonlat[1])<=2.5))[0]
                            f=np.where(D_shelf[new_colonized[lim],count2+1]>0)[0]
                            lim=lim[f]

                            
                            
                            # Select the active NN
                            if lim.size>0:

                                lim=dist_fun(shelf_lonlatAge, new_colonized[lim], step, point_lonlat, lim)

                                idx_neighbor=new_colonized[lim][0]
                                
                                D_shelf[p_idx,count2+1]=max(D0,D_shelf[idx_neighbor,count2+1])#colonize bounded by D0
                                change=True
                        orphans=pos1S[D_shelf[pos1S, count2+1]==0]
               
                D_shelf[pos1S, count2+1] = np.maximum(D_shelf[pos1S, count2+1], D0) # set orphans to D0
                    

            # #2# Special case of continental shelf points that did not exist in time-1
            # and were artificially added by the palaeotectonics model for density reasons
            # thus the age assinged from their nearest continental-shelf points
            # and accordingly  we assign them the diversity accumulated by its active neighbour until the current time step 

            pos2S=posS[np.logical_and(np.isnan(deltaAgeS),ageS[posS]>ts2-ts)] # point in time t-1 did not exist and have been assigned an age greater than the time gap
            pos2S=np.concatenate((pos2S,posS[np.logical_and(shelf_lonlatAge[posS,step-1,2]==0,ageS[posS]>ts2-ts)]))

            
            if pos2S.size > 0:
             
                for k in range(len(pos2S)):

                    ###### Find nearest point to import diversity in an iterative search window

                    point_lonlat = [shelf_lonlatAge[pos2S[k], step, 0],shelf_lonlatAge[pos2S[k], step,1]] # point location

                    # Find points within the spatial window to initialize diversity from, using windows of 5, 10, 15 and 30 degrees to find the nearest neighbor
                    lim=np.where(np.logical_and(np.abs(shelf_lonlatAge[posS,step,0]-point_lonlat[0])<=5, np.abs(shelf_lonlatAge[posS,step,1]-point_lonlat[1]<=5)))[0]
                    f=np.where(D_shelf[posS[lim],count2]>0)[0]
                    if f.size==0:
                        lim=np.where(np.logical_and(np.abs(shelf_lonlatAge[posS,step,0]-point_lonlat[0])<=10,np.abs(shelf_lonlatAge[posS,step,1]-point_lonlat[1]<=10)))[0]                        
                        f=np.where(D_shelf[posS[lim],count2]>0)[0]
                    if f.size==0:
                        lim=np.where(np.logical_and(np.abs(shelf_lonlatAge[posS,step,0]-point_lonlat[0])<=15, np.abs(shelf_lonlatAge[posS,step,1]-point_lonlat[1]<=15)))[0]
                        f=np.where(D_shelf[posS[lim],count2]>0)[0]
                    if f.size==0:
                        lim=np.where(np.logical_and(np.abs(shelf_lonlatAge[posS,step,0]-point_lonlat[0]+5)<=30, np.abs(shelf_lonlatAge[posS,step,1]-point_lonlat[1]<=30)))[0]
                        f=np.where(D_shelf[posS[lim],count2]>0)[0]

                    # Calculate the distance to all points in the area and select the active NN
                    lim=lim[f]

                    lim=dist_fun(shelf_lonlatAge, posS[lim], step, point_lonlat, lim)

                    #Calculate the diversity of the point of interest as the average diversity of the NN points if there are more than one, same as the previous case
                    d=np.nanmean(D_shelf[posS[lim],count2])
                        
                    if d<D0:

                        d=D0 #Force d to be at least D0 (1.0)
              
                    ##### Apply differential equation to calculate diversity, same as before

                    d=max(D0,d+rho_shelf[pos2S[k],count2+1]*d)# bounded by D0
                    D_shelf[pos2S[k],count2+1]=d
                

            #3# Normal points: already active and diversifying
             
            pos3S=posS[np.logical_and(np.logical_and(deltaAgeS>0,np.round(deltaAgeS)<=ts2-ts),shelf_lonlatAge[posS,step-1,2]!=0)] #Exisiting points with normal behaviour continue to accumulate diversity



            # Bound by D0 

            
            D_shelf[pos3S, count2] = np.maximum(D_shelf[pos3S, count2], D0)
            

            d=D_shelf[pos3S,count2]

            d=np.maximum(D0,d)

            #Apply differential equation to continue accumulating diversity 
            d=np.maximum(D0,d+rho_shelf[pos3S,count2+1]*d)
            D_shelf[pos3S,count2+1]=d

            # All active points (the three kinds defined by N above) once assigned d at time count+1, accumulate diversity every Myr over the time gap (count2+2,count+1):

            d=D_shelf[posS,count2+1]
            
            Myr=len(range(count2+2,count+1)) # time gap to still diversify
            scaling=np.ones((d.size,1))
            #for points that appeared mid-period and 
            # normalise time for diversification within the time gap
            # according to their actual life time (deltaAgeS) to account for cases of points created within the gap
            scaling[np.isnan(deltaAgeS)==0]=(np.minimum(deltaAgeS[np.isnan(deltaAgeS)==0]-1,Myr)/Myr)[0] 


            
            
            if np.any(np.isin(np.arange(count2 + 2, count+1), ext_index)): # case with extinction inside the gap
                # to avoid logistic function during the Myrs of extinction 
                
                for gap in range(count2+2,count+1):
                    
                    d=D_shelf[posS,gap-1]

                    d=np.maximum(d,D0) # bound by D0

                    #use the logistic equation for all the Myrs
                    d=np.maximum(D0,d+d*rho_shelf[posS,gap]*scaling.flatten())
                        
                    D_shelf[posS,gap]=d

                    
            

            
            
            else: #for a period without extinction, apply logistic growth with an exponential approach to skip the sum loop
                
                
                
                d=np.maximum(d,D0)
                d=d* np.exp(rho_shelf[posS,count]*Myr*scaling.flatten())

                d=np.maximum(d,D0)

                
                D_shelf[posS,count]=d 
    


        
        #Set the ts (time slice), count and step variables for the next iteration
        ts2=ts
        count2=count
        step = step + 1

        #Save positions of type 0 and type 1

        
    # Flip to order from point time slice 1 (0MA) to 542 (541MA) to match pt
    D_shelf=np.flip(D_shelf, axis=1)
    # Get the 82 Point time slices for which the model is resolved (pt position of -Myr time slices)
    D_shelf=D_shelf[:,pt]

    # Flip back once the point time slices for which the model is resolved are compiled
    D_shelf=np.flip(D_shelf, axis=1)

    #To save the data in a .npz file for tests
    #np.savez("datos_comprobacion_alphadiv.npz", z=z, D_shelf=D_shelf, scaling=scaling)

    return D_shelf