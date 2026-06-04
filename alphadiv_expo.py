from scipy.io import loadmat
import numpy as np
from haversine_distance import haversine_distance


def dist_fun(shelf_lonlatAge, pos, step, point_pos, lim):
    neighbor_lonlat = shelf_lonlatAge[pos, step, 0:2] 

    dist_pos=haversine_distance(point_pos, neighbor_lonlat)#Calculate the distance to all points in the area

    lim=lim[dist_pos!=0]
    dist_pos=dist_pos[dist_pos!=0]##Remove the points that have a distance of 0 (the point itself)
    
    dist=dist_pos
    #Select the nearest point from all points in the area
    lim=lim[dist==min(dist)]

    return lim
    





def alphadiv_expo(Point_timeslices,shelf_lonlatAge,rho_shelf,latWindow,lonWindow,LonDeg, ext_index):

    pt=Point_timeslices#Position of 82 time slices in the 542 Myr (starting from 0 Ma (million years ago)+1=position 1) 
    pt=np.fliplr(pt).flatten()

    Point_timeslices=Point_timeslices[0]

    # 1. Calculate alpha diversity from points
    D0 = 1 # initialise diversity at time 541 MA with #1 genus area^(-1)
    D_shelf=np.full([shelf_lonlatAge.shape[0],542], np.nan)#Initial diversity matrix (n_pointsxn_timeslices)


    count=-1 #time frame resolved (MA) (there are 82 timeframes out of 542MA defined by the Point_timeslices)
    step=0 # 82 time frames (steps in the loop)
    ts2=Point_timeslices[0]+1 #next Point_timeslice (to fill the gap between both at each loop)


    for ts in Point_timeslices:#current Point_timeslice

        count += (ts2-ts)#Update the count variable

        #Get ages and active point positions from shelf data (lonlatAge dimensions: pointsxtimeframesx[longitude,latitude,age])
        ageS = shelf_lonlatAge[:, step, 2]
        posS=np.where(np.logical_and(~np.isnan(ageS), ageS>0))[0]

        
        



        # Initialize diversity for the first timeframe (ts == Point_timeslices[1])
        if ts == Point_timeslices[0]:

            D_shelf[posS, count] = D0 #Seed the coastal platform with 1 genus everywhere (to every active point) at time 541 Ma
        
        else:

            deltaAgeS = ageS[posS] - shelf_lonlatAge[posS, step - 1, 2] #(age at time ts) - (age at time ts-1)

        ############## Different kinds of points are treated a bit different to diversify:
        # #1# Handle newly inundated shelf points ##
        # Points that didn't exist or were not inundated in time t-1 and are now active

            


            pos1S = posS[np.logical_and(np.isnan(deltaAgeS), ageS[posS] <= ts2 - ts)]#Select the points that didn't exist at time t-1
            pos1S=np.concatenate((pos1S,posS[np.logical_and(shelf_lonlatAge[posS,step-1,2]==0,ageS[posS]<=ts2-ts)]))#Select the points that are 0 years old

            if pos1S.size > 0:#If there are points of this type, loop through them 1 by 1 to find its nearest neighbour from which receive diversity, mimicking inmigration

                for k in range(len(pos1S)): #Iterate over all points of this type


                    point_lonlat = shelf_lonlatAge[pos1S[k], step, [0,1]] # point location
                    
                    # Find points within the spatial window to initialize diversity from

                    lon_diff = np.abs(np.abs(point_lonlat[ 1]) - LonDeg[:, 0])  
                    f_diff = np.argmin(lon_diff) 

                    #Normalize the window size by the degree length at the point's latitude to maintain a constant window size in km
                    lon=lonWindow * LonDeg[f_diff,1]

                    #Logical conditions

                    lonMask = abs(shelf_lonlatAge[posS, step, 0] - point_lonlat[0]) <= lon
                    latMask = abs(shelf_lonlatAge[posS, step, 1] - point_lonlat[1]) <= latWindow



                    #Get the positions where both conditions are met

                    lim=np.where(lonMask & latMask)[0]
                    f=np.where(D_shelf[posS[lim],count2]>0)[0]
                    
                    lim = lim[f]

                    
                    if f.size > 0:
                        #Call dist_fun to calculate the distance to all points in the area and select the nearest neighboor
                        lim=dist_fun(shelf_lonlatAge, posS[lim], step, point_lonlat, lim)

                        d=np.nanmean(D_shelf[posS[lim], count2])#The diversity of the point of interest is the average of the diversity of the nearest points (excluding the point itself)

                        #Diversity always follows an exponential equation
                        d=max(D0,d+rho_shelf[pos1S[k],count2+1]*d)#Bounded by D0
                        D_shelf[pos1S[k],count2+1]=d
                    else:
                        D_shelf[pos1S[k], count2+1] = 0
                #It selects the points that are still 0 after the previous process (those that did not receive diversity from any neighbor)
                orphans=pos1S[D_shelf[pos1S, count2+1]==0]

                change=True#Initialize a flag to True to track changes; if no changes occur, it becomes False

                while_element=0
                
                #Iteratively search for neighbors of neighbors until all points have received diversity 
                #or are forced to be D0 because they are too far from any neighbor
                while orphans.size > 0 and change:

                    change=False
                    while_element+=1

                    
                    # Search for the nearest colonized point from the previous iteration to transmit diversity
                    for p_idx in orphans:

                        new_colonized = np.array(pos1S[D_shelf[pos1S, count2+1] > 0])

                        point_lonlat = shelf_lonlatAge[p_idx, step, [0,1]]
                        

                        if new_colonized.size>0:
                            c_coords = shelf_lonlatAge[new_colonized, step, 0:2]#Coordinates of the new colonized point
                            #If points exist, search for them within a 2.5-degree window
                            lim=np.where(np.logical_and(np.abs(c_coords[:,0]-point_lonlat[0])<=2.5*LonDeg[f_diff,1], np.abs(c_coords[:,1]-point_lonlat[1])<=2.5))[0]
                            f=np.where(D_shelf[new_colonized[lim],count2+1]>0)[0]
                            lim=lim[f]

                            
                            
                            # Select the nearest point among all on the search window
                            if lim.size>0:

                                lim=dist_fun(shelf_lonlatAge, new_colonized[lim], step, point_lonlat, lim)

                                idx_neighbor=new_colonized[lim][0]
                                
                                D_shelf[p_idx,count2+1]=max(D0,D_shelf[idx_neighbor,count2+1])
                                change=True
                        #Store the ones that continue without diversity
                        orphans=pos1S[D_shelf[pos1S, count2+1]==0]
               
                D_shelf[pos1S, count2+1] = np.maximum(D_shelf[pos1S, count2+1], D0) #force d to be at least D0, 1.
                    

            # #2# Special case of continental shelf points that did not exist in
            #time-1 and were artificially added in the Gplates model to fill gaps 
            # with age of nearest neighbour continental-shelf points (thus we start diversity with the diversity in the nearest continental shelf points) 

            pos2S=posS[np.logical_and(np.isnan(deltaAgeS),ageS[posS]>ts2-ts)] # point in time t-1 did not exist or was above land
            pos2S=np.concatenate((pos2S,posS[np.logical_and(shelf_lonlatAge[posS,step-1,2]==0,ageS[posS]>ts2-ts)]))

            
            if pos2S.size > 0:

             
                for k in range(len(pos2S)):
                    point_lonlat = [shelf_lonlatAge[pos2S[k], step, 0],shelf_lonlatAge[pos2S[k], step,1]] # point location

                    #Find points within the spatial window to initialize diversity from, using windows of 5, 10, 15 and 30 degrees to find the nearest neighbor
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

                    #Calculate the distance to all points in the area and select the one closest to the point of interest
                    lim=lim[f]

                    lim=dist_fun(shelf_lonlatAge, posS[lim], step, point_lonlat, lim)

                    #Calculate the diversity of the point of interest as the average diversity of the sorrounding points (excluding itself), same as the previous case
                    d=np.nanmean(D_shelf[posS[lim],count2])
                        
                    if d<D0:

                        d=D0 #Force d to be at least D0 (1.0)
              
                    #Calculate diversity following the exponential equation, bounded by D0
                    d=max(D0,d+rho_shelf[pos2S[k],count2+1]*d)
                    D_shelf[pos2S[k],count2+1]=d#Store it in the general matrix
                

             #3# Normal points
             
            

            pos3S=posS[np.logical_and(np.logical_and(deltaAgeS>0,np.round(deltaAgeS)<=ts2-ts),shelf_lonlatAge[posS,step-1,2]!=0)] #Exisiting points with normal behaviour continue to accumulate diversity




            #Bounded by D0 (1 genus area^(-1))

            
            D_shelf[pos3S, count2] = np.maximum(D_shelf[pos3S, count2], 1)
            

            d=D_shelf[pos3S,count2]

            d=np.maximum(D0,d)

            #Calculate diversity following the exponential equation, bounded by D0
            d=np.maximum(D0,d+rho_shelf[pos3S,count2+1]*d)
            D_shelf[pos3S,count2+1]=d
                
            d=np.maximum(d,D0)

            #All active points (the three kinds defined by N above) accumulate diversity over the time gap

            d=D_shelf[posS,count2+1]
            
            Myr=len(range(count2+2,count+1))# time gap to still diversify
            scaling=np.ones((d.size,1))
            scaling[np.isnan(deltaAgeS)==0]=(np.minimum(deltaAgeS[np.isnan(deltaAgeS)==0]-1,Myr)/Myr)[0] #for points that appeared mid-period and 
            #only accumulated diversity during their specific age gap
            rho=rho_shelf[:,count2+2:count+1]


            
            
            if np.any(np.isin(np.arange(count2 + 2, count+1), ext_index)): #For a period with any Myr with a extinction we need to sum Myr step-wise diversity

                
                for gap in range(count2+2,count+1):
                    
                    d=D_shelf[posS,gap-1]

                    d=np.maximum(d,D0)
                    #Calculate the effective diversification rate applying the exponential equation for each Myr step, bounded by D0#for a period with any Myr with a extinction we need to sum Myr step-wise diversity
                    d=np.maximum(D0,d+d*rho_shelf[posS,gap]*scaling.flatten())
                        
                    D_shelf[posS,gap]=d

                    
            

            
            
            else: #for a period without extinction, apply logistic growth as an exponential approach to
                # saturation over the Myr gap to skip the sum loop
                
                
                d=np.maximum(d,D0)
                d=d* np.exp(rho_shelf[posS,count]*Myr*scaling.flatten())

                d=np.maximum(d,D0)

                
                D_shelf[posS,count]=d 
    


        
        #Set the ts (time slice), count and step variables for the next iteration
        ts2=ts
        count2=count
        step = step + 1

        #Save positions of type 0 and type 1

        
    # Flip to order from point time slice 1 (0MA) to 542 (541MA)



    D_shelf=np.flip(D_shelf, axis=1)

    #get the 82 Point time slices for which the model is resolved (pt)

    D_shelf=D_shelf[:,pt]

    #% Flip back once the point time slices for which the model is resolved are compiled

    D_shelf=np.flip(D_shelf, axis=1)

    #To save the data in a .npz file for tests
    #np.savez("datos_comprobacion_alphadiv.npz", z=z, D_shelf=D_shelf, scaling=scaling)

    return D_shelf