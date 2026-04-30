from scipy.io import loadmat
import numpy as np
from haversine_distance import haversine_distance


def dist_fun(shelf_lonlatAge, pos, step, point_pos, lim):
    neighbor_lonlat = shelf_lonlatAge[pos, step, 0:2] 
    #neighbor_temp=temp_shelf[pos, step]

    dist_pos=haversine_distance(point_pos, neighbor_lonlat)#Calculate the distance to all the points in the area
    #diff_temps=abs(neighbor_temp-point_temp)

    lim=lim[dist_pos!=0]
    dist_pos=dist_pos[dist_pos!=0]##Remove the points that are 0 distance (the point itself)
    #Mdist_temp=max(diff_temps)
#
    #if Mdist_temp==0:
    #    Mdist_temp=1
        
    #dist_temps=(1+diff_temps/Mdist_temp)
    dist=dist_pos#*dist_temps
    lim=lim[dist==min(dist)]

    return lim
    





def alphadiv(Point_timeslices,shelf_lonlatAge,rho_shelf,K_shelf,latWindow,lonWindow,LonDeg, ext_index):#, missDisp):

    pt=Point_timeslices# position of 82 time slices in the 542 Myr (starting from 0 Ma (million tears ago)+1=position 1) to retrieve only that info from the final data matrix
    #print(pt)
    pt=np.fliplr(pt).flatten()

    Point_timeslices=Point_timeslices[0]

    save_pos1S=[]
    save_pos2S=[]
    D0_forced=[]
    num_while=[]
    # 1. Calculate alpha diversity from points
    D0 = 1 # initialise diversity at time 541 MA with #1 genus area^(-1)
    D_shelf=np.full([shelf_lonlatAge.shape[0],542], np.nan)
    rho_shelf_eff=np.full([shelf_lonlatAge.shape[0],542], np.nan)


    count=-1 #time frame resolved (MA) (there are 82 timeframes out of 542MA defined by the Point_timeslices)
    step=0 # 82 time frames (steps in the loop)
    ts2=Point_timeslices[0]+1 #next timeframe after ts (to fill the gap between both at each loop)


    for ts in Point_timeslices:

        #print("ts: "+str(ts))

        count += (ts2-ts)#Update the count variable

        #Get ages and active point positions from shelf data (lonlatAge dimensions: pointsxtimeframesx[longitude,latitude,age])
        ageS = shelf_lonlatAge[:, step, 2]
        posS=np.where(np.logical_and(~np.isnan(ageS), ageS>0))[0]

        
        



        # Initialize diversity for the first timeframe (ts == Point_timeslices(1))
        if ts == Point_timeslices[0]:
            pos1S=[]
            pos2S=[]
            D0_element=[]
            D_shelf[posS, count] = D0 #seed the coastal platform with 1 genus everywhere (to every active point) at time 541Ma
        else:

            deltaAgeS = ageS[posS] - shelf_lonlatAge[posS, step - 1, 2] #(age at time ts) - (age at time ts-1)

        ############## different kinds of points are treated a bit different to diversify:
        # #1# Handle newly inundated shelf points ##
        # Points that didn't exist or were not inundated in time t-1 and are now active

            D0_element=[]
            


            pos1S = posS[np.logical_and(np.isnan(deltaAgeS), ageS[posS] <= ts2 - ts)]#Selects the points that didn't exist at time t-1
            pos1S=np.concatenate((pos1S,posS[np.logical_and(shelf_lonlatAge[posS,step-1,2]==0,ageS[posS]<=ts2-ts)]))#Select the points that are 0 years old

            if pos1S.size > 0:#If there are points of this type go through them 1 by 1 to find its nearest neighbour from which receive diversity mimicking inmigration

                for k in range(len(pos1S)): #Iterate over all points of this type


                    point_lonlat = shelf_lonlatAge[pos1S[k], step, [0,1]] # point location
                    #point_temp=temp_shelf[pos1S[k], step]

                    # Find points within the spatial window to initialize diversity from

                    lon_diff = np.abs(np.abs(point_lonlat[ 1]) - LonDeg[:, 0])  
                    f_diff = np.argmin(lon_diff) 

                    lon=lonWindow * LonDeg[f_diff,1]

                    #logical conditions

                    lonMask = abs(shelf_lonlatAge[posS, step, 0] - point_lonlat[0]) <= lon
                    latMask = abs(shelf_lonlatAge[posS, step, 1] - point_lonlat[1]) <= latWindow



                    #Get the position for both conditions combined:

                    lim=np.where(lonMask & latMask)[0]
                    f=np.where(D_shelf[posS[lim],count2]>0)[0]
                    

                    #if f.size==0:
                    #    lim=np.where(np.logical_and(np.abs(shelf_lonlatAge[posS,step,0]-point_lonlat[0])<=3*LonDeg[f_diff,1], np.abs(shelf_lonlatAge[posS,step,1]-point_lonlat[1]<=3)))[0] # No point with accumulated diversity found, initialize with D0
                    #    f=np.where(D_shelf[posS[lim],count2]>0)[0]

                        #ipdb.set_trace()

                        #ipdb.set_trace()
                    lim = lim[f]

                    
                    if f.size > 0:
                        #if f.size> 1:
                        #       ipdb.set_trace()
                        #If there are points with diversity in the area

                        lim=dist_fun(shelf_lonlatAge, posS[lim], step, point_lonlat, lim)

                        #neighbor_lonlat = shelf_lonlatAge[posS[lim], step, 0:2]  
                        #neighbor_temp=temp_shelf[posS[lim], step]

                        #dist_pos=haversine_distance(point_lonlat, neighbor_lonlat)#Calculate the distance to all the points in the area
                        #diff_temps=abs(neighbor_temp-point_temp)

                        #lim=lim[dist_pos!=0]
                        #dist_pos=dist_pos[dist_pos!=0]##Remove the points that are 0 distance (the point itself)
                        #Mdist_temp=max(diff_temps)

                        #if Mdist_temp==0:
                        #    Mdist_temp=1
                        #dist_temps=(1+diff_temps/Mdist_temp)

                        #dist=dist_temps*dist_pos


                        #lim=lim[dist==min(dist)]#Select the point with the minimum distance to the point of interest

                        d=min(np.nanmean(D_shelf[posS[lim], count2]),K_shelf[pos1S[k],step])#The diversity of the point of interest is the average of the diversity of the points in the area (the ones that are not 0 distance)
                        #force local extinction if imported diversity is greater than K.

                        #ipdb.set_trace()

                        if count2+1 in ext_index:
                            d=max(D0,d+rho_shelf[pos1S[k],count2+1]*d)#bounded by D0
                            D_shelf[pos1S[k],count2+1]=min(K_shelf[pos1S[k],step],d)#bounded by K_shelf (The carrying capacity)

                        else: # normal diversification period

                            d=min(K_shelf[pos1S[k],step],d+rho_shelf[pos1S[k],count2+1]*d*(max(0,1-(d/K_shelf[pos1S[k],step])))) 
                            D_shelf[pos1S[k],count2+1]=max(D0,d)
                            #ipdb.set_trace()
                    else:
                        D_shelf[pos1S[k], count2+1] = 0
                    
                huerfanos=pos1S[D_shelf[pos1S, count2+1]==0]
                D0_element=huerfanos

                cambio=True

                while_element=0
                

                while huerfanos.size > 0 and cambio:

                    cambio=False
                    while_element+=1

                    

                    for p_idx in huerfanos:

                        colonizados_hoy = np.array(pos1S[D_shelf[pos1S, count2+1] > 0])

                        point_lonlat = shelf_lonlatAge[p_idx, step, [0,1]]
                        #point_temp=temp_shelf[p_idx, step]

                        
                        #if step==47 and round(point_lonlat[1])==15:
                        #    ipdb.set_trace()
                        

                        if colonizados_hoy.size>0:
                            c_coords = shelf_lonlatAge[colonizados_hoy, step, 0:2]
                            lim=np.where(np.logical_and(np.abs(c_coords[:,0]-point_lonlat[0])<=2.5*LonDeg[f_diff,1], np.abs(c_coords[:,1]-point_lonlat[1])<=2.5))[0]
                            f=np.where(D_shelf[colonizados_hoy[lim],count2+1]>0)[0]
                            lim=lim[f]

                            
                            #if step==47 and round(point_lonlat[1])==15:
                            #    ipdb.set_trace()

                            if lim.size>0:

                                lim=dist_fun(shelf_lonlatAge, colonizados_hoy[lim], step, point_lonlat, lim)

                                #neighbor_lonlat = shelf_lonlatAge[colonizados_hoy[lim], step, 0:2]
                                #neighbor_temp=temp_shelf[colonizados_hoy[lim], step]

                                #dist_pos=haversine_distance(point_lonlat, neighbor_lonlat)
                                #diff_temps=abs(neighbor_temp-point_temp)

                                #Mdist_temp=max(diff_temps)

                                #lim=lim[dist_pos!=0]
                                #dist_pos=dist_pos[dist_pos!=0]

                                #if Mdist_temp==0:
                                #    Mdist_temp=1
                                #dist_temps=(1+diff_temps/Mdist_temp)

                                #dist=dist_temps*dist_pos

                                #lim=lim[dist==min(dist)]

                                vecino_idx=colonizados_hoy[lim][0]
                                #print(p_idx)
                                #print(vecino_idx)
                                
                                D_shelf[p_idx,count2+1]=max(D0,D_shelf[vecino_idx,count2+1])
                                cambio=True
                        huerfanos=pos1S[D_shelf[pos1S, count2+1]==0]
               
                D_shelf[pos1S, count2+1] = np.maximum(D_shelf[pos1S, count2+1], D0) #force d to be at least D0, 1.
                num_while.append(while_element)


                    # diversification keeping d in between D0 and local K bounds
                    #if np.logical_and(len(np.unique(rho_shelf[:,count2+1]))==1, np.all(np.unique(rho_shelf[:,count2+1]))<0): # extinction period
                    
                        
                    
                #ipdb.set_trace()
                    
                #ipdb.set_trace()

            #print(pos1S)
            #print(pos1S.shape)
            #input("pause")
                    

            # #2# Special case of continental shelf points that did not exist in
            #time-1 and were artificially added in the Gplates model to fill gaps 
            # with age of nearest neighbour continental-shelf points (thus we start diversity with the diversity in the nearest continental shelf points) 

            pos2S=posS[np.logical_and(np.isnan(deltaAgeS),ageS[posS]>ts2-ts)] # point in time t-1 did not exist or was above land
            pos2S=np.concatenate((pos2S,posS[np.logical_and(shelf_lonlatAge[posS,step-1,2]==0,ageS[posS]>ts2-ts)]))

           
            #print("Del tipo 2 hay"+str(len(pos2S)))
            
            if pos2S.size > 0:

                #print("Esta en paso 2")
                for k in range(len(pos2S)):
                    point_lonlat = [shelf_lonlatAge[pos2S[k], step, 0],shelf_lonlatAge[pos2S[k], step,1]] # point location
                    #point_temp=temp_shelf[pos2S[k], step]

                    #   Find points within the spatial window to initialize diversity from, the spatial window has a length of 5, 10, 15 and 30 degrees to find the nearest neighbour
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

                    #It calculates the distance to all the points in the area and selects the one with the minimum distance to the point of interest
                    lim=lim[f]

                    lim=dist_fun(shelf_lonlatAge, posS[lim], step, point_lonlat, lim)
                    #neighbor_lonlat = shelf_lonlatAge[posS[lim], step, 0:2]  # forma (N, 2)
                    #dist=haversine_distance(point_lonlat, neighbor_lonlat)
#
                    #lim=lim[dist!=0]
                    #dist=dist[dist!=0]
#
                    #lim = lim[np.isclose(dist, dist.min())]

                    

                    #Then, it calculates the diversity of the point of interest as the average of the diversity of the points in the area (the ones that are not 0 distance), equal to the previous case
                    d=min(np.nanmean(D_shelf[posS[lim],count2]),K_shelf[pos2S[k],step])
                        
                    if d<D0:

                        d=D0 #force d to be at least D0, 1.
                        

                        # diversification keeping d in between D0 and local K bounds
                        #equal to the previous case
                    if count2+1 in ext_index:# extinction period

                        d=max(D0,d+rho_shelf[pos2S[k],count2+1]*d)#bounded by D0
                        D_shelf[pos2S[k],count2+1]=min(K_shelf[pos2S[k],step],d)#bounded by K_shelf (The carrying capacity)
                
                    else: # normal diversification period

                        d=np.fmin(K_shelf[pos2S[k],step],d+rho_shelf[pos2S[k],count2+1]*d*(max(0,1-(d/K_shelf[pos2S[k],step])))) 
                        D_shelf[pos2S[k],count2+1]=max(D0,d) 

                #ipdb.set_trace()

            

             #3# Normal points
             
            

            pos3S=posS[np.logical_and(np.logical_and(deltaAgeS>0,np.round(deltaAgeS)<=ts2-ts),shelf_lonlatAge[posS,step-1,2]!=0)] #exisiting points with normal behaviour continue to accumulate diversity




            #boundaries between the carrying capacity (K_shelf) and D0 (1 genus area^(-1))

            
            D_shelf[pos3S, count2] = np.maximum(D_shelf[pos3S, count2], 1)
            

            d=np.minimum(D_shelf[pos3S,count2], K_shelf[pos3S,step])

            d=np.maximum(D0,d)#bounded by D0

            if count2+1 in ext_index:#if suffers an extinction
                d=np.maximum(D0,d+rho_shelf[pos3S,count2+1]*d)#bounded by D0
                D_shelf[pos3S,count2+1]=np.minimum(K_shelf[pos3S,step],d)#bounded by K_shelf (The carrying capacity)
                

            else: # normal diversification period
                
                
                
                rho_shelf_eff[pos3S,count2+1] = rho_shelf[pos3S,count2+1]* np.maximum(0, (1 - (d / K_shelf[pos3S,step])))
                d=np.fmin(K_shelf[pos3S,step],d+d*rho_shelf[pos3S,count2+1]*(1-(d/K_shelf[pos3S,step])))
                
                D_shelf[pos3S,count2+1]=np.maximum(D0,d) 
                z=D_shelf[pos3S, count2+1]
                
                
                
            d=np.maximum(d,D0)

            #All active points (the 3 kinds defined by #N# above) accumulate diversity along the time gap

            d=D_shelf[posS,count2+1]

            #ipdb.set_trace()
            
            Myr=len(range(count2+2,count+1))# %time gap to still diversify
            scaling=np.ones((d.size,1))
            scaling[np.isnan(deltaAgeS)==0]=(np.minimum(deltaAgeS[np.isnan(deltaAgeS)==0]-1,Myr)/Myr)[0] #for points that appeared in the middle of the period and 
            #therefore did not accumulate diversity for the whole period (only during their age gap)
            rho=rho_shelf[:,count2+2:count+1]

            #ipdb.set_trace()

            
            
            if np.any(np.isin(np.arange(count2 + 2, count+1), ext_index)): #for a period with any Myr with a extinction we need to sum Myr step-wise diversity

                
                for gap in range(count2+2,count+1):
                    
                    d=D_shelf[posS,gap-1]

                    d=np.minimum(d,K_shelf[posS,step])

                    d=np.maximum(d,D0)

                    if gap in ext_index: #if suffers an extinction
                        
                        d=np.maximum(D0,d+d*rho_shelf[posS,gap]*scaling.flatten())#bounded by D0 
                        
                        D_shelf[posS,gap]=np.fmin(K_shelf[posS,step],d)
                        
                    else:
                        rho_shelf_eff[posS,gap] = rho_shelf[posS,gap]* np.maximum(0, (1 - (d / K_shelf[posS,step])))
                        d=np.fmin(K_shelf[posS,step],d+d*rho_shelf[posS,gap]*np.maximum(0, (1-(d/K_shelf[posS,step])))*scaling.flatten())
                        D_shelf[posS,gap]=np.maximum(D0,d)  

                #ipdb.set_trace()

                    
            

            
            
            else: #for a period without extinction we can apply the logistic growth for as an exponential approaching
                # saturation by the Myr gap to skip the sum loop
                
                

                d=np.fmin(K_shelf[posS,step],d)
                
                d=np.maximum(d,D0)
                d=K_shelf[posS,step]/ (1 + ((K_shelf[posS,step] / d) - 1) * np.exp(-rho_shelf[posS,count]*Myr*scaling.flatten()))
                
                d=np.minimum(K_shelf[posS,step],d)
                d=np.maximum(d,D0)

                
                D_shelf[posS,count]=d #included to avoid explosive values due to the explonential growth nature
                rho_shelf_eff[posS,count] = rho_shelf[posS,count] * np.maximum(0, (1 - (d / K_shelf[posS,step])))

                #if 399 in posS and count==16:
                #    print(D_shelf[399,16], "ESta")


            #Reset to D=D0 when the points is covered by ice
            #ice=ice_shelf[:,step]
            #f=np.where(ice>0)[0]
            #D_shelf[f,count]=D0

            #ipdb.set_trace()

        
        #Set the ts (time slice), count and step variables for the next iteration
        ts2=ts
        count2=count
        step = step + 1

        #Save positions of type 0 and type 1
        save_pos2S.append(pos2S)
        save_pos1S.append(pos1S)  
        D0_forced.append(D0_element)

        
    # Flip to order from point time slice 1 (0MA) to 542 (541MA) and get the Point time slices
    #for which the model is resolved (pt)

    #ipdb.set_trace()


    D_shelf=np.flip(D_shelf, axis=1)

    D_shelf=D_shelf[:,pt]

    #% Flip back once the point time slices for which the model is resolved are compiled

    D_shelf=np.flip(D_shelf, axis=1)

    rho_shelf_eff=np.flip(rho_shelf_eff, axis=1)
    rho_shelf_eff=rho_shelf_eff[:, pt]

    # Flip back once the point time slices for which the model is resolved are compiled
    rho_shelf_eff=np.flip(rho_shelf_eff, axis=1)


    #To save the data in a .npz file for tests
    #np.savez("datos_comprobacion_alphadiv.npz", z=z, D_shelf=D_shelf, scaling=scaling)

    #print(D_shelf[399,2])
    

    return rho_shelf_eff, D_shelf