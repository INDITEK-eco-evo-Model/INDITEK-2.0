import numpy as np
import pandas as pd 

def inditek_rhonet(kfood,Kmin,food_shelf,temp_shelf,ext_pattern,Kmax_mean,spec_min_mean,spec_max_mean, Q10_mean,ext_intercept_shelf_mean,ext_slope_mean,shelf_lonlatAge,Point_timeslices, model):
        
    
    data=pd.read_csv('data/rhoExt.csv')
    
    #Load the data with the mass extinction patterns, to simulate mass extinctions
    rhoExt=data.iloc[:,ext_pattern]
    time=data.iloc[:,0]
    time_ext=time[rhoExt<0] # identify time for extinction events (1-Myr resolution) (rho<0) otherwise the file values are 0.01 (no extinction)

    # Matrix to input diversificates rates (rho) fir akk the shelf palaeogeographic points: n_points z 541 time_sliices (from 541MA to 0MA, every 1Myr)
    rho_shelf = np.tile(rhoExt, (shelf_lonlatAge.shape[0], 1))  

    #Compute Carrying Capacity (K) according to the range of the greatest and lowest food available in the whole time series, 
    # (after discarding the 0.01 outliers)

    a=food_shelf[np.isnan(food_shelf)==0] # consider values of food_shelf that are not NaN
    Mfood=np.quantile(a,0.99)
    mfood=np.quantile(a,0.01)

    # ComputeK_shelf if the selected model is not "expo", (in the exponential sensitivity test, K_shelf=infty)
    if model!="expo":
        #Effective carrying capacity: max N of genera that can be supported at a palaeogeographic point according to food availability

        K_shelf=Kmax_mean-(Kmax_mean-Kmin)*((Mfood-food_shelf)/(Mfood-mfood))

        #Bounded between Kmax & Kmin (the maximum and minimum carrying capacity)

        K_shelf = K_shelf.clip(Kmin,Kmax_mean)
    else:
        K_shelf=None
    
    ############### Calculate Speciation Rate
    speciation_shelf=np.empty(food_shelf.shape) # empty matrix, same size as food_shelf to asign speciation

    for i in range(food_shelf.shape[1]):

        # Thermal bounds of time_slice[i], to apply thermal limitation accounting for aclimation, within the 0.01 and 0.99 quantile range to remove outliers

        a=temp_shelf[:,i]
        Mtemp=np.quantile(a[np.isnan(a)==0],0.99)
        mtemp=np.quantile(a[np.isnan(a)==0],0.01)

        # Initialize Qfood and Qtemp to 1
        Qfood_shelf = 1.0 # Food limitation of speciation
        Qtemp_shelf = 1.0 # Thermal limitation of speciation
        
        if model!="food": # Calculate the food limitation if the model is not "food", (in the food sensitivity test, Qfood=1)

        #Food limitation: Michaelis-Menten effect on population growth rate according to food availability
            Qfood_shelf=np.clip(food_shelf[:,i]/(kfood+food_shelf[:,i]),0,1) # Bound Qfood to the 0-1 range
        
        #Thermal limitation: Eppley curve, which defines the effect of temperature on metabolic rates, and consequently, on the population growth rate
        
        if model!="temp": # Calculate the temperature limitation if the model is not "temp", (in the temperature sensitivity test, Qtemp=1)
            EppleyCurve=Q10_mean**((temp_shelf[:,i]-mtemp)/10)
            EppleyCurve_max=Q10_mean**((Mtemp-mtemp)/10)
            Qtemp_shelf=np.clip(EppleyCurve/EppleyCurve_max,0,1) # Bound Qtemp to the 0-1 range

        Qlim_shelf=Qfood_shelf*Qtemp_shelf # Colimitation of food and temperature combined on speciation

        # Compute speciation rates according to food and temperature colimitation
        a=spec_max_mean - (spec_max_mean - spec_min_mean) * (1.0 - Qlim_shelf) # Speciation dependent on food and temp limitation (temp limitation bounded to current thermal range, i.e., considering aclimation)

        # Save speciation rate at each palaeogeographic point for time slice [i]
        speciation_shelf[:,i]=a

    # We only consider net diversification as the combination of speciation rate with mass extinctions
    rho_shelf1 = speciation_shelf

    # Incorporate Mass Extinctions and Fill Gaps
    all_timeslices = np.arange(541, -1, -1) # Begin in 0 (because the last one is -1), if not it shows a problem


    # Index of mass extinction event
    ext_index=np.nonzero(np.isin(all_timeslices, abs(time_ext)))[0] 


    # initialise position vector
    postPT=np.full([len(Point_timeslices),1], np.nan)

    ## save rho for the 82 time frames from rho_shelf1 in  their corresponding position (posPT) in the rho_shelf (-541MA:-1MA:0MA frames) 
    ## that has the big extinction (rho<0) timeframes incorporated


    for i in range(len(Point_timeslices)):

        #Select the Point_timeslices that corresponds to the timeframes in rho_shelf1 (the ones that appear in Point_timeslices) to assign the speciation 
        # values of rho_shelf1 to rho_shelf, and to identify the mass extinction events (rho<0) in those timeframes

        a=np.where(all_timeslices==Point_timeslices[i])[0][0]
        
        f=np.where(rho_shelf[:,a]<0) # Select the mass-extinction time slices

        if f[0].size == 0: #no mass extinction
            rho_shelf[:,a]=rho_shelf1[:,i]
            postPT[i]=a
        else: # mass extinction
            rho_shelf[:,a-1]=rho_shelf1[:,i]
            postPT[i]=a-1


    postPT = postPT[~np.isnan(postPT)].astype(int)

    #Fill the gaps that don't have extinction and are not in the pointslices by copying values from next point_timeslice

    # Identify the timeframes that are not in Point_timeslices and don't have extinction events (rho=0.01)
    f=np.where(rho_shelf[0,:]==0.01)[0]
    for i in range(0,len(postPT)-1): # time interval: [postPT[i], postPT[i+1]]
        
        #Select indices in f that fall strictly within the current interval, i.e., postPT[i] < f < postPT[i+1]
        v=f[np.where((f > postPT[i].item()) & (f < postPT[i + 1].item()))[0]]
        #Extract the values of rho_shelf at the *end* of the interval (postPT[i+1]) for all points, and tile them to fill the gap in rho_shelf for the indices in v
        data_to_tile=rho_shelf[:, postPT[i+1]][:, np.newaxis]
        # Replace rho_shelf values at positions at positions v with the value at postPT[i+1]
        # The column vector is repeated horizontally to match the v time-slices gap
        rho_shelf[:,v]=np.tile(data_to_tile, (1, len(v)))

    return rho_shelf,K_shelf, ext_index
