import numpy as np
import pandas as pd 

def rhonet_evo(kfood,Kmin,food_shelf,temp_shelf,ext_pattern,Kmax_mean,spec_min_mean,spec_max_mean, Q10_mean,ext_intercept_shelf_mean,ext_slope_mean,shelf_lonlatAge,Point_timeslices, model):
        
    
    data=pd.read_csv('data/rhoExt.csv')
    
    #Load the data with the extinction patterns, to be used later to incorporate the mass extinctions
    rhoExt=data.iloc[:,ext_pattern]
    time=data.iloc[:,0]
    time_ext=time[rhoExt<0]#Select the times where suffers mass extinctions (rho<0)

    #Create a matrix with the initial speciation rates (rho) for the shelf, with shape n_pointsxtime_slices (541, from 541MA to 0MA, every 1 million years)
    rho_shelf = np.tile(rhoExt, (shelf_lonlatAge.shape[0], 1))  

    #Calculate Carrying Capacity (K) according to the range of greatest and lowest food available in the whole time series, 
    # after discarding the 0.01 outliers

    a=food_shelf[np.isnan(food_shelf)==0] #Takes the values of food_shelf that are not NaN
    Mfood=np.quantile(a,0.99)
    mfood=np.quantile(a,0.01)

    #Just calculate K_shelf if the model is not "expo", (in expo, K_shelf=infty)
    if model!="expo":
        #Effective carrying capacity: max N of genera that can be supported at a point according to food at that point and time 

        K_shelf=Kmax_mean-(Kmax_mean-Kmin)*((Mfood-food_shelf)/(Mfood-mfood))

        #Bounded between Kmax & Kmin (the maximum and minimum carrying capacity)

        K_shelf = K_shelf.clip(Kmin,Kmax_mean)
    else:
        K_shelf=None
    # Calculate Speciation Rate
    speciation_shelf=np.empty(food_shelf.shape)#empty matrix, same size as food_shelf to asign speciation

    for i in range(food_shelf.shape[1]):
        #Select the temperature values within the 0.01 and 0.99 quantile range to remove outliers
        a=temp_shelf[:,i]
        Mtemp=np.quantile(a[np.isnan(a)==0],0.99)
        mtemp=np.quantile(a[np.isnan(a)==0],0.01)

        #We initialize Qfood and Qtemp to 1
        Qfood_shelf = 1.0#Modifier that constrains speciation based on food supply
        Qtemp_shelf = 1.0#Modifier that constrains speciation based on temperature
        
        if model!="food":#Calculate the food limitation only if the model is not "food", (if it is "food", Qfood=1)

        #Food limitation according to Michaelis-Menten effect on population growth rate according to food availability
            Qfood_shelf=np.clip(food_shelf[:,i]/(kfood+food_shelf[:,i]),0,1)#Bounds Qfood to the 0-1 range
        
        #Thermal limitation based on the Eppley curve, which defines the effect of temperature on metabolic rates, and consequently, 
        # on the population growth rate
        
        if model!="temp":#Calculate the temperature limitation only if the model is not "temp", (if it is "temp", Qtemp=1)
            EppleyCurve=Q10_mean**((temp_shelf[:,i]-mtemp)/10)
            EppleyCurve_max=Q10_mean**((Mtemp-mtemp)/10)
            Qtemp_shelf=np.clip(EppleyCurve/EppleyCurve_max,0,1)#Bounds Qtemp to the 0-1 range

        Qlim_shelf=Qfood_shelf*Qtemp_shelf#colimitation of food and temperature combined (product of both)

        #Calculates speciation rates according to food and temperature colimitation
        a=spec_max_mean - (spec_max_mean - spec_min_mean) * (1.0 - Qlim_shelf) #speciation dependent on food and temp limitation (temp limitation bounded to current thermal range, i.e., considering aclimation)

        #The speciation rate is calculated for each time slice and then stored in the general vector
        speciation_shelf[:,i]=a

    #We only consider net diversification as the combination of speciation rate with mass extinctions
    rho_shelf1 = speciation_shelf

    #Incorporate Mass Extinctions and Fill Gaps
    all_timeslices = np.arange(541, -1, -1)#Begin in 0 (because the last one is -1), if not it shows a problem


    #Create an index of the timeslices that have mass extinctions, to be used later
    ext_index=np.nonzero(np.isin(all_timeslices, abs(time_ext)))[0] 



    postPT=np.full([len(Point_timeslices),1], np.nan)

    ## save rho for the 82 time frames at their corresponding position (posPT) in the -541MA:-1MA:0MA frames 
    ## that alraeady have the big extinction (rho<0) timeframes incorporated


    for i in range(len(Point_timeslices)):

        a=np.where(all_timeslices==Point_timeslices[i])[0][0]#Select in all the sequence of years, just the ones that are in the Point_timeslices
        
        f=np.where(rho_shelf[:,a]<0)#Select the years in point_timeslices (the ones that appear in slices) that suffers a mass extinction

        if f[0].size == 0:#If it doesn't suffer a mass extinctions
            rho_shelf[:,a]=rho_shelf1[:,i]
            postPT[i]=a
        else:
            rho_shelf[:,a-1]=rho_shelf1[:,i]
            postPT[i]=a-1


    postPT = postPT[~np.isnan(postPT)].astype(int)

    #Fill the gaps that don't have extinction and are not in the pointslices by copying values from next point_timeslice
    f=np.where(rho_shelf[0,:]==0.01)[0]
    for i in range(0,len(postPT)-1):

        v=f[np.where((f > postPT[i].item()) & (f < postPT[i + 1].item()))[0]]
        data_to_tile=rho_shelf[:, postPT[i+1]][:, np.newaxis]
        rho_shelf[:,v]=np.tile(data_to_tile, (1, len(v)))

    return rho_shelf,K_shelf, ext_index
