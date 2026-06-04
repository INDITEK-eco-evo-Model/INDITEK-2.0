import numpy as np

def inditek_model_proof (D,proof):
    model=D

    #Calculate the residuals following the equation: residuals=((data_observed-data_obtained)/error)^2
    residuals = (model - proof)/2

    residuals=residuals**2
    #Select only the residuals that are not NaN, corresponding to the active points
    residuals=residuals[np.isnan(residuals)==False]

    #The final RSS is the sum of the residuals for every active point
    rss=np.sum(residuals)

    print("Residual Sum of Squares (RSS):", rss)

    return rss, residuals