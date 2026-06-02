import numpy as np

def inditek_model_proof (D,proof):
    model=D

    #It calculate the residuals following the equation residuals=((data_observed-data_obtained)/error)^2
    residuals = (model - proof)/2

    residuals=residuals**2
    #It just selects the residuals that are not Nan, the ones that correspond to the active points
    residuals=residuals[np.isnan(residuals)==False]

    #The final RSS is the sum of the residuals for every active point
    rss=np.sum(residuals)

    print("Residual Sum of Squares (RSS):", rss)

    return rss, residuals