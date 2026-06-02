from numpy import radians, cos, sin, sqrt, arctan2

def haversine_distance(lonlat1, lonlat_array):

    #Become grades into radians
    radius=6371 # Earth's radius in kilometers
    lat1=radians(lonlat1[1])
    lon1 = radians(lonlat1[0])
    lat2 = radians(lonlat_array[:,1])
    lon2 = radians(lonlat_array[:, 0])

    #Calculate the difference for latitudes and longitudes
    deltaLat = lat2-lat1
    deltaLon = lon2-lon1

    #Calculate the Haversine distance
    a= sin(deltaLat/2)**2+cos(lat1)*cos(lat2)*sin(deltaLon/2)**2
    #Correct to avoid round mistakes
    a[a>1]=1
    #Central angle between points
    c= 2*arctan2(sqrt(a), sqrt(1-a))
    #Become the angle into kilometres
    distances = radius * c

    return distances
