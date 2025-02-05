"""Module with useful tools."""

#--- import modules ---#
import numpy as np


#----------------------------#
#--- Coordinate functions ---#
#----------------------------#

def lonlat_to_xyz(lon, lat, radians=True):
    """Converts geographic coordinates (longitude and latitude) to Cartesian coordinates (x, y, z).

    Args:
        lon (float): Longitude in degrees or radians. If degrees are provided, 
                     they will be automatically converted to radians.
        lat (float): Latitude in degrees or radians. If degrees are provided, 
                     they will be automatically converted to radians.
        radians (bool, optional): Specifies whether the input coordinates are in degrees or radians. Defaults to True (radians).

    Returns:
        numpy.ndarray: A 3-dimensional array representing the Cartesian coordinates (x, y, z) with shape (1, 3).
    """
    if not radians:
        lon = np.deg2rad(lon)
        lat = np.deg2rad(lat)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack((x, y, z), axis=-1)


def xyz_to_lonlat(pts, radians=True):
    """Converts Cartesian coordinates (x, y, z) to geographic coordinates (longitude and latitude).

    Args:
        pts (numpy.ndarray): A 3-dimensional array representing the Cartesian coordinates.
                             The shape should be (n_points, 3). Note that each row corresponds to an individual point.
        radians (bool, optional): Specifies whether the input coordinates are in degrees or radians. Defaults to True (radians).

    Returns:
        numpy.ndarray: A 2-dimensional array representing the geographic coordinates 
                       (longitude and latitude) with shape (n_points, 2).  
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    lon = np.arctan2(y, x) % (2 * np.pi)
    lat = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2))
    if not radians:
        lon = np.rad2deg(lon)
        lat = np.rad2deg(lat)
    return np.vstack((lon, lat)).T


def haversine(lon1, lat1, lon2, lat2):
    """Calculates the distance between two points on Earth using the Haversine formula.

    Args:
        lon1 (float): Longitude of the first point in degrees. 
        lat1 (float): Latitude of the first point in degrees. 
        lon2 (float): Longitude of the second point in degrees.
        lat2 (float): Latitude of the second point in degrees.

    Returns:
        float: The distance between the two points in kilometers.

    Notes:
    - The input longitudes and latitudes are assumed to be in degrees.
    - The Earth's radius is taken as 6371 kilometers.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Difference in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371. # Radius of Earth in kilometers
    distance = c * r
    
    return distance
