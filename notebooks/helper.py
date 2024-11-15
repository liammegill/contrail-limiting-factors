# Helper functions

# Dataset: Supplementary data for Megill and Grewe (2024):
# "Investigating the limiting aircraft design-dependent and environmental factors of persistent contrail formation".
# Author: Liam Megill, https://orcid.org/0000-0002-4199-6962
# Affiliation (1): Deutsches Zentrum für Luft- und Raumfahrt (DLR), Institut für Physik der Atmosphäre
# Affiliation (2): Delft University of Technology, Faculty of Aerospace Engineering
# Correspondence: liam.megill@dlr.de
# DOI: https://doi.org/10.5194/egusphere-2024-3398


#--- Load modules ---#
import numpy as np
import xarray as xr
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.optimize import newton, curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import datetime


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


#----------------------------------------------------------------#
#--- Generating and selecting random hours within 2010 decade ---#
#----------------------------------------------------------------#

# associate months with seasons
seasons = {
    'DJF': [12, 1, 2],  # December, January, February
    'MAM': [3, 4, 5],   # March, April, May
    'JJA': [6, 7, 8],   # June, July, August
    'SON': [9, 10, 11]  # September, October, November
}

def select_random_hours(num_hours_per_season, seed=None):
    """Selects a random set of hours for each season within a specified time range.

    Args:
        num_hours_per_season (int): The number of hours to randomly select for each season.
        seed (int, optional):  A seed value for numpy's random number generator. Defaults to None.

    Returns:
        numpy.ndarray: A NumPy array containing sub-arrays of selected hours for each season.
                      Each sub-array represents the randomly chosen hours for a specific season.
    """
    np.random.seed(seed)  # set numpy seed

    # Define the time range
    start_date = '2009-12-01 00:00:00'
    end_date = '2019-11-30 23:00:00'
    all_hours = pd.date_range(start=start_date, end=end_date, freq='H')

    # Create a dictionary to hold the selected hours for each season
    selected_hours = {season: [] for season in seasons}

    # For each season, select 2160 random hours
    for season, months in seasons.items():
        # Filter hours that fall into the current season
        season_hours = all_hours[all_hours.month.isin(months)]
        
        # Randomly select num_hours_per_season from the season_hours
        selected_season_hours = np.random.choice(season_hours, num_hours_per_season, replace=False)
        selected_hours[season] = selected_season_hours

    # Convert the selected hours to a numpy array with sub-arrays for each season
    combined_selected_hours = np.array([selected_hours[season] for season in seasons])

    return combined_selected_hours


def get_season_year(date):
    """
    Determine the season-year based on the given date. 

    Args:
        date (datetime.datetime object): The date to analyze.

    Returns:
        str: The season-year string representation (e.g., "2023DJF", "2023MAM").
              None if no matching season is found.

    """
    month = date.month
    year = date.year
    for season, months in seasons.items():
        if month in months:
            if season == 'DJF' and month == 12:
                return f"{year+1}{season}"
            else:
                return f"{year}{season}"
    return None


def generate_season_years(start_year, end_year):
    """Generates an ordered list of season-year strings between start_year and end_year.

    For example:
        generate_season_years(2010, 2015)
        -> ['2010DJF', '2010MAM', '2010JJA', '2010SON', '2011DJF', ...]

    Args:
        start_year (int): The starting year for the season-year strings.
        end_year (int): The ending year for the season-year strings.

    Returns:
        list: A list of season-year strings in chronological order.
    """
    season_years = []
    for year in range(start_year, end_year + 1):
        for season in ['DJF', 'MAM', 'JJA', 'SON']:
            if season == 'DJF':
                season_years.append(f"{year}{season}")
            else:
                season_years.append(f"{year}{season}")
    return season_years


def find_month_boundaries(start_date, end_date):
    """
    Calculate the start and end dates of each month within a specified date range.

    Args:
        start_date (datetime.date): The beginning date of the range.
        end_date (datetime.date): The ending date of the range.

    Returns:
        list of tuples: A list where each tuple contains the start and end date for each month
                        within the range. Each tuple is in the format (start_of_month, end_of_month),
                        with:
                        - start_of_month (datetime.date): The first day of the month.
                        - end_of_month (datetime.date): The last day of the month.
    """
    results = []  # List to store the month boundaries
    current_date = start_date
    while current_date <= end_date:
        # Start of the month
        start_of_month = current_date.replace(day=1)
        
        # End of the month is the day before the start of the next month
        next_month = current_date.replace(day=28) + datetime.timedelta(days=4)  # go to next month
        end_of_month = next_month - datetime.timedelta(days=next_month.day)
        
        results.append((start_of_month, end_of_month))  # Store the tuple
        
        # Move to the next month
        current_date = end_of_month + datetime.timedelta(days=1)
        if current_date.day > 1:
            current_date = current_date.replace(day=1)
    
    return results


#-----------------------------#
#--- Atmospheric functions ---#
#-----------------------------#

def e_sat(T):
    """Calculate saturation partial pressure of water vapour with respect to water and ice
    using Tetens' formula. The corresponding values are defined in IFS Documentation CY47R1
    - Part IV: Physical Processes (ECMWF) pages 117-118.
    
    Args:
        T (_float_ or _np.ndarray_): Temperature [K]
    
    Returns:
        _float_ or _np.ndarray_: Saturation partial pressure of water vapour [Pa]
    """
    
    T0 = 273.16  # [K]
    T_ice = 250.16  # [K] 
    
    # calculate alpha
    alpha = np.where(T <= T0,
                     np.where(T <= T_ice, 0, ((T - T_ice) / (T0 - T_ice))**2),
                     1)
    
    # saturation pressure
    e_sat = alpha * e_sat_water(T) + (1 - alpha) * e_sat_ice(T)
    return e_sat


def e_sat_water(T):
    """Calculate saturation partial pressure of water vapour with respect to water using
    Tetens' formula.
    
    Args:
        T (_float_ or _np.ndarray_): Temperature [K]
    
    Returns:
        _float_ or _np.ndarray_: Saturation partial pressure of water vapour w.r.t. water [Pa]
    """
    T0 = 273.16  # [K]
    a1w = 611.21; a3w = 17.502; a4w = 32.19
    e_sat_w = a1w * np.exp(a3w * (T - T0) / (T - a4w))
    return e_sat_w


def e_sat_ice(T):
    """Calculate saturation partial pressure of water vapour with respect to ice using
    Tetens' formula.
    
    Args:
        T (_float_ or _np.ndarray_): Temperature [K]
    
    Returns:
        _float_ or _np.ndarray_: Saturation partial pressure of water vapour w.r.t. ice [Pa]
    """
    T0 = 273.16  # [K]
    a1i = 611.21; a3i = 22.587; a4i = -0.7
    e_sat_i = a1i * np.exp(a3i * (T - T0) / (T - a4i))
    return e_sat_i


def e_sat_water_prime(T):
    """Calculate first derivative of the water vapour saturation pressure with respect to water
    using Tetens' formula.
    
    Args:
        T (_float_ or _np.ndarray_): Temperature [K]
    
    Returns:
        _float_ or _np.ndarray_: Derivative of water vapour saturation pressure w.r.t. water partial pressure [Pa/K]
    
    """
    T0 = 273.16  # [K]
    a1w = 611.21; a3w = 17.502; a4w = 32.19
    return -a1w * a3w * (a4w - T0) / ((T - a4w) ** 2) * np.exp((a3w * (T - T0)) / (T - a4w)) 


#----------------------------------#
#--- SAC and contrail formation ---#
#----------------------------------#

def calc_sac_slope(fuel_type, c_p, p, eps, EI_H2O, eta, Q, R, eta_k, dH_mol, c_p_bar):
    """Calculates the slope of the mixing line

    Args:
        fuel_type (str): Descriptor of the fuel type. Options: JA1, Hybrid, H2C, H2FC.
        c_p (_float_): Isobaric heat capacity of air [J/kg/K]
        p (_float_): Ambient pressure [Pa]
        eps (_float_):  Molar mass ratio of water vapour and dry air [-]
        EI_H2O (_float_): Emission index of water vapour [kg/kg]
        eta (_float_): Overall propulsive efficiency [-]
        Q (_float_): Lower heating value of the fuel [MJ/kg]
        R (_float_): Degree of hybridisation. R=1 is pure liquid fuel operation; R=0 pure electric operation
        eta_k (_float_): Efficiency of the liquid fuel system [-]
        dH_mol (_float_): Formation enthaply of water vapour [J/mol]
        c_p_bar (_float_): Mol-based mean heat capacity of the exhaust gases [J/mol/K]
    
    Returns:
        _float_: The slope of the mixing line, G [Pa/K]
    """
    if fuel_type == "JA1" or fuel_type == "H2C":
        G = c_p * p / eps * EI_H2O / (1 - eta) / abs(Q)
    elif fuel_type == "Hybrid":
        G = c_p * p / eps * R * EI_H2O / (R * (1 - eta_k) * Q + (1 - R) * (1 - eta) * Q * eta_k / eta)
    elif fuel_type == "H2FC":
        G = c_p_bar * p / (1 - eta) / abs(dH_mol)
    return G


def calc_single_T_max(kind, G):
    """Calculate maximum temperature threshold for contrail formation. Equations from Gierens (2021).

    Args:
        kind (_int_): 0 for G <= 2.0 or T <= 233.0; else 1
        G (_float_): Slope of the mixing line [Pa/K]
    
    Returns:
        _float_: The temperature at which the threshold mixing line touches the water vapour saturation curve, T_max [K]
    """
    if kind == 0:
        T_max = 226.69 + 9.43 * np.log(G - 0.053) + 0.72 * np.log(G - 0.053) ** 2
    else:
        T_max = 226.031 + 10.2249 * np.log(G) + 0.335372 * np.log(G) ** 2 + 0.0642105 * np.log(G) ** 3
    return T_max


def calc_single_T_min(T_max, G, e_sat_T_max_l):
    """Calculate minimum temperature threshold for contrail formation. Equation from Gierens (2021)

    Args:
        T_max (_float_): Maximum temperature threshold [K]
        G (_float_): Slope of the mixing line [Pa/K]
        e_sat_T_max_l (_float_): Saturation pressure of liquid water vapour at T_max [Pa]

    Returns:
        _float_: The temperature at which the threshold mixing line crosses RH = 0, T_min [K]
    """
    T_min = T_max - e_sat_T_max_l / G
    return T_min


def fmin_T_max(T_max, T_a, p_a):
    """Minimisation function to calculate T_max.
    
    Args:
        T_max (_float_): Maximum temperature threshold [K]
        T_a (_float_): Ambient temperature [K]
        p_a (_float_): Ambient water vapour partial pressure [Pa]
    """
    return p_a - e_sat_water(T_max) + e_sat_water_prime(T_max) * (T_max - T_a)


def calc_pp_min(T, T_max, G, pplq_sat_T_max):
    """Calculate the minimum partial pressure required for contrail formation for T_min < T < T_max

    Args:
        T (_float_): Ambient temperature [K]
        T_max (_float_): Maximum temperature threshold for contrail formation [K]
        G (_float_): Slope of the mixing line [Pa/K]
        pplq_sat_T_max (_float_): Saturation partial pressure of liquid water at T_max [Pa]

    Returns:
        _float_: Minimum partial pressure required at temperature T for a persistent contrail to form [Pa]
    """
    pp_min = pplq_sat_T_max - G * (T_max - T)
    return pp_min


#----------------------------------#
#--- Limiting factors functions ---#
#----------------------------------#

def limfac_matrix_calc_flat(mask_flat, neighbors, perimeters, cont_bool_flat):
    """Calculates the normalised edge lengths of all cells according to the flattened (time, level) mask 
    for the horizontal limiting factors study. 
    
    Args:
        mask_flat (_np.ndarray_): Flattened (time, level) limiting factor mask
        neighbors (_dict_): Dictionary of neighbours and the edge lengths between them
        perimeters (_np.ndarray_): Array of cell perimeters for normalisation
        cont_bool_flat (_np.ndarray_): Contrail formation boolean array (True = contrail forms)
    
    Returns:
        _np.ndarray_: Array of normalised edge lengths for all cells
    """
    
    n = len(mask_flat)
    n_nodes = len(neighbors)
    
    # create CSR matrix
    rows = []; cols = []; data = []
    for i in range(n):
        if cont_bool_flat[i]:  # if contrail has formed in cell i
            mod_i = i % n_nodes
            base_j = int(i / n_nodes) * n_nodes
            for j in neighbors[mod_i]:
                if mask_flat[j + base_j]:  # if cell j meets mask criteria
                    rows.append(i)
                    cols.append(j)
                    data.append(neighbors[mod_i][j]['length'])
    adj_matrix_csr = csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # calculate normalised edges
    total_edge_vals = adj_matrix_csr.sum(axis=1).A1
    norm_edge_vals = total_edge_vals / np.tile(perimeters, n // n_nodes)  # normalise with perimeters of each cell
    return norm_edge_vals


def vert_limfac_matrix_calc_flat(mask_flat, neighbors, areas, cont_bool_flat):
    """Calculates the normalised areas of all cells according to the flattened (time, level) mask for the
    vertical limiting factors study. 
    
    Args:
        mask_flat (_np.ndarray_): Flattened (time, level) vertical limiting factor mask
        neighbors (_dict_): Dictionary of vertical neighbors and the areas between them
        areas (_np.ndarray_): Array of cell areas for normalisation
        cont_bool_flat (_np.ndarray_): Contrail formation boolean array (True = contrail forms)
    
    Returns:
        _np.ndarray_: Array of normalised areas for all cells
    """
    
    n = len(mask_flat)
    n_nodes = len(neighbors)
    
    # create CSR matrix
    rows = []; cols = []; data = []
    for i in range(n):
        if cont_bool_flat[i]:  # if contrail has formed in cell i
            mod_i = i % n_nodes
            base_j = int(i / n_nodes) * n_nodes
            for j in neighbors[mod_i]:
                if mask_flat[j + base_j]:  # if cell j meets mask criteria
                    rows.append(i)
                    cols.append(j)
                    data.append(neighbors[mod_i][j]['area'])
    adj_matrix_csr = csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # calculate normalised areas
    total_area_vals = adj_matrix_csr.sum(axis=1).A1
    norm_area_vals = total_area_vals / np.tile(areas, n // len(areas))  # normalise with areas of each cell
    return norm_area_vals


def calc_limfacs_rnd(ds, ac, nbrs, prmts, rhi_cor=1.):
    """Calculate the horizontal limiting factors of random hours within the 2010 decade.

    Args:
        ds (_xarray.Dataset_): ERA5 dataset with temperature and relative humidity stored on reduced Gaussian grid
        ac (_xarray.Dataset_): Dataset of aircraft definitions
        nbrs (_dict_): Dictionary of neighbours and the edge lengths between them
        prmts (_np.ndarray_): Array of cell perimeters for normalisation
        rhi_cor (_float_, optional): Correction to relative humidity. Defaults to 1.0.

    Returns:
        _xarray.Dataset_: A 1D dataset containing the sum of all limiting factors for a single day.

    The function calculates the following limiting factors:
      - limfac_tot: Sum of all limiting factors (non-normalised)
      - limfac_frm: Sum of formation limiting factor (non-normalised)
      - limfac_frz: Sum of freezing limiting factor (non-normalised)
      - limfac_per: Sum of persistence limiting factor (non-normalised)
      - limfac_wss: Sum of water supersaturation limiting factor (non-normalised)

    Each variable in the returned dataset has an associated long_name, units and description.

    Notes:
        In this version, there is no normalisation of the limfac sums! This is because there is an irregular number of
        hours per day, so it is easier to perform the normalisation outside of this function.
    """

    # calculate relative humidity and partial pressures
    ppi_sat = e_sat_ice(ds.t)
    ppw_sat = e_sat_water(ds.t)
    pp_H2O = ds.r / 100. * e_sat(ds.t) / rhi_cor  # with RHi correction

    # calculate aircraft G and corresponding T_min, T_max and pp_min
    G_ac = np.empty((len(ds.time), len(ds.level), len(ds.latitude)))
    for i_lvl, lvl in enumerate(ds.level):
        # CAREFUL! level can be defined in hPa or in Pa. Check aircraft input file.
        G_ac[:, i_lvl, :] = calc_sac_slope(ac.fuel, ac.cp, float(lvl)*100., ac.eps, ac.EI_H2O, ac.eta, ac.Q, ac.R, 0.4, ac.dH_mol, ac.cp_mol)
    G_type_bool = np.where((G_ac <= 2.0) | (ds.t <= 233.0), True, False)  # two calculation methods for G
    T_max_ac = G_type_bool * calc_single_T_max(0, G_ac) + ~G_type_bool * calc_single_T_max(1, G_ac)
    ppw_sat_T_max = e_sat_water(T_max_ac)  # saturation liquid partial pressure at T_max
    T_max_frm = np.minimum(T_max_ac - 1 / G_ac * (ppw_sat_T_max - pp_H2O), T_max_ac)  # maximum T for formation to occur

    # calculate contrail boolean
    # limfacs:            droplet formation     droplet freezing   persistence           water supersaturation
    cont_bool = np.where((ds.t <= T_max_frm) & (ds.t <= 235.15) & (pp_H2O >= ppi_sat) & (pp_H2O <= ppw_sat), True, False)

    # create masks
    mask_limfac_tot = np.array(~cont_bool)  # full limfac mask
    mask_limfac_frm = np.array((~cont_bool) & (ds.t > T_max_frm))  # droplet formation limfac
    mask_limfac_frz = np.array((~cont_bool) & (ds.t > 235.15))  # droplet freezing limfac (TBD for H2!!)
    mask_limfac_per = np.array((~cont_bool) & (pp_H2O < ppi_sat))  # persistence limfac
    mask_limfac_wss = np.array((~cont_bool) & (pp_H2O > ppw_sat))  # water supersaturation limfac

    # initialise limfac calculations
    n_time = len(ds.time)
    n_lvl = len(ds.level)
    n_lat = len(ds.latitude)
    cont_bool_flat = cont_bool.reshape(n_time*n_lvl*n_lat)
    limfac_tot_arr_time = np.zeros((n_time, n_lvl, n_lat))
    limfac_frm_arr_time = np.zeros((n_time, n_lvl, n_lat))
    limfac_frz_arr_time = np.zeros((n_time, n_lvl, n_lat))
    limfac_per_arr_time = np.zeros((n_time, n_lvl, n_lat))
    limfac_wss_arr_time = np.zeros((n_time, n_lvl, n_lat))

    # total limfac
    mask_tot_flat = mask_limfac_tot.reshape(n_time*n_lvl*n_lat)
    if mask_tot_flat.sum() != 0:
        res = limfac_matrix_calc_flat(mask_tot_flat, nbrs, prmts, cont_bool_flat)
        limfac_tot_arr_time[:, :, :] = res.reshape(n_time, n_lvl, n_lat)

    # formation limfac
    mask_frm_flat = mask_limfac_frm.reshape(n_time*n_lvl*n_lat)
    if mask_frm_flat.sum() != 0:
        res = limfac_matrix_calc_flat(mask_frm_flat, nbrs, prmts, cont_bool_flat)
        limfac_frm_arr_time[:, :, :] = res.reshape(n_time, n_lvl, n_lat)

    # freezing limfac
    mask_frz_flat = mask_limfac_frz.reshape(n_time*n_lvl*n_lat)
    if mask_frz_flat.sum() != 0:
        res = limfac_matrix_calc_flat(mask_frz_flat, nbrs, prmts, cont_bool_flat)
        limfac_frz_arr_time[:, :, :] = res.reshape(n_time, n_lvl, n_lat)

    # persistence limfac
    mask_per_flat = mask_limfac_per.reshape(n_time*n_lvl*n_lat)
    if mask_per_flat.sum() != 0:
        res = limfac_matrix_calc_flat(mask_per_flat, nbrs, prmts, cont_bool_flat)
        limfac_per_arr_time[:, :, :] = res.reshape(n_time, n_lvl, n_lat)

    # water supersaturation limfac
    mask_wss_flat = mask_limfac_wss.reshape(n_time*n_lvl*n_lat)
    if mask_wss_flat.sum() != 0:
        res = limfac_matrix_calc_flat(mask_wss_flat, nbrs, prmts, cont_bool_flat)
        limfac_wss_arr_time[:, :, :] = res.reshape(n_time, n_lvl, n_lat)
    
    # calculate sums
    limfac_tot_arr = np.sum(limfac_tot_arr_time, axis=0)  # sum along time axis (0)
    limfac_frm_arr = np.sum(limfac_frm_arr_time, axis=0)
    limfac_frz_arr = np.sum(limfac_frz_arr_time, axis=0)
    limfac_per_arr = np.sum(limfac_per_arr_time, axis=0)
    limfac_wss_arr = np.sum(limfac_wss_arr_time, axis=0)

    # calculate potential persistent contrail formation
    ppcf_arr = np.sum(cont_bool, axis=0)
    
    # create dataset
    ds_1d = xr.Dataset({"limfac_tot": (["level", "values"], limfac_tot_arr),
                        "limfac_frm": (["level", "values"], limfac_frm_arr),
                        "limfac_frz": (["level", "values"], limfac_frz_arr),
                        "limfac_per": (["level", "values"], limfac_per_arr),
                        "limfac_wss": (["level", "values"], limfac_wss_arr),
                        "ppcf": (["level", "values"], ppcf_arr)},
                      coords = {"level": ds.level.data, 
                                "latitude": ("values", ds.latitude.data),
                                "longitude": ("values", ds.longitude.data)})

    # share attributes from ds
    ds_1d.level.attrs = ds.level.attrs
    ds_1d.latitude.attrs = ds.latitude.attrs
    ds_1d.longitude.attrs = ds.longitude.attrs

    # add new data variable attributes
    ds_1d.limfac_tot.attrs.update({"units": "-", "long_name": "limfac_tot", "description": "Sum of all limiting factors (non-normalised)"})
    ds_1d.limfac_frm.attrs.update({"units": "-", "long_name": "limfac_frm", "description": "Sum of formation limiting factor (non-normalised)"})
    ds_1d.limfac_frz.attrs.update({"units": "-", "long_name": "limfac_frz", "description": "Sum of freezing limiting factor (non-normalised)"})
    ds_1d.limfac_per.attrs.update({"units": "-", "long_name": "limfac_per", "description": "Sum of persistence limiting factor (non-normalised)"})
    ds_1d.limfac_wss.attrs.update({"units": "-", "long_name": "limfac_wss", "description": "Sum of water supersaturation limiting factor (non-normalised)"})
    ds_1d.ppcf.attrs.update({"units": "-", "long_name": "pPCF", "description": "Potential persistent contrail formation (non-normalised)"})
    ds_1d.attrs.update({"n_time": n_time})

    return ds_1d


def calc_vert_limfacs_rnd(ds, ac, nbrs, areas, rhi_cor=1.):
    """Calculate the vertical limiting factors of random hours within the 2010 decade.

    Args:
        ds (_xarray.Dataset_): ERA5 dataset with temperature and relative humidity stored on reduced Gaussian grid
        ac (_xarray.Dataset_): Dataset of aircraft definitions
        nbrs (_dict_): Dictionary of neighbours and the edge lengths between them
        areas (_np.ndarray_): Array of cell areas for normalisation
        rhi_cor (_float_, optional): Correction to relative humidity. Defaults to 1.0.

    Returns:
        _xarray.Dataset_: A 1D dataset containing the sum of all limiting factors for a single day.

    The function calculates the following limiting factors:
      - limfac_tot: Sum of all limiting factors (non-normalised)
      - limfac_frm: Sum of formation limiting factor (non-normalised)
      - limfac_frz: Sum of freezing limiting factor (non-normalised)
      - limfac_per: Sum of persistence limiting factor (non-normalised)
      - limfac_wss: Sum of water supersaturation limiting factor (non-normalised)

    Each variable in the returned dataset has an associated long_name, units and description.

    Notes:
        In this version, there is no normalisation of the limfac sums! This is because there is an irregular number of
        hours per day, so it is easier to perform the normalisation outside of this function.
    """

    # calculate relative humidity and partial pressures
    ppi_sat = e_sat_ice(ds.t)
    ppw_sat = e_sat_water(ds.t)
    pp_H2O = ds.r / 100. * e_sat(ds.t) / rhi_cor  # with RHi correction

    # calculate aircraft G and corresponding T_min, T_max and pp_min
    G_ac = np.empty((len(ds.time), len(ds.level), len(ds.latitude)))
    for i_lvl, lvl in enumerate(ds.level):
        # CAREFUL! level can be defined in hPa or in Pa
        G_ac[:, i_lvl, :] = calc_sac_slope(ac.fuel, ac.cp, float(lvl)*100., ac.eps, ac.EI_H2O, ac.eta, ac.Q, ac.R, 0.4, ac.dH_mol, ac.cp_mol)
    G_type_bool = np.where((G_ac <= 2.0) | (ds.t <= 233.0), True, False)  # two calculation methods for G
    T_max_ac = G_type_bool * calc_single_T_max(0, G_ac) + ~G_type_bool * calc_single_T_max(1, G_ac)
    ppw_sat_T_max = e_sat_water(T_max_ac)  # saturation liquid partial pressure at T_max
    T_max_frm = np.minimum(T_max_ac - 1 / G_ac * (ppw_sat_T_max - pp_H2O), T_max_ac)  # maximum T for formation to occur

    # calculate contrail boolean
    # limfacs:            droplet formation     droplet freezing   persistence           water supersaturation
    cont_bool = np.where((ds.t <= T_max_frm) & (ds.t <= 235.15) & (pp_H2O >= ppi_sat) & (pp_H2O <= ppw_sat), True, False)

    # create masks
    mask_limfac_tot = np.array(~cont_bool)  # full limfac mask
    mask_limfac_frm = np.array((~cont_bool) & (ds.t > T_max_frm))  # droplet formation limfac
    mask_limfac_frz = np.array((~cont_bool) & (ds.t > 235.15))  # droplet freezing limfac (TBD for H2!!)
    mask_limfac_per = np.array((~cont_bool) & (pp_H2O < ppi_sat))  # persistence limfac
    mask_limfac_wss = np.array((~cont_bool) & (pp_H2O > ppw_sat))  # water supersaturation limfac

    # initialise limfac calculations
    n_time = len(ds.time)
    n_lvl = len(ds.level)
    n_lat = len(ds.latitude)
    cont_bool_flat = cont_bool.reshape(n_time*n_lvl*n_lat)
    limfac_tot_arr_time = np.zeros((n_time, n_lvl, n_lat))
    limfac_frm_arr_time = np.zeros((n_time, n_lvl, n_lat))
    limfac_frz_arr_time = np.zeros((n_time, n_lvl, n_lat))
    limfac_per_arr_time = np.zeros((n_time, n_lvl, n_lat))
    limfac_wss_arr_time = np.zeros((n_time, n_lvl, n_lat))

    # total limfac
    mask_tot_flat = mask_limfac_tot.reshape(n_time*n_lvl*n_lat)
    if mask_tot_flat.sum() != 0:
        res = vert_limfac_matrix_calc_flat(mask_tot_flat, nbrs, areas, cont_bool_flat)
        limfac_tot_arr_time[:, :, :] = res.reshape(n_time, n_lvl, n_lat)

    # formation limfac
    mask_frm_flat = mask_limfac_frm.reshape(n_time*n_lvl*n_lat)
    if mask_frm_flat.sum() != 0:
        res = vert_limfac_matrix_calc_flat(mask_frm_flat, nbrs, areas, cont_bool_flat)
        limfac_frm_arr_time[:, :, :] = res.reshape(n_time, n_lvl, n_lat)

    # freezing limfac
    mask_frz_flat = mask_limfac_frz.reshape(n_time*n_lvl*n_lat)
    if mask_frz_flat.sum() != 0:
        res = vert_limfac_matrix_calc_flat(mask_frz_flat, nbrs, areas, cont_bool_flat)
        limfac_frz_arr_time[:, :, :] = res.reshape(n_time, n_lvl, n_lat)

    # persistence limfac
    mask_per_flat = mask_limfac_per.reshape(n_time*n_lvl*n_lat)
    if mask_per_flat.sum() != 0:
        res = vert_limfac_matrix_calc_flat(mask_per_flat, nbrs, areas, cont_bool_flat)
        limfac_per_arr_time[:, :, :] = res.reshape(n_time, n_lvl, n_lat)

    # water supersaturation limfac
    mask_wss_flat = mask_limfac_wss.reshape(n_time*n_lvl*n_lat)
    if mask_wss_flat.sum() != 0:
        res = vert_limfac_matrix_calc_flat(mask_wss_flat, nbrs, areas, cont_bool_flat)
        limfac_wss_arr_time[:, :, :] = res.reshape(n_time, n_lvl, n_lat)
    
    # calculate sums
    limfac_tot_arr = np.sum(limfac_tot_arr_time, axis=0)  # sum along time axis (0)
    limfac_frm_arr = np.sum(limfac_frm_arr_time, axis=0)
    limfac_frz_arr = np.sum(limfac_frz_arr_time, axis=0)
    limfac_per_arr = np.sum(limfac_per_arr_time, axis=0)
    limfac_wss_arr = np.sum(limfac_wss_arr_time, axis=0)

    # calculate potential persistent contrail formation
    ppcf_arr = np.sum(cont_bool, axis=0)
    
    # create dataset
    ds_1d = xr.Dataset({"limfac_tot": (["level", "values"], limfac_tot_arr),
                        "limfac_frm": (["level", "values"], limfac_frm_arr),
                        "limfac_frz": (["level", "values"], limfac_frz_arr),
                        "limfac_per": (["level", "values"], limfac_per_arr),
                        "limfac_wss": (["level", "values"], limfac_wss_arr),
                        "ppcf": (["level", "values"], ppcf_arr)},
                      coords = {"level": ds.level.data, 
                                "latitude": ("values", ds.latitude.data),
                                "longitude": ("values", ds.longitude.data)})

    # share attributes from ds
    ds_1d.level.attrs = ds.level.attrs
    ds_1d.latitude.attrs = ds.latitude.attrs
    ds_1d.longitude.attrs = ds.longitude.attrs

    # add new data variable attributes
    ds_1d.limfac_tot.attrs.update({"units": "-", "long_name": "limfac_tot", "description": "Sum of all limiting factors (non-normalised)"})
    ds_1d.limfac_frm.attrs.update({"units": "-", "long_name": "limfac_frm", "description": "Sum of formation limiting factor (non-normalised)"})
    ds_1d.limfac_frz.attrs.update({"units": "-", "long_name": "limfac_frz", "description": "Sum of freezing limiting factor (non-normalised)"})
    ds_1d.limfac_per.attrs.update({"units": "-", "long_name": "limfac_per", "description": "Sum of persistence limiting factor (non-normalised)"})
    ds_1d.limfac_wss.attrs.update({"units": "-", "long_name": "limfac_wss", "description": "Sum of water supersaturation limiting factor (non-normalised)"})
    ds_1d.ppcf.attrs.update({"units": "-", "long_name": "pPCF", "description": "Potential persistent contrail formation (non-normalised)"})
    ds_1d.attrs.update({"n_time": n_time})

    return ds_1d


def limfac_example(ds, lvl, ac, nbrs, prmts, output):
    """Calculate the horizontal limiting factors of a single time, used to plot example plots to help explain the function.

    Args:
        ds (_xarray.Dataset_): ERA5 dataset with temperature and relative humidity stored on reduced Gaussian grid
        lvl (_float_): Pressure level [hPa]
        ac (_xarray.Dataset_): Dataset of aircraft definitions
        nbrs (_dict_): Dictionary of neighbours and the edge lengths between them
        prmts (_np.ndarray_): Array of cell perimeters for normalisation
        output (_str_): Desired output of the function

    Returns:
        _xarray.Dataset_: A 1D dataset of the output variable.
    """

    # calculate relative humidity and partial pressures
    ppi_sat = e_sat_ice(ds.t)
    ppw_sat = e_sat_water(ds.t)
    pp_H2O = ds.r / 100. * e_sat(ds.t)

    # calculate aircraft G and corresponding T_min, T_max and pp_min
    G_ac = calc_sac_slope(ac.fuel, ac.cp, float(lvl)*100., ac.eps, ac.EI_H2O, ac.eta, ac.Q, ac.R, 0.4, ac.dH_mol, ac.cp_mol).values
    G_type_bool = np.where((G_ac <= 2.0) | (ds.t <= 233.0), True, False)  # two calculation methods for G
    T_max_ac = G_type_bool * calc_single_T_max(0, G_ac) + ~G_type_bool * calc_single_T_max(1, G_ac)
    ppw_sat_T_max = e_sat_water(T_max_ac)  # saturation liquid partial pressure at T_max
    T_max_frm = np.minimum(T_max_ac - 1 / G_ac * (ppw_sat_T_max - pp_H2O), T_max_ac)  # maximum T for formation to occur

    # calculate contrail boolean
    # limfacs:            droplet formation     droplet freezing   persistence           water supersaturation
    cont_bool = np.where((ds.t <= T_max_frm) & (ds.t <= 235.15) & (pp_H2O >= ppi_sat) & (pp_H2O <= ppw_sat), True, False)

    if output == "cont_bool":
        return cont_bool
    elif output == "issr_bool":
        return np.where((ppi_sat <= pp_H2O), True, False)
    elif output == "issrt_bool":
        return np.where((ds.t <= 235.15) & (ppi_sat <= pp_H2O), True, False)
    elif output == "temp_bool":
        return np.where((ds.t <= 235.15), True, False)
    elif output == "slope_bool":
        return np.where((ds.t <= T_max_frm), True, False)

    else:
        mask_limfac_tot = np.array(~cont_bool)  # full limfac mask
        mask_limfac_frz = np.array((~cont_bool) & (ds.t > 235.15))  # droplet freezing limfac (TBD for H2!!)
        mask_limfac_per = np.array((~cont_bool) & (pp_H2O < ppi_sat))  # persistence limfac
        
        # initialise limfac calculations
        n_lat = len(ds.latitude.values)
        limfac_tot_arr_time = np.zeros((n_lat))
        limfac_frz_arr_time = np.zeros((n_lat))
        limfac_per_arr_time = np.zeros((n_lat))
    
        # total limfac
        if mask_limfac_tot.sum() != 0:
            limfac_tot_arr = limfac_matrix_calc_flat(mask_limfac_tot, nbrs, prmts, cont_bool)
    
        # freezing limfac
        if mask_limfac_frz.sum() != 0:
            limfac_frz_arr = limfac_matrix_calc_flat(mask_limfac_frz, nbrs, prmts, cont_bool)
    
        # persistence limfac
        if mask_limfac_per.sum() != 0:
            limfac_per_arr = limfac_matrix_calc_flat(mask_limfac_per, nbrs, prmts, cont_bool)
    
        if output == "limfac_tot":
            return limfac_tot_arr
        elif output == "limfac_frz":
            return limfac_frz_arr
        elif output == "limfac_per":
            return limfac_per_arr
        else:
            print("Output unknown")


def calc_limfacs_nonborder(ds, ac_full, ac_ids, rhi_cor=1.0):
    """
    Calculate limiting factors for persistent contrail formation within each grid cell (non-border).

    Args:
        ds (xarray.Dataset): Atmospheric dataset containing temperature (`t`), relative humidity (`r`), 
                             and level, time, latitude, and longitude dimensions.
        ac_full (xarray.Dataset): Dataset containing aircraft-specific data needed to calculate `G` values.
        ac_ids (array-like): Array of aircraft IDs for which the analysis is to be calculated.
        rhi_cor (float, optional): Correction factor for relative humidity over ice (RHi). Defaults to 1.0.

    Returns:
        xarray.Dataset: A dataset of the limiting factor results, containing:
            - "ppcf" (array): Potential persistent contrail formation (non-normalized).
            - "frm" (array): Droplet formation limiting factor.
            - "frz" (array): Droplet freezing limiting factor.
            - "per" (array): Persistence limiting factor.
    """
    
    # calculate relative humidity and partial pressures
    ppi_sat = e_sat_ice(ds.t)
    ppw_sat = e_sat_water(ds.t)
    pp_H2O = ds.r / 100. * e_sat(ds.t) / rhi_cor  # with RHi correction
    
    # calculate booleans for non-aircraft related variables
    per_bool = np.where((pp_H2O >= ppi_sat), True, False)
    frz_bool = np.where((ds.t <= 235.15), True, False)
    
    # for each aircraft
    cont_bool_arr = np.empty((len(ac_ids), len(ds.time), len(ds.level), len(ds.latitude)))
    frm_bool_arr = np.empty((len(ac_ids), len(ds.time), len(ds.level), len(ds.latitude)))
    
    for i_ac, ac_id in enumerate(ac_ids):
        ac = ac_full.sel(id=ac_id)
        
        # calculate aircraft G and corresponding T_min, T_max and pp_min
        G_ac = np.empty((len(ds.time), len(ds.level), len(ds.latitude)))
        for i_lvl, lvl in enumerate(ds.level):
            # CAREFUL! level can be defined in hPa or in Pa
            G_ac[:, i_lvl, :] = calc_sac_slope(ac.fuel, ac.cp, float(lvl)*100., ac.eps,
                                               ac.EI_H2O, ac.eta, ac.Q, ac.R, 0.4, ac.dH_mol, ac.cp_mol)
        G_type_bool = np.where((G_ac <= 2.0) | (ds.t <= 233.0), True, False)
        T_max_ac = G_type_bool * calc_single_T_max(0, G_ac) + ~G_type_bool * calc_single_T_max(1, G_ac)
        ppw_sat_T_max = e_sat_water(T_max_ac)  # saturation liquid partial pressure at T_max
        pp_min_ac = np.maximum(calc_pp_min(ds.t, T_max_ac, G_ac, ppw_sat_T_max), 0.)
    
        # calculate booleans for aircraft related variables
        cont_bool_arr[i_ac, :, :, :] = np.where((ds.t <= T_max_ac) & (ds.t <= 235.15) &
                                                (ppi_sat <= pp_H2O) & (pp_min_ac <= pp_H2O) &
                                                (pp_H2O <= ppw_sat), True, False)
        frm_bool_arr[i_ac, :, :, :] = np.where((pp_min_ac <= pp_H2O) & (ds.t <= T_max_ac) &
                                               (pp_H2O <= ppw_sat), True, False)
    
    
    # sum over time
    ppcf_arr = np.sum(cont_bool_arr, axis=1)
    frm_arr = np.sum(frm_bool_arr, axis=1)
    per_arr = np.sum(per_bool, axis=0)
    frz_arr = np.sum(frz_bool, axis=0)

    # create dataset
    ds_1d = xr.Dataset(
        {
            "ppcf": (["AC", "level", "values"], ppcf_arr),
            "frm": (["AC", "level", "values"], frm_arr),
            "frz": (["level", "values"], frz_arr),
            "per": (["level", "values"], per_arr),
        },
        coords={
            "AC": ac_ids,
            "level": ds.level.data,
            "latitude": (["values"], ds.latitude.data),
            "longitude": (["values"], ds.longitude.data),
        },
    )

    # share attributes from ds
    ds_1d.AC.attrs.update({"description": "Aircraft ID"})
    ds_1d.level.attrs = ds.level.attrs
    ds_1d.latitude.attrs = ds.latitude.attrs
    ds_1d.longitude.attrs = ds.longitude.attrs

    # add new data variable attributes
    ds_1d.ppcf.attrs.update({"units": "-", "long_name": "pPCF",
                             "description": "Potential persistent contrail formation (non-normalised)"})
    ds_1d.per.attrs.update({"units": "-", "long_name": "persistence",
                            "description": "Where persistence requirement is met"})
    ds_1d.frz.attrs.update({"units": "-", "long_name": "persistence",
                            "description": "Where freezing requirement is met"})
    ds_1d.frm.attrs.update({"units": "-", "long_name": "persistence",
                            "description": "Where formation requirement is met"})
    ds_1d.attrs.update({"n_time": len(ds.time)})

    return ds_1d
    

#---------------------------#
#--- ppcf vs G functions ---#
#---------------------------#

def calc_hist_arr(ds, bin_edges, bin_centres, rhi_cor=1.):
    """
    Calculate histograms of `G_max` across four different latitude bands: global ("tot_hist"), northern extratropics
    ("xtropN_hist"), tropics ("trop_hist") and southern extratropics ("xtropS_hist").

    Args:
        ds (xarray.Dataset): The dataset containing temperature (`t`), RH (`r`), and pressure levels (`level`).
        bin_edges (array-like): The edges of the bins used for histogram calculations.
        bin_centres (array-like): The center values of each bin, used for constructing the histogram array.
        rhi_cor (float, optional): Correction factor for relative humidity over ice (RHi).
                                   Defaults to 1.0 (no correction).

    Returns:
        numpy.ndarray: A 3D array of histograms with shape (num_levels, 4, num_bins), where:
                       - The first dimension corresponds to different altitude levels in `ds`.
                       - The second dimension includes four regions (total, xtropN, trop, xtropS).
                       - The third dimension corresponds to the binned values.
    """
    
    # calculate relative humidity and partial pressures
    ppi_sat = e_sat_ice(ds.t)
    ppw_sat = e_sat_water(ds.t)
    pp_H2O = ds.r / 100. * e_sat(ds.t) / rhi_cor  # with RHi correction
    
    # calculate G_max
    t_only_pers = ds.t.values * np.where((ds.t <= 235.15) & (ppi_sat <= pp_H2O) & (pp_H2O <= ppw_sat), 1, np.nan)
    T_max = newton(fmin_T_max, t_only_pers+10., args=(t_only_pers, pp_H2O), tol=0.5)
    G_max = (e_sat_water(T_max) - pp_H2O) / (T_max - t_only_pers)
    
    # calculate histograms
    hist_arr = np.empty((len(ds.level), 4, len(bin_centres)))  # dependence: level, lat region, G bins
    for ilvl, lvl in enumerate(ds.level.values):
        G_max_lvl = G_max.isel(level=ilvl)
        
        # define latitudes belonging to the lat regions
        xtropN_lats = G_max_lvl.latitude>30.
        trop_lats = (G_max_lvl.latitude<=30.) & (G_max_lvl.latitude>=-30.)
        xtropS_lats = G_max_lvl.latitude<-30.
        
        # calculate histograms
        tot_hist, _    = np.histogram(G_max_lvl, bins=bin_edges, density=False)
        xtropN_hist, _ = np.histogram(G_max_lvl[:, xtropN_lats], bins=bin_edges, density=False)
        trop_hist, _   = np.histogram(G_max_lvl[:, trop_lats], bins=bin_edges, density=False)
        xtropS_hist, _ = np.histogram(G_max_lvl[:, xtropS_lats], bins=bin_edges, density=False)
        
        # save to array
        hist_arr[ilvl, :, :] = [tot_hist, xtropN_hist, trop_hist, xtropS_hist]
    
    return hist_arr


def logistic(x, L, k, x0):
    """
    Computes the logistic function, a sigmoid curve, commonly used to model growth or decay.

    Args:
        x (float or np.ndarray): The input values for which the logistic function will be computed.
        L (float): The maximum value or carrying capacity of the function.
        k (float): The steepness of the curve.
        x0 (float): The midpoint value of `x` where the function reaches half of `L`.

    Returns:
        float or array-like: The logistic function values for the given input `x`.
    """
    return L / (1 + np.exp(-k * (x - x0)))


def logistic_gen(x, L, k, x0, d):
    """
    Computes a generalized logistic function with an additional vertical shift.

    Args:
        x (float or np.ndarray): The input values for which the logistic function will be computed.
        L (float): The maximum value or carrying capacity of the function.
        k (float): The steepness of the curve.
        x0 (float): The midpoint value of `x` where the function reaches half of `L`.
        d (float): The vertical shift applied to the function.

    Returns:
        float or array-like: The values of the shifted logistic function for the input `x`.
    """
    return L / (1 + np.exp(-k * (x - x0))) + d


def combined_fit_accuracy(x_split, x_data, y_data):
    """
    Fits segmented logistic functions to a data set and computes the accuracy metrics.

    This function divides the data set at `x_split`, fits two logistic curves to each segment,
    and evaluates the combined fit using mean squared error (MSE) and R-squared (R²) metrics.

    Parameters:
        x_split (float): The point in `x_data` at which the data set is divided into two segments.
        x_data (np.ndarray): The independent variable data.
        y_data (np.ndarray): The dependent variable data.

    Returns:
        tuple: Contains the following elements: 
            - mse (float): Mean squared error of the combined fit.
            - r2 (float): R-squared value for the combined fit.
            - params_1 (tuple): Parameters for the logistic function fit to the first segment.
            - params_2 (tuple): Parameters for the logistic function fit to the second segment.
    """
    
    # calculate optimal parameters using a curve fit
    split_index = np.searchsorted(x_data, x_split)
    params_1, _ = curve_fit(logistic_gen, x_data[:split_index], y_data[:split_index])
    params_2, _ = curve_fit(logistic, x_data[split_index:], y_data[split_index:]
                            - logistic_gen(x_data[split_index:], *params_1))
    # calculate fitted y values
    y1 = logistic_gen(x_data, *params_1)
    y2 = logistic(x_data, *params_2)
    y_combined = y1 + y2

    # calculate mse and r2 values
    mse = mean_squared_error(y_data, y_combined)
    r2 = r2_score(y_data, y_combined)
    
    return mse, r2, params_1, params_2


def calc_cum_hist(ds, lat_band, new_bin_centres):
    """
    Calculate the cumulative histogram of the dataset for a specified latitude band.

    Args:
        ds (xarray.Dataset): The input dataset containing seasonal and level-based histograms.
        lat_band (str): The latitude band of interest for the cumulative histogram. 
                        Must be one of ["tot_hist", "xtropN_hist", "trop_hist", "xtropS_hist"].
        new_bin_centres (array-like): Additional bin centers to extend the histogram range.

    Returns:
        tuple:
            - bin_centres (numpy.ndarray): The combined array of original and new bin centers.
            - cum_hist (numpy.ndarray): The cumulative histogram of `ds` for each season and level.
                                        Shape is (num_seasons, num_levels, num_bin_centres).
    """

    lat_bands = ["tot_hist", "xtropN_hist", "trop_hist", "xtropS_hist"]  # in this order!
    hist_ratios = [1, 136999/542080, 268082/542080, 136999/542080]  # total, xtropN, trop, xtropS
    i_lat = lat_bands.index(lat_band)
    
    # extend histogram with new_bin_centres
    bin_centres = np.concatenate((ds.bin_centre.values, new_bin_centres))

    # initialise the cumulative histogram
    cum_hist = np.empty((len(ds.season), len(ds.level), len(bin_centres)))

    # calculate cumulative histogram
    for i in range(len(ds.season)):
        ds_i = ds.isel(season=i)
        num_vals = ds_i.num_vals
        cumsum = ds_i[lat_band].cumsum(dim="bin_centre") / (num_vals * hist_ratios[i_lat])
        cum_hist[i, :, :] = np.concatenate((cumsum, np.tile(cumsum[:,-1].values[:, np.newaxis], len(new_bin_centres))), axis=1)
    
    return bin_centres, cum_hist


def calc_ppcf_fits(ds, cum_hist, bin_centres):
    """
    Calculate the best fit parameters for Potential Persistent Contrail Formation (p_PCF) as a function
    of the mixing line slope G. This function fits segmented logistic models to the cumulative histogram 
    data at each altitude level and calculates the Mean Squared Error (MSE) and R-squared (R²) values
    for each fit.

    Args:
        ds (xarray.Dataset): The dataset containing seasonal and altitude level data.
        cum_hist (numpy.ndarray): The cumulative histogram values for each season and altitude level.
        bin_centres (numpy.ndarray): The bin centers for cumulative histogram.

    Returns:
        dict: A dictionary with altitude levels as keys. Each key maps to a dictionary containing:
            - "x_split" (float): The optimal x-axis split point for segmented fitting.
            - "mse" (float): Mean Squared Error for the fit.
            - "r2" (float): R-squared value of the fit.
            - "params_1" (tuple): Parameters of the logistic function for the first segment.
            - "params_2" (tuple): Parameters of the logistic function for the second segment (if applicable).
            - "Lpd" (float): Fit supremum (L+d).

    Notes:
        The fitting process involves finding the optimal x-split for segmenting the logistic fit 
        at lower altitude levels. For high altitude levels (175 and 150 hPa), a single logistic function
        is used without segmentation.
    """
    
    optimal_results = {}

    for i_lvl, lvl in enumerate(ds.level.data):

        # define data
        lvl_cum_hist = cum_hist[:, i_lvl, :]
        y_data_unsrt = lvl_cum_hist.flatten()
        x_data_unsrt = np.tile(bin_centres, len(ds.season))
        sorted_indices = np.argsort(x_data_unsrt)
        x_data = x_data_unsrt[sorted_indices]
        y_data = y_data_unsrt[sorted_indices]

        # calculate best x split
        x_splits = np.linspace(0.1, 3.9, 500)
        if i_lvl < 5:
            lvl_results = []
            for x_split in x_splits:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        mse, r2, params_1, params_2 = combined_fit_accuracy(x_split, x_data, y_data)
                        lvl_results.append((x_split, mse, r2, params_1, params_2))
                except:
                    continue

            # find optimal split based on MSE
            optimal_split_mse = min(lvl_results, key=lambda x: x[1])
            opt_Lpd = optimal_split_mse[3][0] + optimal_split_mse[3][3] + optimal_split_mse[4][0]
            optimal_results[lvl] = {"x_split": optimal_split_mse[0],
                                    "mse": optimal_split_mse[1],
                                    "r2": optimal_split_mse[2],
                                    "params_1": optimal_split_mse[3],
                                    "params_2": optimal_split_mse[4],
                                    "Lpd": opt_Lpd}
            
        else:  # at high altitude, only single logistic function is used
            params, _ = curve_fit(logistic_gen, x_data, y_data)
            y = logistic_gen(x_data, *params)
            mse = mean_squared_error(y_data, y)
            r2 = r2_score(y_data, y)
            optimal_results[lvl] = {"x_split": 0,
                                    "mse": mse,
                                    "r2": r2,
                                    "params": params,
                                    "Lpd": params[0]+params[3]}

    # calculate optimal fit for all data (level independent)
    y_data_unsrt = cum_hist.flatten()
    x_data_unsrt = np.tile(bin_centres, len(ds.season) * len(ds.level))
    sorted_indices = np.argsort(x_data_unsrt)
    x_data = x_data_unsrt[sorted_indices]
    y_data = y_data_unsrt[sorted_indices]
    x_splits = np.linspace(0.1, 3.9, 500)

    alldata_results = []
    for x_split in x_splits:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mse, r2, params_1, params_2 = combined_fit_accuracy(x_split, x_data, y_data)
                alldata_results.append((x_split, mse, r2, params_1, params_2))
        except:
            continue

    optimal_split_mse = min(alldata_results, key=lambda x: x[1])
    opt_Lpd = optimal_split_mse[3][0] + optimal_split_mse[3][3] + optimal_split_mse[4][0]
    optimal_results["all"] = {"x_split": optimal_split_mse[0],
                              "mse": optimal_split_mse[1],
                              "r2": optimal_split_mse[2],
                              "params_1": optimal_split_mse[3],
                              "params_2": optimal_split_mse[4],
                              "Lpd": opt_Lpd}
                
    return optimal_results