"""Module to load and configure input data."""

#--- import modules ---#
import warnings
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import cf2cdm

#----------------------#
#--- defining dates ---#
#----------------------#

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


#--------------------------------#
#--- loading and reading data ---#
#--------------------------------#

def load_era5_dataset(date):
    """Loads ERA5 data from DKRZ Levante for a certain input `date`.

    Args:
        date (str): Date in strftime format %Y-%m-%d, e.g. 2010-01-31

    Returns:
        xr.dataset: ERA5 dataset with coordinates `t` and `r` for levels
            [350, 300, 250, 225, 200, 175, 150] (hPa).
    """
    # change dir_path and levels if necessary
    dir_path = "/pool/data/ERA5/E5/pl/an/1H/"
    levels = [18, 19, 20, 21, 22, 23, 24]

    # load temperature dataset (130)
    t_file_path = "130/E5pl00_1H_{}_130.grb".format(date)
    dsg_t = xr.open_dataset(
        dir_path+t_file_path,
        engine="cfgrib",
        backend_kwargs={"indexpath":None},
        # chunks={'time': 1}  # chunking for parallel processing
    )
    # load relative humidity dataset (157)
    r_file_path = "157/E5pl00_1H_{}_157.grb".format(date)
    dsg_r = xr.open_dataset(
        dir_path+r_file_path,
        engine="cfgrib",
        backend_kwargs={"indexpath":None},
        # chunks={'time': 1}  # chunking for parallel processing
    )
    # merge datasets and translate coordinates to ECMWF
    # warnings package used to ignore UserWarning when converting time -> time
    dsg = xr.merge([dsg_t, dsg_r])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dsg = cf2cdm.translate_coords(dsg, cf2cdm.ECMWF)
    dsg = dsg.isel(level=levels)

    return dsg
