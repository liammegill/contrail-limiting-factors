"""Module providing max G helper functions."""

#--- import modules ---#
import warnings
import numpy as np
from scipy.optimize import newton, curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from limfac.calc_atmos import e_sat, e_sat_ice, e_sat_water, e_sat_water_prime


#-------------------------------#
#--- ERA5 analysis functions ---#
#-------------------------------#

def fmin_t_max(t_max, t_a, p_a):
    """Minimisation function to calculate T_max.
    
    Args:
        t_max (_float_): Maximum temperature threshold [K]
        t_a (_float_): Ambient temperature [K]
        p_a (_float_): Ambient water vapour partial pressure [Pa]
    """
    return p_a - e_sat_water(t_max) + e_sat_water_prime(t_max) * (t_max - t_a)


def calc_g_max(t, rh, rhi_cor=1.0, t_cor=0.0):
    """Calculates the maximum G slope for a given combination of temperature
    `t` and relative humidity `rh`.

    Args:
        t (array-like): Ambient temperature [K].
        rh (array-like): Ambient relative humidity [%].
        rhi_cor (float, optional): Correction to relative humidity
            (multiplier). Defaults to 1.0.
        t_cor (float, optional): Correction to temperature (addition) [K].
            Defaults to 0.0.

    Returns:
        array-like: Maximum G slope `G_max`
    """
    # with temperature correction
    tn = t + t_cor
    # calculate relative humidity and partial pressures
    ppi_sat = e_sat_ice(tn)
    ppw_sat = e_sat_water(tn)
    pp_h2o = rh / 100.0 * e_sat(tn) / rhi_cor  # with RHi correction

    # calculate T_max (using limit at 235.15 K)
    t_only_pers = tn * np.where(
        (tn <= 235.15) & (ppi_sat <= pp_h2o) & (pp_h2o <= ppw_sat),
        1, np.nan
    )
    t_max = newton(fmin_t_max, t_only_pers+10.0,
                   args=(t_only_pers, pp_h2o), tol=1e-2)
    t_max2 = np.where(t_max <= 235.15, t_max, 235.15)

    # calculate G_max and return
    g_max = (e_sat_water(t_max2) - pp_h2o) / (t_max2 - t_only_pers)
    return g_max


def calc_hist_arr(ds, bin_edges, bin_centres, rhi_cor=1.0, t_cor=0.0):
    """
    Calculate histograms of `G_max` across four different latitude bands:
    global ("tot_hist"), northern extratropics ("xtropN_hist"), tropics
    ("trop_hist") and southern extratropics ("xtropS_hist").

    Args:
        ds (xarray.Dataset): ERA5 dataset with temperature and relative
            humidity stored on reduced Gaussian grid. Must include coordinates
            `level`, `latitude` and `longitude` - if a subset of a larger
            dataset is used, then `drop=False` must be called.
        bin_edges (array-like): The edges of the bins used for histogram
            calculations.
        bin_centres (array-like): The center values of each bin, used for
            constructing the histogram array.
        rhi_cor (float, optional): Correction to relative humidity
            (multiplier). Defaults to 1.0.
        t_cor (float, optional): Correction to temperature (addition) [K].
            Defaults to 0.0.

    Returns:
        numpy.ndarray: A 3D array of histograms with shape
            (num_levels, 4, num_bins), where:
            - The first dimension corresponds to different altitude levels in
                `ds`.
            - The second dimension includes four regions (total, xtropN, trop,
                xtropS).
            - The third dimension corresponds to the binned values.
    """
    # calculate G_max
    g_max = calc_g_max(ds.t, ds.r, rhi_cor, t_cor)

    # calculate histograms
    hist_arr = np.empty((ds.level.size, 4, len(bin_centres)), dtype=np.int32)
    for i_lvl in range(len(np.atleast_1d(ds.level.data))):
        g_max_lvl = g_max.isel(level=i_lvl)

        # define latitudes belonging to the lat region
        xtropn_lats = g_max_lvl.latitude > 30.0
        trop_lats = (g_max_lvl.latitude <= 30.0) & (g_max_lvl.latitude >= -30.0)
        xtrops_lats = g_max_lvl.latitude < -30.0

        # calculate histograms
        tot_hist, _ = np.histogram(
            g_max_lvl,
            bins=bin_edges, density=False
        )
        xtropn_hist, _ = np.histogram(
            g_max_lvl.isel(values=xtropn_lats),
            bins=bin_edges, density=False
        )
        trop_hist, _ = np.histogram(
            g_max_lvl.isel(values=trop_lats),
            bins=bin_edges, density=False
        )
        xtrops_hist, _ = np.histogram(
            g_max_lvl.isel(values=xtrops_lats),
            bins=bin_edges, density=False
        )

        # save to array
        hist_arr[i_lvl, :, :] = [tot_hist, xtropn_hist, trop_hist, xtrops_hist]

    return hist_arr


def calc_cumulative_hist(ds, lat_band):
    """
    Calculate the cumulative histogram of the dataset for a specified latitude
    band.

    Args:
        ds (xarray.Dataset): The input dataset containing seasonal and
            level-based histograms.
        lat_band (str): The latitude band of interest for the cumulative
            histogram. Must be one of ["tot_hist", "xtropN_hist", "trop_hist",
            "xtropS_hist"].

    Returns:
        tuple:
            - bin_centres (numpy.ndarray): The combined array of original and
                new bin centers.
            - cum_hist (numpy.ndarray): The cumulative histogram of `ds` for
                each season and level. Shape is (num_seasons, num_levels,
                num_bin_centres).

    Notes:
        `hist_ratios` is the number of values in the ERA5 grid for each 
        latitude band and has been pre-calculated. If a different grid is used,
        these values will need to be re-calculated.
    """
    # constants
    lat_bands = ["tot_hist", "xtropN_hist", "trop_hist", "xtropS_hist"]
    hist_ratios = [1, 136999/542080, 268082/542080, 136999/542080]
    assert lat_band in lat_bands, "Input `lat_band` not in `lat_bands`."
    i_lat = lat_bands.index(lat_band)

    # calculate normalised cumulative histogram
    cum_hist = np.empty((ds.season.size, ds.level.size, ds.bin_centre.size))
    for i in range(len(ds.season)):
        ds_i = ds.isel(season=i)
        num_vals = int(ds_i.num_vals) * hist_ratios[i_lat]  # weighted num_vals
        cum_hist[i, :, :] = ds_i[lat_band].cumsum(dim="bin_centre") / num_vals

    return cum_hist


#---------------------------#
#--- ppcf vs G functions ---#
#---------------------------#

def logistic(x, l, k, x0):
    """
    Computes the logistic function, a sigmoid curve, commonly used to model
    growth or decay.

    Args:
        x (float or np.ndarray): The input values for which the logistic
            function will be computed.
        l (float): The maximum value or carrying capacity of the function.
        k (float): The steepness of the curve.
        x0 (float): The midpoint value of `x` where the function reaches half
            of `l`.

    Returns:
        float or array-like: The logistic function values for the given
            input `x`.
    """
    return l / (1 + np.exp(-k * (x - x0)))


def logistic_gen(x, l, k, x0, d):
    """
    Computes a generalized logistic function with an additional vertical shift.

    Args:
        x (float or np.ndarray): The input values for which the logistic
            function will be computed.
        l (float): The maximum value or carrying capacity of the function.
        k (float): The steepness of the curve.
        x0 (float): The midpoint value of `x` where the function reaches half
            of `l`.
        d (float): The vertical shift applied to the function.

    Returns:
        float or array-like: The values of the shifted logistic function for
            the input `x`.
    """
    return l / (1 + np.exp(-k * (x - x0))) + d


def combined_fit_accuracy(x_split, x_data, y_data) -> dict:
    """
    Fits segmented logistic functions to a data set and computes the accuracy
    metrics.

    This function divides the data set at `x_split`, fits two logistic curves
        to each segment, and evaluates the combined fit using mean squared
        error (MSE) and R-squared (R²) metrics.

    Parameters:
        x_split (float): The point in `x_data` at which the data set is
            divided into two segments.
        x_data (array-like): The independent variable data.
        y_data (array-like): The dependent variable data.

    Returns:
        dict: A dictionary containing:
            - mse (float): Mean squared error of the combined fit.
            - r2 (float): R-squared value for the combined fit.
            - params_1 (tuple): Parameters for the logistic function fit to
                the first segment.
            - params_2 (tuple): Parameters for the logistic function fit to
                the second segment.
    """
    # calculate optimal parameters using a curve fit
    split_index = np.searchsorted(x_data, x_split)
    params_1, *_ = curve_fit(
        logistic_gen, x_data[:split_index], y_data[:split_index]
    )
    params_2, *_ = curve_fit(
        logistic, x_data[split_index:],
        y_data[split_index:] - logistic_gen(x_data[split_index:], *params_1)
    )

    # calculate fitted y values
    y1 = logistic_gen(x_data, *params_1)
    y2 = logistic(x_data, *params_2)
    y_combined = y1 + y2

    # calculate mse and r2 values
    mse = mean_squared_error(y_data, y_combined)
    r2 = r2_score(y_data, y_combined)

    # return dictionary of results
    return {"mse": mse,
            "r2": r2,
            "params_1": params_1,
            "params_2": params_2,
            "lpd": params_1[0] + params_1[3] + params_2[0]}


def single_fit_accuracy(x_data, y_data) -> dict:
    """
    Fits a single logistic function to the data.

    Args:
        x_data (numpy.ndarray): The x-axis data for fitting.
        y_data (numpy.ndarray): The y-axis data for fitting.

    Returns:
        dict: A dictionary containing:
            - mse (float): Mean Squared Error of the fit.
            - r2 (float): R-squared value of the fit.
            - params (tuple): Parameters of the logistic function.
            - Lpd (float): Fit supremum (L+d).
    """
    params, *_ = curve_fit(logistic_gen, x_data, y_data)
    y = logistic_gen(x_data, *params)
    mse = mean_squared_error(y_data, y)
    r2 = r2_score(y_data, y)
    return {
        "mse": mse,
        "r2": r2,
        "params": params,
        "lpd": params[0] + params[3]
    }


def sort_cum_hist_data(bin_centres, hist_data, tiling_factor):
    """Sort and tile cumulative histogram data.

    Args:
        bin_centres (array-like): The bin centres for cumulative histogra.
        hist_data (array-like): The cumulative histogram values.
        tiling_factor (int): Number of times x data must be tiled.

    Returns:
        tuple: Two np.ndarrays, the sorted x_data and y_data.
    """
    x_unsorted = np.tile(bin_centres, tiling_factor)
    y_unsorted = hist_data.flatten()
    sorted_indices = np.argsort(x_unsorted)
    x_sorted = x_unsorted[sorted_indices]
    y_sorted = y_unsorted[sorted_indices]
    return x_sorted, y_sorted


def evaluate_segmented_fit(x_splits, x_data, y_data):
    """
    Evaluates segmented logistic fits over a range of x-splits.

    Args:
        x_splits (array-like): Array of potential x-split values.
        x_data (array-like): The x data for fitting.
        y_data (array-like): The y data for fitting.

    Returns:
        list: A list of tuples, each containing:
            - x_split (float): The x-split value.
            - mse (float): Mean Squared Error of the fit.
            - r2 (float): R-squared value of the fit.
            - params_1 (tuple): Parameters of the first logistic segment.
            - params_2 (tuple): Parameters of the second logistic segment.
    """
    results = []
    for x_split in x_splits:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = combined_fit_accuracy(x_split, x_data, y_data)
                res["x_split"] = x_split
                results.append(res)
        except Exception:
            continue
    return results


def find_optimal_split(results):
    """
    Finds the optimal x-split based on Mean Squared Error (MSE).

    Args:
        results (list): A list of results from evaluate_segmented_fit.

    Returns:
        dict: A dictionary containing:
            - x_split (float): Optimal x-split value.
            - mse (float): Optimal MSE value.
            - r2 (float): R-squared value for the optimal fit.
            - params_1 (tuple): Parameters of the first logistic segment.
            - params_2 (tuple): Parameters of the second logistic segment.
            - lpd (float): Fit supremum (L+d).
    """
    optimal = min(results, key=lambda x: x["mse"])  # Find result with minimum MSE
    return {
        "x_split": optimal["x_split"],
        "mse": optimal["mse"],
        "r2": optimal["r2"],
        "params_1": optimal["params_1"],
        "params_2": optimal["params_2"],
        "lpd": optimal["lpd"]
    }


def calc_ppcf_fits(ds, cum_hist, bin_centres):
    """
    Calculates the best-fit parameters for Potential Persistent Contrail
    Formation (p_PCF).

    Args:
        ds (xarray.Dataset): The dataset containing seasonal and altitude
            level data.
        cum_hist (numpy.ndarray): The cumulative histogram values for each
            season and altitude level.
        bin_centres (numpy.ndarray): The bin centers for the cumulative
            histogram.

    Returns:
        dict: A dictionary with altitude levels as keys. Each key maps to a
            dictionary containing:
            - "x_split" (float): The optimal x-axis split point for segmented
                fitting.
            - "mse" (float): Mean Squared Error for the fit.
            - "r2" (float): R-squared value of the fit.
            - "params_1" (tuple): Parameters of the logistic function for the
                first segment.
            - "params_2" (tuple): Parameters of the logistic function for the
                second segment (if applicable).
            - "lpd" (float): Fit supremum (l+d).
    """
    optimal_results = {}

    # Iterate over each altitude level
    for i_lvl, lvl in enumerate(ds.level.data):
        lvl_cum_hist = cum_hist[:, i_lvl, :]
        x_data, y_data = sort_cum_hist_data(
            bin_centres, lvl_cum_hist, len(ds.season)
        )

        if i_lvl < 5:  # Low altitude: segmented fitting
            x_splits = np.linspace(0.1, 3.9, 500)
            results = evaluate_segmented_fit(x_splits, x_data, y_data)
            optimal_results[lvl] = find_optimal_split(results)
        else:  # High altitude: single logistic fit
            optimal_results[lvl] = single_fit_accuracy(x_data, y_data)

    # Fit for all data
    x_data, y_data = sort_cum_hist_data(
        bin_centres, cum_hist, len(ds.season) * len(ds.level)
    )
    x_splits = np.linspace(0.1, 3.9, 500)
    results = evaluate_segmented_fit(x_splits, x_data, y_data)
    optimal_results["all"] = find_optimal_split(results)

    return optimal_results
