"""Module providing atmospheric helper functions."""

#--- import modules ---#
import numpy as np

#-----------------------------#
#--- Atmospheric functions ---#
#-----------------------------#

def e_sat(t):
    """Calculate saturation partial pressure of water vapour with respect to
    water and ice using Tetens' formula. The corresponding values are defined
    in IFS Documentation CY47R1 - Part IV: Physical Processes (ECMWF) pages
    117-118.
    
    Args:
        T (_float_ or _np.ndarray_): Temperature [K]
    
    Returns:
        _float_ or _np.ndarray_: Saturation partial pressure of water vapour
            [Pa]
    """

    t0 = 273.16  # [K]
    t_ice = 250.16  # [K]

    # calculate alpha
    alpha = np.where(t <= t0,
                     np.where(t <= t_ice, 0, ((t - t_ice) / (t0 - t_ice))**2),
                     1)

    # saturation pressure
    return alpha * e_sat_water(t) + (1 - alpha) * e_sat_ice(t)


def e_sat_water(t):
    """Calculate saturation partial pressure of water vapour with respect to
    water using Tetens' formula.
    
    Args:
        T (_float_ or _np.ndarray_): Temperature [K]
    
    Returns:
        _float_ or _np.ndarray_: Saturation partial pressure of water vapour
            w.r.t. water [Pa]
    """
    t0 = 273.16  # [K]
    a1w = 611.21
    a3w = 17.502
    a4w = 32.19
    e_sat_w = a1w * np.exp(a3w * (t - t0) / (t - a4w))
    return e_sat_w


def e_sat_ice(t):
    """Calculate saturation partial pressure of water vapour with respect to
    ice using Tetens' formula.

    Args:
        t (_float_ or _np.ndarray_): Temperature [K]

    Returns:
        _float_ or _np.ndarray_: Saturation partial pressure of water vapour
            w.r.t. ice [Pa]
    """
    t0 = 273.16  # [K]
    a1i = 611.21
    a3i = 22.587
    a4i = -0.7
    e_sat_i = a1i * np.exp(a3i * (t - t0) / (t - a4i))
    return e_sat_i


def e_sat_water_prime(t):
    """Calculate first derivative of the water vapour saturation pressure with
    respect to water using Tetens' formula.

    Args:
        t (_float_ or _np.ndarray_): Temperature [K]

    Returns:
        _float_ or _np.ndarray_: Derivative of water vapour saturation pressure
            w.r.t. water partial pressure [Pa/K]
    """
    t0 = 273.16  # [K]
    a1w = 611.21
    a3w = 17.502
    a4w = 32.19
    a = -a1w * a3w * (a4w - t0) / ((t - a4w) ** 2)
    b = np.exp((a3w * (t - t0)) / (t - a4w))
    return a * b