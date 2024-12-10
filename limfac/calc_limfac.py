"""Module providing limiting factors helper functions."""

#--- import modules ---#
import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix
from limfac.calc_atmos import e_sat, e_sat_ice, e_sat_water, e_sat_water_prime


#----------------------------------#
#--- SAC and contrail formation ---#
#----------------------------------#

def calc_sac_slope(fuel_type, c_p, p, eps, ei_h2o, eta, q, r, eta_k, dh_mol,
                   c_p_bar):
    """Calculates the slope of the mixing line

    Args:
        fuel_type (_str_): Descriptor of the fuel type. 
            Options: JA1, Hybrid, H2C, H2FC.
        c_p (_float_): Isobaric heat capacity of air [J/kg/K]
        p (_float_): Ambient pressure [Pa]
        eps (_float_):  Molar mass ratio of water vapour and dry air [-]
        ei_h2o (_float_): Emission index of water vapour [kg/kg]
        eta (_float_): Overall propulsive efficiency [-]
        q (_float_): Lower heating value of the fuel [MJ/kg]
        r (_float_): Degree of hybridisation. r=1 is pure liquid fuel
            operation; r=0 pure electric operation
        eta_k (_float_): Efficiency of the liquid fuel system [-]
        dh_mol (_float_): Formation enthaply of water vapour [J/mol]
        c_p_bar (_float_): Mol-based mean heat capacity of the exhaust gases
            [J/mol/K]
    
    Returns:
        _float_: The slope of the mixing line, G [Pa/K]
    """
    if fuel_type in ("JA1", "H2C"):
        return c_p * p / eps * ei_h2o / (1 - eta) / abs(q)
    if fuel_type == "Hybrid":
        numerator = c_p * p / eps * r * ei_h2o
        denominator = r * (1 - eta_k) * q + (1-r) * (1 - eta) * q * eta_k / eta
        return numerator / denominator
    if fuel_type == "H2FC":
        return c_p_bar * p / (1 - eta) / abs(dh_mol)

    raise ValueError("Invalid fuel type")


def calc_single_t_max(kind, g):
    """Calculate maximum temperature threshold for contrail formation.
    Equations from Gierens (2021).

    Args:
        kind (_int_): 0 for G <= 2.0 or T <= 233.0; else 1
        g (_float_): Slope of the mixing line [Pa/K]

    Returns:
        _float_: The temperature at which the threshold mixing line touches the
            water vapour saturation curve, T_max [K]
    """
    if kind == 0:
        t_max = 226.69 + 9.43 * np.log(g - 0.053) + \
            0.72 * np.log(g - 0.053) ** 2
    else:
        t_max = 226.031 + 10.2249 * np.log(g) + 0.335372 * np.log(g) ** 2 + \
            0.0642105 * np.log(g) ** 3
    return t_max


def calc_single_t_min(t_max, g, e_sat_t_max_l):
    """Calculate minimum temperature threshold for contrail formation.
    Equation from Gierens (2021)

    Args:
        t_max (_float_): Maximum temperature threshold [K]
        g (_float_): Slope of the mixing line [Pa/K]
        e_sat_t_max_l (_float_): Saturation pressure of liquid water vapour
            at T_max [Pa]

    Returns:
        _float_: The temperature at which the threshold mixing line crosses
            RH = 0, T_min [K]
    """
    t_min = t_max - e_sat_t_max_l / g
    return t_min


def fmin_t_max(t_max, t_a, p_a):
    """Minimisation function to calculate T_max.
    
    Args:
        t_max (_float_): Maximum temperature threshold [K]
        t_a (_float_): Ambient temperature [K]
        p_a (_float_): Ambient water vapour partial pressure [Pa]
    """
    return p_a - e_sat_water(t_max) + e_sat_water_prime(t_max) * (t_max - t_a)


def calc_pp_min(t, t_max, g, pplq_sat_t_max):
    """Calculate the minimum partial pressure required for contrail formation
    for T_min < T < T_max

    Args:
        t (_float_): Ambient temperature [K]
        t_max (_float_): Maximum temperature threshold for contrail formation
            [K]
        g (_float_): Slope of the mixing line [Pa/K]
        pplq_sat_t_max (_float_): Saturation partial pressure of liquid water
            at T_max [Pa]

    Returns:
        _float_: Minimum partial pressure required at temperature T for a
            persistent contrail to form [Pa]
    """
    pp_min = pplq_sat_t_max - g * (t_max - t)
    return pp_min


#----------------------------------#
#--- Limiting factors functions ---#
#----------------------------------#

def calc_limfac_matrix(mask, cont_bool, neighbors, weight_val, weights):
    """Calculates the normalised weighting of all cells according to the
    flattened (time, level) mask for the limiting factors study. In horizontal
    direction, `weights` corresponds to the cell perimeters; in vertical
    direction to the cell areas.
    
    Args:
        mask (_np.ndarray_): Flattened (time, level) horizontal/vertical
            limiting factor mask
        cont_bool (_np.ndarray_): Contrail formation boolean array (True = 
            contrail forms)
        neighbors (_dict_): Dictionary of horizontal/vertical neighbors. Must
            include the corresponding `weight_val` (length or area) between
            the cells.
        weight_val (_str_): The weighting value. Choice of `length` for
            horizontal direction, and `area` for vertical direction.
        weights (_np.ndarray_): Array of weights for normalisation. Corresponds
            to cell perimeters in horizontal direction and cell areas in 
            vertical direction.
    
    Returns:
        _np.ndarray_: Array of normalised areas for all cells
    """
    # pre-conditions
    assert len(mask) > 0, "Mask cannot be empty."
    assert len(mask) == len(cont_bool), "Contrail boolean array " \
        "and mask must be the same size."
    assert len(neighbors) > 0, "Neighbors dictionary cannot be empty."
    assert len(neighbors) % len(weights) == 0, "Neighbors dictionary size " \
        "must be an integer multiple of weights."
    assert len(mask) % len(neighbors) == 0, "Contrail mask length must " \
        "be an integer multiple of neighbors length."
    assert weight_val in ("length", "area"), "weight_val must be either " \
        "`length` (horizontal) or `area` (vertical)."

    n = len(mask)
    n_nodes = len(neighbors)
    n_weights = len(weights)

    # create CSR matrix
    rows = []
    cols = []
    data = []
    for i in range(n):
        if cont_bool[i]:  # if contrail has formed in cell i
            mod_i = i % n_nodes
            base_j = int(i / n_nodes) * n_nodes
            for j in neighbors[mod_i]:
                if mask[j + base_j]:  # if cell j meets mask criteria
                    rows.append(i)
                    cols.append(j)
                    data.append(neighbors[mod_i][j][weight_val])
    adj_matrix_csr = csr_matrix((data, (rows, cols)), shape=(n, n))

    # calculate normalised values
    total_area_vals = adj_matrix_csr.sum(axis=1).A1
    norm_area_vals = total_area_vals / np.tile(weights, n // n_weights)
    return norm_area_vals


def calc_t_frm_lim(g, t, pp_h2o):
    """Calculates limiting formation temperature, which is equal to T_max if
    T_max < 235.15 K, else 235.15 K. All args must have the same length, 
    or all args except one must have length one.

    Args:
        g (array-like): Slope of the aircraft SAC mixing line [Pa/K]
        t (array-like): Ambient temperature [K]
        pp_h2o (array-like): Ambient partial pressure of water vapour [Pa]

    Returns:
        array-like: Limiting formation temperature [K].
    """
    # calculate T_max
    g_type = np.where((g < 2.0) | (t <= 233.0), True, False)
    t_max = g_type * calc_single_t_max(0, g) + ~g_type * calc_single_t_max(1, g)
    t_max_lt_235 = np.where(t_max <= 235.15, True, False)

    # if T_max > 235.15, then T_lim = 235.15. Else T_lim = T_max
    t_lim = t_max_lt_235 * t_max + ~t_max_lt_235 * 235.15
    ppw_sat_t_lim = e_sat_water(t_lim)
    t_frm_lim = np.minimum(t_lim - 1 / g * (ppw_sat_t_lim - pp_h2o), t_lim)

    return t_frm_lim


def calc_limfac_bools(t, rh, g, rhi_cor=1.0):
    """Calculates the limiting factor and persistent contrail formation
    booleans based on input temperature and relative humidity.
    
    Args:
        t (array-like): Ambient temperature [K].
        rh (array-like): Ambient relative humidity [%].
        g (array-like): SAC mixing line slope [Pa/K].
        ac (xarray.Dataset): Dataset of aircraft design definitions.
        rhi_cor (float, optional): Correction to relative humidity.
            Defaults to 1.0.
        
    Returns:
        dict of bool: Dictionary of limiting factor booleans. Keys are:
            'frm', 'frz', 'per', 'wss' and 'cont'.
    """

    # calculate relative humidity and partial pressures
    ppi_sat = e_sat_ice(t)
    ppw_sat = e_sat_water(t)
    pp_h2o = rh / 100. * e_sat(t) / rhi_cor  # with RHi correction

    # calculate limiting formation temperature (T_max or 235.15 K, whichever
    # is lower)
    t_frm_lim = calc_t_frm_lim(g, t, pp_h2o)

    # determine limiting factors
    frm_limfac = t <= t_frm_lim     # droplet formation
    frz_limfac = t <= 235.15        # droplet freezing
    per_limfac = ppi_sat <= pp_h2o  # persistence
    wss_limfac = pp_h2o <= ppw_sat  # water supersaturation
    cont_bool = np.where(frm_limfac & frz_limfac & per_limfac & wss_limfac,
                         True, False)

    # create output dictionary
    limfac_bools = {
        "frm": frm_limfac,
        "frz": frz_limfac,
        "per": per_limfac,
        "wss": wss_limfac,
        "cont": cont_bool
    }

    return limfac_bools


def calc_limfacs(ds, ac, direction, nbrs, weights, rhi_cor=1.):
    """Calculate the limiting factors of random hours within the
    2010 decade.

    Args:
        ds (_xarray.Dataset_): ERA5 dataset with temperature and relative
            humidity stored on reduced Gaussian grid. Must include coordinates
            `level`, `latitude` and `longitude` - if a subset of a larger
            dataset is used, then `drop=False` must be called.
        ac (_xarray.Dataset_): Dataset of aircraft definitions
        direction (_str)_: One of 'h' (horizontal) or 'v' (vertical).
        nbrs (_dict_): Dictionary of neighbours and edge lengths between them
            (horizontal) or cell areas (vertical).
        weights (_np.ndarray_): Array of weights for normalisation
        rhi_cor (_float_, optional): Correction to relative humidity.
            Defaults to 1.0.

    Returns:
        _xarray.Dataset_: A 1D dataset containing the sum of all limiting
            factors for a single day.

    The function calculates the following limiting factors:
      - limfac_tot: Sum of all limiting factors (non-normalised)
      - limfac_frm: Sum of formation limiting factor (non-normalised)
      - limfac_frz: Sum of freezing limiting factor (non-normalised)
      - limfac_per: Sum of persistence limiting factor (non-normalised)
      - limfac_wss: Sum of water supersaturation limiting factor
        (non-normalised)

    Each variable in the returned dataset has an associated long_name, units
    and description.

    Notes:
        In this version, there is no normalisation of the limfac sums! This is
        because there is an irregular number of hours per day, so it is easier
        to perform the normalisation outside of this function.
    """

    # pre-conditions
    assert 'level' in ds.coords, "The 'level' coordinate was not included or "\
        "has been dropped. Ensure that drop=False is used when selecting data."
    assert 'time' in ds.coords, "The 'time' coordiante was not included or "\
        "has been dropped. Ensure that drop=False is used when selecting data."
    assert 'latitude' in ds, "The 'latitude' variable is missing."
    assert direction in ("h", "v"), "`direction` must be one of 'h'" \
        "(horizontal) or 'v' (vertical)."
    assert ds.latitude.size == len(weights), "The lat/lon data must match "\
        "the shape of the pre-calculated weights."
    assert len(nbrs) % ds.latitude.size == 0, "The lat/lon data must match "\
        "the shape of the pre-calculated neighbors."

    # calculate SAC slopes
    g_lvl = np.empty(ds.level.size)
    for i_lvl, lvl in enumerate(np.atleast_1d(ds.level.data)):
        g_lvl[i_lvl] = calc_sac_slope(
            ac.fuel, ac.cp, lvl*100., ac.eps, ac.EI_H2O, ac.eta, ac.Q, ac.R,
            0.4, ac.dH_mol, ac.cp_mol
        )
    g = np.tile(
        g_lvl[:, np.newaxis, np.newaxis],
        (1, ds.time.size, ds.latitude.size)
    ).transpose(1, 0, 2).squeeze()

    # calculate limiting factor boolean dictionary
    lf_dict = calc_limfac_bools(ds.t, ds.r, g, rhi_cor)
    cont_bool = lf_dict["cont"]

    # create masks
    mask_tot = np.array(~cont_bool)  # full limfac mask
    mask_frm = np.array(~cont_bool & ~lf_dict["frm"])
    mask_frz = np.array(~cont_bool & ~lf_dict["frz"])
    mask_per = np.array(~cont_bool & ~lf_dict["per"])
    mask_wss = np.array(~cont_bool & ~lf_dict["wss"])

    # initialise limfac calculations
    lf_arr = np.zeros((5,) + ds.t.shape)
    cont_bool_flat = cont_bool.flatten()

    # calculate limfacs
    for i_lf, lf_mask in enumerate([mask_tot, mask_frm, mask_frz, mask_per,
                                    mask_wss]):
        mask_flat = lf_mask.flatten()
        if mask_flat.sum() != 0:
            weight_val = "length" if direction == "h" else "area"
            res = calc_limfac_matrix(
                mask_flat, cont_bool_flat, nbrs, weight_val, weights
            )
            lf_arr[i_lf, :] = res.reshape(ds.t.shape)

    # initialise output xarray dataset
    ds_out = xr.Dataset(coords = {"level": ds.level.data,
                                  "latitude": ("values", ds.latitude.data),
                                  "longitude": ("values", ds.longitude.data)})
    labels = ["limfac_tot", "limfac_frm", "limfac_frz", "limfac_per",
              "limfac_wss"]

    # calculate results and ensure single time and level values can be used
    # TODO if the input is a function of time, maybe this isn't necessary
    if ds.time.size > 1:
        lf_res = np.sum(lf_arr, axis=1)
        ppcf_arr = np.sum(cont_bool, axis=0)
    else:
        lf_res = lf_arr
        ppcf_arr = cont_bool.astype(float)

    if ds.level.size > 1:
        for idx, lbl in enumerate(labels):
            ds_out[lbl] = (["level", "values"], lf_res[idx])
            ds_out["ppcf"] = (["level", "values"], ppcf_arr)
    else:
        for idx, lbl in enumerate(labels):
            ds_out[lbl] = (["values"], lf_res[idx])
            ds_out["ppcf"] = (["values"], ppcf_arr)

    # share attributes from ds
    ds_out.level.attrs = ds.level.attrs
    ds_out.latitude.attrs = ds.latitude.attrs
    ds_out.longitude.attrs = ds.longitude.attrs

    # add new data variable attributes
    dirln = "horizontal" if direction == "h" else "vertical"
    ds_out.limfac_tot.attrs.update(
        {"units": "-", "long_name": "limfac_tot",
         "description": f"Sum of all {dirln} limiting factors (non-normalised)"}
        )
    ds_out.limfac_frm.attrs.update(
        {"units": "-", "long_name": "limfac_frm",
         "description": f"Sum of {dirln} formation limiting factor (non-normalised)"}
        )
    ds_out.limfac_frz.attrs.update(
        {"units": "-", "long_name": "limfac_frz",
         "description": f"Sum of {dirln} freezing limiting factor (non-normalised)"}
        )
    ds_out.limfac_per.attrs.update(
        {"units": "-", "long_name": "limfac_per",
         "description": f"Sum of {dirln} persistence limiting factor (non-normalised)"}
        )
    ds_out.limfac_wss.attrs.update(
        {"units": "-", "long_name": "limfac_wss",
         "description": f"Sum of {dirln} water supersaturation limiting factor (non-normalised)"}
        )
    ds_out.ppcf.attrs.update(
        {"units": "-", "long_name": "pPCF",
         "description": "Potential persistent contrail formation (non-normalised)"}
        )
    ds_out.attrs.update({"n_time": ds.time.size})

    return ds_out


def calc_limfacs_nonborder(ds, ac_full, ac_ids, rhi_cor=1.0):
    """Calculate the non-border (cell) limiting factors using ERA5 data
    for the 2010 decade.

    Args:
        ds (_xarray.Dataset_): ERA5 dataset with temperature and relative
            humidity stored on reduced Gaussian grid. Must include coordinates
            `level`, `latitude` and `longitude` - if a subset of a larger
            dataset is used, then `drop=False` must be called.
        ac_full (_xarray.Dataset_): Dataset with aircraft design parameters.
            Data variable "id" must correspond with `ac_ids`.
        ac_ids (_list_): List of strings corresponding to aircraft IDs.
        rhi_cor (_float_, optional): Correction to relative humidity.
            Defaults to 1.0.
    
    Returns:
        _xarray.Dataset_: A 1D dataset containing the sum of all limiting
            factors (for the cell, not the borders) for a single day.

    The function calculates the following limiting factors:
      - limfac_tot: Sum of all limiting factors (non-normalised)
      - limfac_frm: Sum of formation limiting factor (non-normalised)
      - limfac_frz: Sum of freezing limiting factor (non-normalised)
      - limfac_per: Sum of persistence limiting factor (non-normalised)
      - limfac_wss: Sum of water supersaturation limiting factor
        (non-normalised)

    Each variable in the returned dataset has an associated long_name, units
    and description.

    Notes:
        In this version, there is no normalisation of the limfac sums! This is
        because there is an irregular number of hours per day, so it is easier
        to perform the normalisation outside of this function.
    """

    # pre-conditions
    assert 'level' in ds.coords, "The 'level' coordinate was not included or "\
        "has been dropped. Ensure that drop=False is used when selecting data."
    assert 'time' in ds.coords, "The 'time' coordiante was not included or "\
        "has been dropped. Ensure that drop=False is used when selecting data."
    assert 'latitude' in ds, "The 'latitude' variable is missing."
    assert 't' in ds, "The 't' (temperature) variable is missing."
    assert 'r' in ds, "The 'r' (relative humidity) variable is missing."
    assert ds.t.shape == (ds.time.size, ds.level.size, ds.latitude.size), \
        f"Data variable `t` is size {ds.t.shape} but should be size " \
        f"{(ds.time.size, ds.level.size, ds.latitude.size)}."
    assert ds.r.shape == (ds.time.size, ds.level.size, ds.latitude.size), \
        f"Data variable `r` is size {ds.r.shape} but should be size " \
        f"{(ds.time.size, ds.level.size, ds.latitude.size)}."

    # calculate relative humidity and partial pressures
    ppi_sat = e_sat_ice(ds.t)
    ppw_sat = e_sat_water(ds.t)
    pp_h2o = ds.r / 100. * e_sat(ds.t) / rhi_cor  # with RHi correction

    # aircraft-independent limiting factors
    per_bool = np.where(ppi_sat <= pp_h2o, 1.0, 0.0)
    wss_bool = np.where(pp_h2o <= ppw_sat, 1.0, 0.0)
    frz_bool = np.where(ds.t <= 235.15, 1.0, 0.0)

    # initialise arrays for storing data
    cont_bool_arr = np.empty((len(ac_ids), ds.time.size,
                              ds.level.size, ds.latitude.size))
    frm_bool_arr = np.empty((len(ac_ids), ds.time.size,
                             ds.level.size, ds.latitude.size))

    # loop per aircraft design
    for i_ac, ac_id in enumerate(ac_ids):
        ac = ac_full.sel(id=ac_id)

        # calculate SAC slopes
        g_lvl = np.empty(ds.level.size)
        for i_lvl, lvl in enumerate(np.atleast_1d(ds.level.data)):
            g_lvl[i_lvl] = calc_sac_slope(
                ac.fuel, ac.cp, lvl*100., ac.eps, ac.EI_H2O, ac.eta, ac.Q, ac.R,
                0.4, ac.dH_mol, ac.cp_mol
            )
        g = np.tile(
            g_lvl[:, np.newaxis, np.newaxis],
            (1, ds.time.size, ds.latitude.size)
        ).transpose(1, 0, 2).squeeze()

        # formation limiting factor and persistent contrail boolean
        t_frm_lim = calc_t_frm_lim(g, ds.t, pp_h2o)
        frm_bool = np.where(ds.t <= t_frm_lim, 1.0, 0.0)
        cont_bool = np.where(frm_bool & frz_bool & per_bool & wss_bool,
                             1.0, 0.0)
        # store in array
        frm_bool_arr[i_ac, :] = frm_bool
        cont_bool_arr[i_ac, :] = cont_bool

    # sum over time
    ppcf_arr = cont_bool_arr.sum(axis=1)
    frm_arr = frm_bool_arr.sum(axis=1)
    per_arr = per_bool.sum(axis=0)
    frz_arr = frz_bool.sum(axis=0)

    # store as dataset
    ds_out = xr.Dataset(
        {
            "ppcf": (["AC", "level", "values"], ppcf_arr),
            "frm": (["AC", "level", "values"], frm_arr),
            "frz": (["level", "values"], frz_arr),
            "per": (["level", "values"], per_arr)
        },
        coords={
            "AC": ac_ids,
            "level": ds.level,
            "latitude": ds.latitude,
            "longitude": ds.longitude,
        }
    )

    # update attributes
    ds_out.AC.attrs.update({"description": "Aircraft ID"})
    ds_out.ppcf.attrs.update(
        {"units": "-", "long_name": "pPCF",
         "description": "Potential persistent contrail formation (non-normalised)"}
    )
    ds_out.per.attrs.update(
        {"units": "-", "long_name": "persistence",
         "description": "Where persistence requirement is met"}
    )
    ds_out.frz.attrs.update(
        {"units": "-", "long_name": "persistence",
         "description": "Where freezing requirement is met"}
    )
    ds_out.frm.attrs.update(
        {"units": "-", "long_name": "persistence",
         "description": "Where formation requirement is met"}
    )
    ds_out.attrs.update({"n_time": ds.time.size})

    return ds_out
