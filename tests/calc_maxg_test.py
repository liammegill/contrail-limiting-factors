"""
Provides tests for module calc_maxg
"""

import numpy as np
import xarray as xr
import pytest
import limfac as lf

class TestCalcGMax:
    """Tests the function `calc_g_max`."""

    def test_basic_functionality(self):
        """Tests basic functionality."""
        # use t_max values to calcualate an expected G_max
        t_max = np.array([220.0, 232.0, 235.15])
        ppw_t_max = lf.e_sat_water(t_max)
        g_exp = lf.e_sat_water_prime(t_max)

        # calculate new ambient values along G_max curve
        dt = np.array([-6.0, -5.0, -2.0])
        t = t_max + dt
        rhi_cor = 1.0
        pp_h2o = ppw_t_max - g_exp * (t_max - t)
        r = pp_h2o / lf.e_sat(t) * 100.0

        # compare
        g_res = lf.calc_g_max(t, r, rhi_cor)
        np.testing.assert_allclose(g_res, g_exp, atol=1e-3)

    def test_temp_boundary(self):
        """Tests the temperature boundary above 235.15 K."""
        t = 234.0  # very close to 235.15 K such that limit is reached
        r = 100.0
        rhi_cor = 1.0
        pp_h2o = r / 100.0 * lf.e_sat(t) / rhi_cor
        ppw_235 = lf.e_sat_water(235.15)
        g_exp = (ppw_235 - pp_h2o) / (235.15 - t)
        g_res = lf.calc_g_max(t, r, rhi_cor)
        np.testing.assert_allclose(g_res, g_exp, atol=1e-3)


class TestCalcHistArr:
    """Tests the function `calc_hist_arr`."""

    @pytest.fixture(scope="class")
    def ds(self):
        """Fixture to create basic dataset for testing."""
        level = np.array([350.0])  # [hPa]
        time = np.array([0.0])
        values = np.arange(5)
        t = np.array([[[220.0, 228.0, 233.0, 234.0, 234.2]]])
        r = np.array([[[100.0, 100.0, 100.0, 100.0, 100.0]]])
        latitudes = np.linspace(-89.0, 89.0, num=5)
        longitudes = np.linspace(0.0, 350.0, num=5)
        return xr.Dataset(
            {"t": (["time", "level", "values"], t),
             "r": (["time", "level", "values"], r)},
            coords={"values": values, "time": time, "level": level,
                    "latitude": (["values"], latitudes),
                    "longitude": (["values"], longitudes)}
        ).drop_vars("values")

    def test_basic_functionality(self, ds):
        """Tests basic functionality."""
        bin_edges = np.array([0, 2, 4, 6, 8, 10])
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        result = lf.calc_hist_arr(ds, bin_edges, bin_centres)
        # tests
        assert result.shape == (ds.level.size, 4, len(bin_centres)), "Output "\
            "has incorrect shape."
        assert result.sum() == 2 * ds.latitude.size, "Total histogram count "\
            "should match twice the input data points."


class TestCalcCumulativeHist:
    """Tests the function `calc_cumulative_hist`."""

    @pytest.fixture(scope="class")
    def ds(self):
        """Fixture to create basic dataset for testing."""
        seasons     = np.array(["2010DJF"])
        levels      = np.array([350.0])
        bin_centres = np.array([1., 3., 5., 7., 9.])
        tot_hist    = np.array([[[1., 1., 1., 1., 1.]]])
        xtropn_hist = np.array([[[0., 0., 0., 1., 1.]]])
        trop_hist   = np.array([[[0., 0., 1., 0., 0.]]])
        xtrops_hist = np.array([[[1., 1., 0., 0., 0.]]])
        return xr.Dataset(
            {"tot_hist": (["season", "level", "bin_centre"], tot_hist),
             "xtropN_hist": (["season", "level", "bin_centre"], xtropn_hist),
             "trop_hist": (["season", "level", "bin_centre"], trop_hist),
             "xtropS_hist": (["season", "level", "bin_centre"], xtrops_hist),
             "num_vals": (["season"], np.array([5]))},
             coords={"season": seasons, "level": levels,
                     "bin_centre": bin_centres}
        )

    def test_basic_functionality(self, ds):
        """Test basic functionality."""
        lat_band = "tot_hist"
        result = lf.calc_cumulative_hist(ds, lat_band)
        expected = np.array([[[0.2, 0.4, 0.6, 0.8, 1.0]]])
        assert result.shape == (ds.season.size, ds.level.size, ds.bin_centre.size)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_invalid_lat_band(self, ds):
        """Tests an invalid `lat_band` input."""
        lat_band = "invalid_input"
        with pytest.raises(AssertionError):
            lf.calc_cumulative_hist(ds, lat_band)
