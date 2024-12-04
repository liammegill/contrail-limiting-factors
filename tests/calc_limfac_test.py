"""
Provides tests for module calc_limfac
"""

import numpy as np
import xarray as xr
import pytest
import limfac as lf

class TestCalcTFrmLim:
    """Tests the function `calc_t_frm_lim`."""

    def test_basic_functionality(self):
        """Tests basic functionality."""
        g = np.array([1.5, 2.5])
        t = np.array([220.0, 240.0])
        pp_h2o = np.array([20.0, 30.0])
        result = lf.calc_t_frm_lim(g, t, pp_h2o)
        assert result.shape == g.shape, "Result should match input shape."
        assert np.all(result >= 0), "Result temperatures should be positive."

    @pytest.mark.parametrize("g", [1.8, 2.8])
    def test_t_max(self, g):
        """Tests function for T_max < 235.15 K."""
        t = 225.0
        pp_h2o = 4.5
        result = lf.calc_t_frm_lim(g, t, pp_h2o)
        t_max = lf.calc_single_t_max(0, g)
        assert np.all(result <= 235.15), "Result temperatures should be less "\
            "than the homogeneous freezing temperature (235.15 K)."
        assert np.all(result <= t_max), "Result temperature should be less "\
            "than T_max."


class TestCalcLimfacBools:
    """Tests the function `calc_limfac_bools`."""

    def test_basic_functionality(self):
        """Tests basic functionality."""
        g = 1.8
        t = np.array([225.0, 228.0])
        rh = np.array([100., 100.])
        result = lf.calc_limfac_bools(t, rh, g)
        assert all(
            key in result for key in ["frm", "frz", "per", "wss", "cont"]
            ), "Missing keys in returned dictionary."
        assert result["frm"].shape == t.shape, "Result shape should match input"
        assert result["cont"].dtype == bool, "Output 'cont' should be boolean"

    @pytest.mark.parametrize("t,frm_bool", [(225.0, True), (228.0, False)])
    def test_formation(self, t, frm_bool):
        """Tests the formation limiting factor calculation."""
        g = 1.8
        rh = 100.
        result = lf.calc_limfac_bools(t, rh, g)
        assert result["frm"] == frm_bool, "Incorrect formation calculation."
        assert result["cont"] == frm_bool, "Incorrect contrail boolean."

    @pytest.mark.parametrize("t,frz_bool", [(234.0, True), (236.0, False)])
    def test_freezing(self, t, frz_bool):
        """Tests the freezing limiting factor calculation."""
        g = 7.0
        rh = 130.0
        result = lf.calc_limfac_bools(t, rh, g)
        assert result["frz"] == frz_bool, "Incorrect freezing calculation."
        assert result["cont"] == frz_bool, "Incorrect contrail boolean."

    @pytest.mark.parametrize("rh,per_bool", [(105.0, True), (100.0, True),
                                             (99.0, False)])
    def test_persistence(self, rh, per_bool):
        """Tests the persistence limiting factor calculation."""
        g = 1.8
        t = 224.0
        result = lf.calc_limfac_bools(t, rh, g)
        assert result["per"] == per_bool, "Incorrect persistence calculation."
        assert result["cont"] == per_bool, "Incorrect contrail boolean."

    @pytest.mark.parametrize("rh,wss_bool", [(170.0, False), (140.0, True)])
    def test_water_supersaturation(self, rh, wss_bool):
        """Tests the water supersaturation limiting factor calculation."""
        g = 1.8
        t = 224.0
        result = lf.calc_limfac_bools(t, rh, g)
        assert result["wss"] == wss_bool, "Incorrect water supersaturation" \
            "calculation."
        assert result["cont"] == wss_bool, "Incorrect contrail boolean."


class TestCalcLimfacMatrix:
    """Tests the function `calc_limfac_matrix`."""

    @pytest.fixture(scope="class")
    def neighbors(self):
        """Fixture to create a test square of horizontal neighbors."""
        example = {
            0: {1: {'length': 1.0},
                2: {'length': 1.0},
                3: {'length': 1.0},
                4: {'length': 1.0}},
            1: {0: {'length': 1.0},
                2: {'length': 1.0},
                4: {'length': 1.0}},
            2: {0: {'length': 1.0},
                1: {'length': 1.0},
                3: {'length': 1.0}},
            3: {0: {'length': 1.0},
                2: {'length': 1.0},
                4: {'length': 1.0}},
            4: {0: {'length': 1.0},
                1: {'length': 1.0},
                3: {'length': 1.0}}
        }
        return example

    @pytest.fixture(scope="class")
    def perimeters(self):
        """Fixture to create the perimeters of the test square."""
        return np.array([4.0, 3.0, 3.0, 3.0, 3.0])

    def test_basic_case_1(self, neighbors, perimeters):
        """Test basic case 1: A persistent contrail forms only in cell 0."""
        cont_bool_flat = np.array([True, False, False, False, False])
        mask_flat = ~cont_bool_flat
        result = lf.calc_limfac_matrix(
            mask_flat, cont_bool_flat, neighbors, "length", perimeters
        )
        expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_basic_case_1_vertical(self, neighbors, perimeters):
        """Test basic case 1 in vertical direction."""
        areas = perimeters
        vert_neighbors = {  # change 'length' to 'area'
            outer_key: {
                inner_key: {
                    ('area' if key == 'length' else key): value
                    for key, value in inner_dict.items()
                }
                for inner_key, inner_dict in outer_dict.items()
            }
            for outer_key, outer_dict in neighbors.items()
        }
        cont_bool_flat = np.array([True, False, False, False, False])
        mask_flat = ~cont_bool_flat
        result = lf.calc_limfac_matrix(
            mask_flat, cont_bool_flat, vert_neighbors, "area", areas
        )
        expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_basic_case_2(self, neighbors, perimeters):
        """Test basic case 2: Persistent contrails form everywhere except in
        cell 0."""
        cont_bool_flat = np.array([False, True, True, True, True])
        mask_flat = ~cont_bool_flat
        result = lf.calc_limfac_matrix(
            mask_flat, cont_bool_flat, neighbors, "length", perimeters
        )
        expected = np.array([0.0, 0.333, 0.333, 0.333, 0.333])
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_no_contrails(self, neighbors, perimeters):
        """Test case where cont_bool_flat is False everywhere."""
        cont_bool_flat = np.array([False, False, False, False, False])
        mask_flat = ~cont_bool_flat
        result = lf.calc_limfac_matrix(
            mask_flat, cont_bool_flat, neighbors, "length", perimeters
        )
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_equal(result, expected)

    def test_always_contrails(self, neighbors, perimeters):
        """Test case where cont_bool_flat is True everywhere."""
        cont_bool_flat = np.array([True, True, True, True, True])
        mask_flat = ~cont_bool_flat
        result = lf.calc_limfac_matrix(
            mask_flat, cont_bool_flat, neighbors, "length", perimeters
        )
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_equal(result, expected)

    def test_no_masked_neigbors(self, neighbors, perimeters):
        """Test case where mask_flat is False everywhere."""
        cont_bool_flat = np.array([True, False, False, False, False])
        mask_flat = np.array([False, False, False, False, False])
        result = lf.calc_limfac_matrix(
            mask_flat, cont_bool_flat, neighbors, "length", perimeters
        )
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_equal(result, expected)

    def test_empty_inputs(self):
        """Test case with empty inputs."""
        cont_bool_flat = np.array([])
        mask_flat = np.array([])
        neighbors = {}
        perimeters = np.array([])
        with pytest.raises(AssertionError):
            lf.calc_limfac_matrix(mask_flat, cont_bool_flat, neighbors,
                                  "", perimeters)

    def test_mismatched_inputs(self, neighbors, perimeters):
        """Test case where cont_bool and mask length is mismatched to
        neighbors."""
        cont_bool_flat = np.array([True, False, False, False, False, False])
        mask_flat = ~cont_bool_flat
        with pytest.raises(AssertionError):
            lf.calc_limfac_matrix(mask_flat, cont_bool_flat, neighbors,
                                  "", perimeters)


class TestCalcLimfacs:
    """Tests the function `calc_limfacs`."""

    @pytest.fixture(scope="class")
    def neighbors(self):
        """Fixture to create a test square of horizontal neighbors."""
        example = {
            0: {1: {'length': 1.0},
                2: {'length': 1.0},
                3: {'length': 1.0},
                4: {'length': 1.0}},
            1: {0: {'length': 1.0},
                2: {'length': 1.0},
                4: {'length': 1.0}},
            2: {0: {'length': 1.0},
                1: {'length': 1.0},
                3: {'length': 1.0}},
            3: {0: {'length': 1.0},
                2: {'length': 1.0},
                4: {'length': 1.0}},
            4: {0: {'length': 1.0},
                1: {'length': 1.0},
                3: {'length': 1.0}}
        }
        return example

    @pytest.fixture(scope="class")
    def perimeters(self):
        """Fixture to create the perimeters of the test square."""
        return np.array([4.0, 3.0, 3.0, 3.0, 3.0])

    @pytest.fixture(scope="class")
    def ac(self):
        """Fixture to create an example aircraft design specifications."""
        return xr.Dataset(
            {"fuel": "JA1", "cp": 1004.0, "cp_mol": None, "EI_H2O": 1.25,
             "Q": 43.6e6, "dH_mol": None, "eta": 0.4, "eps": 0.622, "R": None}
        )

    @pytest.fixture(scope="class")
    def ds(self):
        """Fixture to create basic dataset for testing."""
        level = 350.0  # [hPa]
        time = 0.0
        values = np.arange(5)
        t = np.array([228.0, 231.0, 228.0, 228.0, 240.0])
        r = np.array([100.0, 100.0, 99.0, 160.0, 100.0])
        latitudes = np.linspace(-89.0, 89.0, num=5)
        longitudes = np.linspace(0.0, 350.0, num=5)
        return xr.Dataset(
            {"t": (["values"], t), "r": (["values"], r)},
            coords={"values": values, "time": time, "level": level,
                    "latitude": (["values"], latitudes),
                    "longitude": (["values"], longitudes)}
        ).drop_vars("values")

    def test_basic_case_1(self, neighbors, perimeters, ac, ds):
        """Test basic case 1: A persistent contrail forms only in cell 0."""
        ds["t"].values = np.array([228.0, 231.0, 228.0, 228.0, 240.0])
        ds["r"].values = np.array([100.0, 100.0, 99.0, 160.0, 100.0])
        result = lf.calc_limfacs(ds, ac, "h", neighbors, perimeters)
        np.testing.assert_equal(result["ppcf"], np.array([1., 0., 0., 0., 0.]))
        np.testing.assert_equal(result["limfac_tot"],  # all neighbours
                                np.array([1., 0., 0., 0., 0.]))
        np.testing.assert_equal(result["limfac_frm"],  # two neighbours
                                np.array([0.5, 0., 0., 0., 0.]))
        np.testing.assert_equal(result["limfac_frz"],  # one neighbour
                                np.array([0.25, 0., 0., 0., 0.]))
        np.testing.assert_equal(result["limfac_per"],  # one neighbour
                                np.array([0.25, 0., 0., 0., 0.]))
        np.testing.assert_equal(result["limfac_wss"],  # one neighbour
                                np.array([0.25, 0., 0., 0., 0.]))

    def test_basic_case_2(self, neighbors, perimeters, ac, ds):
        """Test basic case 2: Persistent contrails form everywhere except in
        cell 0."""
        ds["t"].values = np.array([240.0, 228.0, 228.0, 228.0, 228.0])
        ds["r"].values = np.array([99.0, 100.0, 100.0, 100.0, 100.0])
        result = lf.calc_limfacs(ds, ac, "h", neighbors, perimeters)
        np.testing.assert_equal(result["ppcf"], np.array([0., 1., 1., 1., 1.]))
        for lf_var in ["limfac_tot", "limfac_frm", "limfac_frz", "limfac_per"]:
            np.testing.assert_allclose(result[lf_var],
                                    np.array([0.0, 0.333, 0.333, 0.333, 0.333]),
                                    atol=1e-3)
        np.testing.assert_equal(result["limfac_wss"],
                                np.array([0., 0., 0., 0., 0.]))

    def test_no_contrails(self, neighbors, perimeters, ac, ds):
        """Test case where no contrails form."""
        ds["t"].values = np.array([240.0, 240.0, 240.0, 240.0, 240.0])
        ds["r"].values = np.array([99.0, 99.0, 99.0, 99.0, 99.0])
        result = lf.calc_limfacs(ds, ac, "h", neighbors, perimeters)
        for lf_var in ["limfac_tot", "limfac_frm", "limfac_frz", "limfac_per",
                       "limfac_wss", "ppcf"]:
            np.testing.assert_equal(result[lf_var],
                                    np.array([0., 0., 0., 0., 0.]))

    def test_always_contrails(self, neighbors, perimeters, ac, ds):
        """Test case where contrails always form."""
        ds["t"].values = np.array([228.0, 228.0, 228.0, 228.0, 228.0])
        ds["r"].values = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        result = lf.calc_limfacs(ds, ac, "h", neighbors, perimeters)
        np.testing.assert_equal(result["ppcf"], np.array([1., 1., 1., 1., 1.]))
        for lf_var in ["limfac_tot", "limfac_frm", "limfac_frz", "limfac_per",
                       "limfac_wss"]:
            np.testing.assert_equal(result[lf_var],
                                    np.array([0., 0., 0., 0., 0.]))

    def test_multiple_times(self, neighbors, perimeters, ac, ds):
        """Test basic case 1 for time coordinate with multiple values."""
        ds["t"].values = np.array([228.0, 231.0, 228.0, 228.0, 240.0])
        ds["r"].values = np.array([100.0, 100.0, 99.0, 160.0, 100.0])
        expanded_time = np.array([0.0, 1.0])
        ds = ds.expand_dims({"time": expanded_time})
        result = lf.calc_limfacs(ds, ac, "h", neighbors, perimeters)

        # check output shape
        assert "values" in result.sizes, "Dimension 'values' is missing in" \
            "dataset."
        assert result.sizes["values"] == 5, "Incorrect output size."

        # check numerical results
        np.testing.assert_equal(result["ppcf"], np.array([2., 0., 0., 0., 0.]))
        np.testing.assert_equal(result["limfac_tot"],  # all neighbours
                                np.array([2., 0., 0., 0., 0.]))
        np.testing.assert_equal(result["limfac_frm"],  # two neighbours
                                np.array([1., 0., 0., 0., 0.]))
        np.testing.assert_equal(result["limfac_frz"],  # one neighbour
                                np.array([0.5, 0., 0., 0., 0.]))
        np.testing.assert_equal(result["limfac_per"],  # one neighbour
                                np.array([0.5, 0., 0., 0., 0.]))
        np.testing.assert_equal(result["limfac_wss"],  # one neighbour
                                np.array([0.5, 0., 0., 0., 0.]))
        assert result.n_time == 2, "Incorrect n_time value."

    def test_multiple_levels(self, neighbors, perimeters, ac, ds):
        """Test basic case 1 for level coordinate with multiple values."""
        ds["t"].values = np.array([228.0, 231.0, 228.0, 228.0, 240.0])
        ds["r"].values = np.array([100.0, 100.0, 99.0, 160.0, 100.0])
        expanded_level = np.array([350., 350.])  # twice same level
        ds = ds.expand_dims({"level": expanded_level})
        result = lf.calc_limfacs(ds, ac, "h", neighbors, perimeters)

        # check output shape
        expected_sizes = {"level": 2, "values": 5}
        for dim, size in expected_sizes.items():
            assert dim in result.sizes, f"Dimension '{dim}' missing in output."
            assert result.sizes[dim] == size, f"Dimension '{dim}' has size " \
                f"{ds.sizes[dim]} but should have size {size}."

        # check numerical results
        for i_lvl in range(2):
            res = result.isel(level=i_lvl)
            np.testing.assert_equal(res["ppcf"],
                                    np.array([1., 0., 0., 0., 0.]))
            np.testing.assert_equal(res["limfac_tot"],  # all neighbours
                                    np.array([1., 0., 0., 0., 0.]))
            np.testing.assert_equal(res["limfac_frm"],  # two neighbours
                                    np.array([0.5, 0., 0., 0., 0.]))
            np.testing.assert_equal(res["limfac_frz"],  # one neighbour
                                    np.array([0.25, 0., 0., 0., 0.]))
            np.testing.assert_equal(res["limfac_per"],  # one neighbour
                                    np.array([0.25, 0., 0., 0., 0.]))
            np.testing.assert_equal(res["limfac_wss"],  # one neighbour
                                    np.array([0.25, 0., 0., 0., 0.]))

    def test_multiple_times_levels(self, neighbors, perimeters, ac, ds):
        """Test basic case 1 for multiple level and time values."""
        ds["t"].values = np.array([228.0, 231.0, 228.0, 228.0, 240.0])
        ds["r"].values = np.array([100.0, 100.0, 99.0, 160.0, 100.0])
        expanded_time = np.array([0.0, 1.0])
        expanded_level = np.array([350.0, 350.0])
        ds = ds.expand_dims({"time": expanded_time, "level": expanded_level})
        result = lf.calc_limfacs(ds, ac, "h", neighbors, perimeters)

        # check output shape
        expected_sizes = {"level": 2, "values": 5}
        for dim, size in expected_sizes.items():
            assert dim in result.sizes, f"Dimension '{dim}' missing in output."
            assert result.sizes[dim] == size, f"Dimension '{dim}' has size " \
                f"{ds.sizes[dim]} but should have size {size}."
        assert result.n_time == 2, "Incorrect n_time value."

        # check numerical results
        for i_lvl in range(2):
            res = result.isel(level=i_lvl)
            np.testing.assert_equal(res["ppcf"],
                                    np.array([2., 0., 0., 0., 0.]))
            np.testing.assert_equal(res["limfac_tot"],  # all neighbours
                                    np.array([2., 0., 0., 0., 0.]))
            np.testing.assert_equal(res["limfac_frm"],  # two neighbours
                                    np.array([1., 0., 0., 0., 0.]))
            np.testing.assert_equal(res["limfac_frz"],  # one neighbour
                                    np.array([0.5, 0., 0., 0., 0.]))
            np.testing.assert_equal(res["limfac_per"],  # one neighbour
                                    np.array([0.5, 0., 0., 0., 0.]))
            np.testing.assert_equal(res["limfac_wss"],  # one neighbour
                                    np.array([0.5, 0., 0., 0., 0.]))

    def test_vertical_case_1(self, ac, ds):
        """Test vertical base case 1."""
        # create vertical areas and neighbors for testing
        areas = np.array([1.0, 1.0])
        vert_neighbors = {
            0: {2: {'area': 1.0}},
            1: {3: {'area': 1.0}},
            2: {0: {'area': 1.0},
                4: {'area': 1.0}},
            3: {1: {'area': 1.0},
                5: {'area': 1.0}},
            4: {2: {'area': 1.0}},
            5: {3: {'area': 1.0}}
        }

        # create vertical ds and calculate result
        t = np.array([[228.0, 228.0], [228.0, 236.0], [228.0, 228.0]])
        r = np.array([[99.0, 100.0], [100.0, 100.0], [160.0, 100.0]])
        level = np.array([351.0, 350.0, 349.0])
        ds = xr.Dataset(
            {"t": (["level", "values"], t), "r": (["level", "values"], r)},
            coords={"time": 0.0, "level": level, "values": np.empty(2),
                    "latitude": (["values"], np.array([-20.0, 20.0])),
                    "longitude": (["values"], np.array([0.0, 180.0]))}
        ).drop_vars("values")
        result = lf.calc_limfacs(ds, ac, "v", vert_neighbors, areas)

        # check output shape
        expected_sizes = {"level": 3, "values": 2}
        for dim, size in expected_sizes.items():
            assert dim in result.sizes, f"Dimension '{dim}' missing in output."
            assert result.sizes[dim] == size, f"Dimension '{dim}' has size " \
                f"{ds.sizes[dim]} but should have size {size}."
        assert result.n_time == 1, "Incorrect n_time value."

        # check numerical results
        np.testing.assert_equal(
            result["ppcf"], np.array([[0., 1.], [1., 0.], [0., 1.]])
        )
        np.testing.assert_equal(
            result["limfac_tot"], np.array([[0., 1.], [2., 0.], [0., 1.]])
        )
        np.testing.assert_equal(
            result["limfac_frm"], np.array([[0., 1.], [0., 0.], [0., 1.]])
        )
        np.testing.assert_equal(
            result["limfac_frz"], np.array([[0., 1.], [0., 0.], [0., 1.]])
        )
        np.testing.assert_equal(
            result["limfac_per"], np.array([[0., 0.], [1., 0.], [0., 0.]])
        )
        np.testing.assert_equal(
            result["limfac_wss"], np.array([[0., 0.], [1., 0.], [0., 0.]])
        )
