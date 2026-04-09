"""Tests for the skymap_overlap Python bindings."""

import numpy as np
import pytest

import skymap_overlap


class TestSkymap:
    """Tests for the Skymap class."""

    def test_from_dense_basic(self):
        nside = 4
        npix = 12 * nside * nside
        probs = [0.0] * npix
        probs[0] = 0.5
        probs[10] = 0.3
        probs[100] = 0.2

        sky = skymap_overlap.Skymap.from_dense(nside, probs)
        assert sky.nside == nside
        assert sky.nnz == 3
        assert sky.npix == npix
        assert sky.depth == 2  # log2(4)

    def test_from_dense_normalization(self):
        """Probabilities should be normalized to sum to 1."""
        nside = 4
        npix = 12 * nside * nside
        probs = [0.0] * npix
        probs[0] = 2.0
        probs[1] = 3.0

        sky = skymap_overlap.Skymap.from_dense(nside, probs)
        # After normalization: 2/5 and 3/5
        assert abs(sky.probability_at(0) - 0.4) < 1e-10
        assert abs(sky.probability_at(1) - 0.6) < 1e-10

    def test_from_dense_wrong_length(self):
        with pytest.raises(ValueError, match="Expected 192"):
            skymap_overlap.Skymap.from_dense(4, [1.0, 2.0])

    def test_probability_at_zero(self):
        nside = 4
        npix = 12 * nside * nside
        probs = [0.0] * npix
        probs[0] = 1.0

        sky = skymap_overlap.Skymap.from_dense(nside, probs)
        assert sky.probability_at(0) == pytest.approx(1.0)
        assert sky.probability_at(1) == 0.0
        assert sky.probability_at(100) == 0.0

    def test_max_prob_position(self):
        nside = 4
        npix = 12 * nside * nside
        probs = [0.0] * npix
        probs[42] = 1.0

        sky = skymap_overlap.Skymap.from_dense(nside, probs)
        ra, dec = sky.max_prob_position()
        assert isinstance(ra, float)
        assert isinstance(dec, float)
        assert 0.0 <= ra < 360.0
        assert -90.0 <= dec <= 90.0

    def test_repr(self):
        nside = 4
        npix = 12 * nside * nside
        probs = [0.0] * npix
        probs[0] = 1.0

        sky = skymap_overlap.Skymap.from_dense(nside, probs)
        r = repr(sky)
        assert "Skymap" in r
        assert "nside=4" in r
        assert "nnz=1" in r

    def test_from_fits_missing_file(self):
        with pytest.raises(ValueError):
            skymap_overlap.Skymap.from_fits("/nonexistent/path.fits")


class TestOverlap:
    """Tests for the overlap function."""

    def _make_skymap(self, nside, pixel_indices):
        """Helper to create a uniform skymap over given pixel indices."""
        npix = 12 * nside * nside
        probs = [0.0] * npix
        for i in pixel_indices:
            probs[i] = 1.0
        return skymap_overlap.Skymap.from_dense(nside, probs)

    def test_self_overlap(self):
        """Self-overlap = sum(p_i^2) = N * (1/N)^2 = 1/N."""
        n_pix = 50
        sky = self._make_skymap(64, range(1000, 1000 + n_pix))
        ov = skymap_overlap.overlap(sky, sky)
        expected = 1.0 / n_pix
        assert ov == pytest.approx(expected, rel=1e-8)

    def test_disjoint_overlap(self):
        """Non-overlapping skymaps should have zero overlap."""
        a = self._make_skymap(64, range(0, 50))
        b = self._make_skymap(64, range(10000, 10050))
        ov = skymap_overlap.overlap(a, b)
        assert ov == pytest.approx(0.0, abs=1e-15)

    def test_overlap_symmetric(self):
        a = self._make_skymap(32, range(0, 30))
        b = self._make_skymap(32, range(20, 50))
        assert skymap_overlap.overlap(a, b) == pytest.approx(
            skymap_overlap.overlap(b, a), rel=1e-12
        )

    def test_overlap_partial(self):
        """Partially overlapping skymaps."""
        a = self._make_skymap(32, range(0, 100))
        b = self._make_skymap(32, range(50, 150))
        ov = skymap_overlap.overlap(a, b)
        # 50 shared pixels, each with prob 1/100 in both maps
        expected = 50.0 * (1.0 / 100.0) * (1.0 / 100.0)
        assert ov == pytest.approx(expected, rel=1e-8)


class TestPvalue:
    """Tests for the pvalue function."""

    def _make_skymap(self, nside, pixel_indices):
        npix = 12 * nside * nside
        probs = [0.0] * npix
        for i in pixel_indices:
            probs[i] = 1.0
        return skymap_overlap.Skymap.from_dense(nside, probs)

    def test_basic_pvalue(self):
        sky_a = self._make_skymap(32, range(500, 520))
        sky_b = self._make_skymap(32, range(500, 520))

        result = skymap_overlap.pvalue(sky_a, sky_b, n_trials=100, seed=42)
        assert 0.0 <= result.p_value <= 1.0
        assert result.n_trials == 100
        assert result.n_above >= 0
        assert result.n_above <= result.n_trials
        assert result.observed_overlap > 0

    def test_trial_overlaps_numpy(self):
        sky_a = self._make_skymap(16, range(0, 20))
        sky_b = self._make_skymap(16, range(0, 20))

        result = skymap_overlap.pvalue(sky_a, sky_b, n_trials=50, seed=123)
        assert isinstance(result.trial_overlaps, np.ndarray)
        assert result.trial_overlaps.shape == (50,)
        assert result.trial_overlaps.dtype == np.float64

    def test_reproducible_with_seed(self):
        sky_a = self._make_skymap(16, range(0, 20))
        sky_b = self._make_skymap(16, range(10, 30))

        r1 = skymap_overlap.pvalue(sky_a, sky_b, n_trials=50, seed=999)
        r2 = skymap_overlap.pvalue(sky_a, sky_b, n_trials=50, seed=999)
        assert r1.p_value == r2.p_value
        np.testing.assert_allclose(r1.trial_overlaps, r2.trial_overlaps, atol=1e-15)

    def test_pvalue_result_repr(self):
        sky = self._make_skymap(16, range(0, 10))
        result = skymap_overlap.pvalue(sky, sky, n_trials=10, seed=1)
        r = repr(result)
        assert "PvalueResult" in r
        assert "p_value" in r


class TestFAR:
    """Tests for FAR functions."""

    def test_gbm_rate_constant(self):
        assert skymap_overlap.GBM_RATE_HZ > 0
        assert skymap_overlap.GBM_RATE_HZ == pytest.approx(
            325.0 / (365.25 * 24 * 3600), rel=1e-6
        )

    def test_gbm_rate_per_year(self):
        assert skymap_overlap.GBM_RATE_PER_YEAR == 325.0

    def test_far_raven_basic(self):
        far = skymap_overlap.far_raven(
            far_gw=1e-7,
            grb_rate=skymap_overlap.GBM_RATE_HZ,
            time_window=600.0,
            overlap_val=0.01,
        )
        assert far > 0
        assert np.isfinite(far)

    def test_far_raven_higher_overlap_lower_far(self):
        """Higher overlap should give lower RAVEN FAR."""
        far_low = skymap_overlap.far_raven(1e-7, skymap_overlap.GBM_RATE_HZ, 600.0, 0.1)
        far_high = skymap_overlap.far_raven(1e-7, skymap_overlap.GBM_RATE_HZ, 600.0, 0.01)
        assert far_low < far_high

    def test_far_remapped_basic(self):
        far = skymap_overlap.far_remapped(
            far_gw=1e-7,
            grb_rate=skymap_overlap.GBM_RATE_HZ,
            time_window=600.0,
            p_value=0.05,
            far_gw_max=2.0 / 86400.0,
        )
        assert far > 0
        assert np.isfinite(far)

    def test_far_remapped_lower_p_lower_far(self):
        """Lower p-value should give lower FAR."""
        far_high_p = skymap_overlap.far_remapped(
            1e-7, skymap_overlap.GBM_RATE_HZ, 600.0, 0.1, 2.0 / 86400.0
        )
        far_low_p = skymap_overlap.far_remapped(
            1e-7, skymap_overlap.GBM_RATE_HZ, 600.0, 0.001, 2.0 / 86400.0
        )
        assert far_low_p < far_high_p

    def test_far_temporal(self):
        far = skymap_overlap.far_temporal(
            far_gw=1e-7,
            grb_rate=skymap_overlap.GBM_RATE_HZ,
            time_window=600.0,
        )
        expected = 1e-7 * skymap_overlap.GBM_RATE_HZ * 600.0
        assert far == pytest.approx(expected, rel=1e-10)

    def test_far_remapped_vs_temporal(self):
        """With p=1, corrected FAR should exceed temporal FAR."""
        far_t = skymap_overlap.far_temporal(1e-7, skymap_overlap.GBM_RATE_HZ, 600.0)
        far_c = skymap_overlap.far_remapped(
            1e-7, skymap_overlap.GBM_RATE_HZ, 600.0, 1.0, 2.0 / 86400.0
        )
        assert far_c >= far_t


class TestFITS:
    """Tests using real FITS skymap fixtures."""

    GW_PATH = "tests/fixtures/gw_skymap.fits"
    GRB_PATH = "tests/fixtures/grb_skymap.fits"

    def test_load_moc_skymap(self):
        gw = skymap_overlap.Skymap.from_fits(self.GW_PATH)
        assert gw.nside == 64
        assert gw.nnz > 0

    def test_load_flat_skymap(self):
        grb = skymap_overlap.Skymap.from_fits(self.GRB_PATH)
        assert grb.nside == 32
        assert grb.nnz > 0

    def test_overlap_real_skymaps(self):
        gw = skymap_overlap.Skymap.from_fits(self.GW_PATH)
        grb = skymap_overlap.Skymap.from_fits(self.GRB_PATH)
        ov = skymap_overlap.overlap(gw, grb)
        assert ov > 0
        assert ov < 1

    def test_pvalue_real_skymaps(self):
        gw = skymap_overlap.Skymap.from_fits(self.GW_PATH)
        grb = skymap_overlap.Skymap.from_fits(self.GRB_PATH)
        result = skymap_overlap.pvalue(gw, grb, n_trials=50, seed=42)
        assert 0 <= result.p_value <= 1
        assert result.trial_overlaps.shape == (50,)

    def test_end_to_end_far(self):
        gw = skymap_overlap.Skymap.from_fits(self.GW_PATH)
        grb = skymap_overlap.Skymap.from_fits(self.GRB_PATH)
        result = skymap_overlap.pvalue(gw, grb, n_trials=100, seed=42)
        far = skymap_overlap.far_remapped(
            1e-7, skymap_overlap.GBM_RATE_HZ, 600.0,
            result.p_value, 2.0 / 86400.0,
        )
        assert far > 0
        assert np.isfinite(far)
