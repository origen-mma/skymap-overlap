//! Python bindings for skymap-overlap via PyO3.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::far;
use crate::overlap as overlap_mod;
use crate::skymap;

/// A sparse HEALPix skymap storing only non-zero pixels.
///
/// Load from a FITS file with :meth:`Skymap.from_fits`, or construct
/// from a dense probability array with :meth:`Skymap.from_dense`.
#[pyclass(name = "Skymap")]
#[derive(Clone)]
pub struct PySkymap {
    inner: skymap::SparseSkymap,
}

#[pymethods]
impl PySkymap {
    /// Load a HEALPix skymap from a FITS file.
    ///
    /// Supports both flat HEALPix maps (LIGO, Fermi GBM) and multi-order
    /// MOC maps (bayestar, etc.).
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to the FITS file.
    ///
    /// Returns
    /// -------
    /// Skymap
    #[staticmethod]
    fn from_fits(path: &str) -> PyResult<Self> {
        let inner = skymap::SparseSkymap::from_fits(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PySkymap { inner })
    }

    /// Create a skymap from a dense probability array.
    ///
    /// Parameters
    /// ----------
    /// nside : int
    ///     HEALPix NSIDE parameter (must be a power of 2).
    /// probs : list[float]
    ///     Probability for each pixel (length must be 12 * nside^2).
    ///
    /// Returns
    /// -------
    /// Skymap
    #[staticmethod]
    fn from_dense(nside: u32, probs: Vec<f64>) -> PyResult<Self> {
        let expected = 12 * (nside as usize) * (nside as usize);
        if probs.len() != expected {
            return Err(PyValueError::new_err(format!(
                "Expected {} probabilities for nside={}, got {}",
                expected,
                nside,
                probs.len()
            )));
        }
        let inner = skymap::SparseSkymap::from_dense(nside, &probs);
        Ok(PySkymap { inner })
    }

    /// HEALPix NSIDE parameter.
    #[getter]
    fn nside(&self) -> u32 {
        self.inner.nside
    }

    /// HEALPix depth (log2 of NSIDE).
    #[getter]
    fn depth(&self) -> u8 {
        self.inner.depth
    }

    /// Number of non-zero pixels.
    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Total number of pixels at this NSIDE.
    #[getter]
    fn npix(&self) -> usize {
        self.inner.npix()
    }

    /// Look up probability at a nested pixel index.
    ///
    /// Parameters
    /// ----------
    /// nested_idx : int
    ///     HEALPix nested pixel index.
    ///
    /// Returns
    /// -------
    /// float
    fn probability_at(&self, nested_idx: u64) -> f64 {
        self.inner.probability_at(nested_idx)
    }

    /// Look up probability at a sky position.
    ///
    /// Parameters
    /// ----------
    /// ra_deg : float
    ///     Right ascension in degrees.
    /// dec_deg : float
    ///     Declination in degrees.
    ///
    /// Returns
    /// -------
    /// float
    fn probability_at_position(&self, ra_deg: f64, dec_deg: f64) -> f64 {
        self.inner.probability_at_position(ra_deg, dec_deg)
    }

    /// Position (RA, Dec) in degrees of the maximum-probability pixel.
    ///
    /// Returns
    /// -------
    /// tuple[float, float]
    fn max_prob_position(&self) -> (f64, f64) {
        self.inner.max_prob_position()
    }

    fn __repr__(&self) -> String {
        format!(
            "Skymap(nside={}, nnz={}, npix={})",
            self.inner.nside,
            self.inner.nnz(),
            self.inner.npix()
        )
    }
}

/// Result of an empirical p-value computation.
#[pyclass(name = "PvalueResult")]
pub struct PyPvalueResult {
    inner: overlap_mod::PvalueResult,
}

#[pymethods]
impl PyPvalueResult {
    /// Observed overlap integral (before rotation trials).
    #[getter]
    fn observed_overlap(&self) -> f64 {
        self.inner.observed_overlap
    }

    /// Empirical p-value: fraction of trials with overlap >= observed.
    #[getter]
    fn p_value(&self) -> f64 {
        self.inner.p_value
    }

    /// Number of rotation trials performed.
    #[getter]
    fn n_trials(&self) -> usize {
        self.inner.n_trials
    }

    /// Number of trials with overlap >= observed.
    #[getter]
    fn n_above(&self) -> usize {
        self.inner.n_above
    }

    /// Trial overlap values as a numpy array.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     1-D float64 array of length n_trials.
    #[getter]
    fn trial_overlaps<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let numpy = py.import_bound("numpy")?;
        let list = PyList::new_bound(py, &self.inner.trial_overlaps);
        let arr = numpy.call_method1("array", (list,))?;
        Ok(arr.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "PvalueResult(p_value={:.6}, observed_overlap={:.6e}, n_trials={}, n_above={})",
            self.inner.p_value,
            self.inner.observed_overlap,
            self.inner.n_trials,
            self.inner.n_above
        )
    }
}

/// Compute the overlap integral between two skymaps.
///
/// The overlap integral is defined as:
///
///     I_Omega = sum_i p_gw(i) * p_grb(i)
///
/// For skymaps at different NSIDEs, the coarser map is automatically
/// upsampled.
///
/// Parameters
/// ----------
/// gw : Skymap
///     Gravitational wave skymap.
/// grb : Skymap
///     Gamma-ray burst skymap.
///
/// Returns
/// -------
/// float
///     Overlap integral value.
#[pyfunction]
fn overlap(gw: &PySkymap, grb: &PySkymap) -> f64 {
    overlap_mod::overlap_integral(&gw.inner, &grb.inner)
}

/// Compute an empirical p-value for the spatial overlap of two skymaps.
///
/// Rotates the GRB skymap to random positions and computes the overlap
/// integral with the GW skymap each time. The p-value is the fraction
/// of trials where the overlap >= the observed overlap.
///
/// Uses all available CPU cores via rayon.
///
/// Parameters
/// ----------
/// gw : Skymap
///     Gravitational wave skymap (held fixed).
/// grb : Skymap
///     Gamma-ray burst skymap (rotated randomly).
/// n_trials : int
///     Number of random rotation trials.
/// seed : int, optional
///     RNG seed for reproducibility. Defaults to 42.
///
/// Returns
/// -------
/// PvalueResult
#[pyfunction]
#[pyo3(signature = (gw, grb, n_trials, seed=None))]
fn pvalue(gw: &PySkymap, grb: &PySkymap, n_trials: usize, seed: Option<u64>) -> PyPvalueResult {
    let inner = overlap_mod::empirical_pvalue(&gw.inner, &grb.inner, n_trials, seed);
    PyPvalueResult { inner }
}

/// Corrected joint FAR using empirical p-values (Eq 3, Piotrzkowski 2023).
///
///     FAR_c = FAR_gw * R_grb * dt * p * [1 - ln(FAR_gw * p / FAR_gw_max)]
///
/// Parameters
/// ----------
/// far_gw : float
///     GW false alarm rate in Hz.
/// grb_rate : float
///     GRB detection rate in Hz.
/// time_window : float
///     Coincidence time window in seconds.
/// p_value : float
///     Empirical spatial p-value from rotation trials.
/// far_gw_max : float
///     Maximum GW FAR threshold of the pipeline in Hz.
///
/// Returns
/// -------
/// float
///     Joint FAR in Hz.
#[pyfunction]
fn far_remapped(
    far_gw: f64,
    grb_rate: f64,
    time_window: f64,
    p_value: f64,
    far_gw_max: f64,
) -> f64 {
    far::far_remapped(far_gw, grb_rate, time_window, p_value, far_gw_max)
}

/// Original RAVEN joint FAR (Eq 1, Piotrzkowski 2023).
///
///     FAR_c = FAR_gw * R_grb * dt / I_Omega
///
/// This formula is known to produce biased FAR distributions. Prefer
/// :func:`far_remapped` for production use.
///
/// Parameters
/// ----------
/// far_gw : float
///     GW false alarm rate in Hz.
/// grb_rate : float
///     GRB detection rate in Hz.
/// time_window : float
///     Coincidence time window in seconds.
/// overlap : float
///     Skymap overlap integral I_Omega.
///
/// Returns
/// -------
/// float
///     Joint FAR in Hz.
#[pyfunction]
fn far_raven(far_gw: f64, grb_rate: f64, time_window: f64, overlap_val: f64) -> f64 {
    far::far_raven(far_gw, grb_rate, time_window, overlap_val)
}

/// Temporal-only joint FAR (no spatial information).
///
///     FAR_t = FAR_gw * R_grb * dt
///
/// Parameters
/// ----------
/// far_gw : float
///     GW false alarm rate in Hz.
/// grb_rate : float
///     GRB detection rate in Hz.
/// time_window : float
///     Coincidence time window in seconds.
///
/// Returns
/// -------
/// float
///     Temporal joint FAR in Hz.
#[pyfunction]
fn far_temporal(far_gw: f64, grb_rate: f64, time_window: f64) -> f64 {
    far::far_temporal(far_gw, grb_rate, time_window)
}

/// Python module for skymap-overlap.
#[pymodule]
fn skymap_overlap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySkymap>()?;
    m.add_class::<PyPvalueResult>()?;
    m.add_function(wrap_pyfunction!(overlap, m)?)?;
    m.add_function(wrap_pyfunction!(pvalue, m)?)?;
    m.add_function(wrap_pyfunction!(far_remapped, m)?)?;
    m.add_function(wrap_pyfunction!(far_raven, m)?)?;
    m.add_function(wrap_pyfunction!(far_temporal, m)?)?;
    m.add("GBM_RATE_HZ", far::GBM_RATE_HZ)?;
    m.add("GBM_RATE_PER_YEAR", far::GBM_RATE_PER_YEAR)?;
    Ok(())
}
