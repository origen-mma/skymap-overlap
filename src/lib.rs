//! # skymap-overlap
//!
//! Fast joint GW-GRB False Alarm Rate computation using empirical
//! skymap overlap p-values, using random skymap rotations.
//!
//! The current RAVEN method (dividing temporal FAR by the overlap integral)
//! produces biased FAR distributions. This crate implements the corrected
//! method: rotate the GRB skymap to random positions, compute the overlap
//! each time, and use the resulting empirical p-value with a remapping
//! formula to obtain a properly uniform FAR distribution.
//!
//! ## Features
//!
//! - **Sparse HEALPix skymaps**: only stores non-zero pixels, making
//!   rotation and overlap computation proportional to the localization
//!   area rather than the full sky.
//! - **Parallel Monte Carlo**: uses rayon to run rotation trials across
//!   all CPU cores.
//! - **FITS loading**: supports both flat HEALPix and multi-order MOC
//!   formats (LIGO, Fermi GBM, Swift, etc.).
//! - **Python bindings**: optional PyO3 bindings for use in gwcelery/RAVEN.
//!
//! ## Quick Start
//!
//! ```no_run
//! use skymap_overlap::{SparseSkymap, empirical_pvalue, far_remapped, GBM_RATE_HZ};
//!
//! let gw = SparseSkymap::from_fits("gw_skymap.fits").unwrap();
//! let grb = SparseSkymap::from_fits("grb_skymap.fits").unwrap();
//!
//! let result = empirical_pvalue(&gw, &grb, 1000, Some(42));
//! println!("p-value: {:.4}", result.p_value);
//!
//! let far = far_remapped(
//!     1e-7,          // GW FAR in Hz
//!     GBM_RATE_HZ,   // GRB rate
//!     600.0,         // time window in seconds
//!     result.p_value,
//!     2.0 / 86400.0, // FAR_max (2/day)
//! );
//! println!("Joint FAR: {:.2e} Hz", far);
//! ```

pub mod error;
pub mod far;
pub mod overlap;
pub mod rotation;
pub mod skymap;

#[cfg(feature = "python")]
mod python;

pub use error::Error;
pub use far::{far_raven, far_remapped, far_temporal, GBM_RATE_HZ, GBM_RATE_PER_YEAR};
pub use overlap::{empirical_pvalue, overlap_integral, PvalueResult};
pub use skymap::SparseSkymap;
