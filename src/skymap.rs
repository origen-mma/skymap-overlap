//! Sparse HEALPix skymap representation and FITS loading.

use crate::error::{Error, Result};
use cdshealpix::nested::{center, hash};
use fitsio::FitsFile;
use std::path::Path;

/// A sparse HEALPix skymap storing only non-zero pixels.
///
/// This is much more efficient than a dense representation for typical
/// GRB skymaps where <1% of pixels have non-zero probability.
#[derive(Debug, Clone)]
pub struct SparseSkymap {
    /// HEALPix NSIDE parameter (power of 2).
    pub nside: u32,
    /// HEALPix depth = log2(nside).
    pub depth: u8,
    /// Non-zero pixels as (nested_index, probability), sorted by index.
    pub pixels: Vec<(u64, f64)>,
}

impl SparseSkymap {
    /// Load a HEALPix skymap from a FITS file.
    ///
    /// Supports both flat HEALPix maps (with NSIDE keyword) and
    /// multi-order MOC maps (with UNIQ column).
    pub fn from_fits<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut fptr = FitsFile::open(&path).map_err(|e| Error::Fits(e.to_string()))?;

        // Try to find NSIDE in HDU 0 first, then HDU 1 (Fermi GBM maps
        // have it in the extension, not the primary).
        let (_hdu, nside_result) = {
            let hdu0 = fptr.hdu(0).map_err(|e| Error::Fits(e.to_string()))?;
            let nside0: std::result::Result<i64, _> = hdu0.read_key(&mut fptr, "NSIDE");
            if nside0.is_ok() {
                (hdu0, nside0)
            } else if let Ok(hdu1) = fptr.hdu(1) {
                let nside1: std::result::Result<i64, _> = hdu1.read_key(&mut fptr, "NSIDE");
                if nside1.is_ok() {
                    (hdu1, nside1)
                } else {
                    (hdu0, nside0)
                }
            } else {
                (hdu0, nside0)
            }
        };

        if nside_result.is_err() {
            drop(fptr);
            return Self::from_fits_multiorder(path);
        }

        let nside = nside_result.unwrap();
        if !is_valid_nside(nside) {
            return Err(Error::InvalidNside(nside));
        }

        // Read probability column
        let probs = read_prob_column(&mut fptr)?;
        if probs.is_empty() {
            return Err(Error::EmptyMap);
        }

        // Normalize
        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 || sum.is_nan() {
            return Err(Error::InvalidProbSum(sum));
        }

        // Collect non-zero pixels
        let mut pixels: Vec<(u64, f64)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, &p)| (i as u64, p / sum))
            .collect();
        pixels.sort_by_key(|&(idx, _)| idx);

        let depth = (nside as f64).log2() as u8;
        Ok(Self {
            nside: nside as u32,
            depth,
            pixels,
        })
    }

    /// Load a multi-order MOC FITS skymap.
    fn from_fits_multiorder<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut fptr = FitsFile::open(path).map_err(|e| Error::Fits(e.to_string()))?;

        let hdu = fptr.hdu(1).map_err(|e| Error::Fits(e.to_string()))?;

        let uniq: Vec<i64> = hdu
            .read_col(&mut fptr, "UNIQ")
            .map_err(|e| Error::MissingColumn(format!("UNIQ: {}", e)))?;

        let probdensity: Vec<f64> = hdu
            .read_col(&mut fptr, "PROBDENSITY")
            .or_else(|_| hdu.read_col(&mut fptr, "PROB"))
            .map_err(|e| Error::MissingColumn(format!("PROBDENSITY/PROB: {}", e)))?;

        if uniq.is_empty() || probdensity.is_empty() {
            return Err(Error::EmptyMap);
        }

        // Find max order
        let mut max_order = 0u8;
        for &u in &uniq {
            let order = ((u as f64 / 4.0).log2() / 2.0).floor() as u8;
            if order > max_order {
                max_order = order;
            }
        }

        let nside = 2_i64.pow(max_order as u32);
        let npix = 12 * nside * nside;

        // Build dense map, then sparsify
        let mut dense = vec![0.0f64; npix as usize];
        for (&u, &prob) in uniq.iter().zip(probdensity.iter()) {
            let order = ((u as f64 / 4.0).log2() / 2.0).floor() as u8;
            let ipix = (u - 4 * (4_i64.pow(order as u32))) as u64;

            if order == max_order {
                dense[ipix as usize] = prob;
            } else {
                let ratio = 4usize.pow((max_order - order) as u32);
                let start = (ipix * ratio as u64) as usize;
                let sub_prob = prob / ratio as f64;
                for i in 0..ratio {
                    if start + i < dense.len() {
                        dense[start + i] = sub_prob;
                    }
                }
            }
        }

        // Normalize and sparsify
        let sum: f64 = dense.iter().sum();
        if sum <= 0.0 {
            return Err(Error::InvalidProbSum(sum));
        }

        let mut pixels: Vec<(u64, f64)> = dense
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, &p)| (i as u64, p / sum))
            .collect();
        pixels.sort_by_key(|&(idx, _)| idx);

        let depth = max_order;
        Ok(Self {
            nside: nside as u32,
            depth,
            pixels,
        })
    }

    /// Create a sparse skymap from a dense probability array.
    pub fn from_dense(nside: u32, probs: &[f64]) -> Self {
        let sum: f64 = probs.iter().sum();
        let norm = if sum > 0.0 { sum } else { 1.0 };
        let mut pixels: Vec<(u64, f64)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, &p)| (i as u64, p / norm))
            .collect();
        pixels.sort_by_key(|&(idx, _)| idx);

        let depth = (nside as f64).log2() as u8;
        Self {
            nside,
            depth,
            pixels,
        }
    }

    /// Look up probability at a nested pixel index (binary search).
    pub fn probability_at(&self, nested_idx: u64) -> f64 {
        match self.pixels.binary_search_by_key(&nested_idx, |&(idx, _)| idx) {
            Ok(pos) => self.pixels[pos].1,
            Err(_) => 0.0,
        }
    }

    /// Look up probability at a sky position (RA, Dec in degrees).
    pub fn probability_at_position(&self, ra_deg: f64, dec_deg: f64) -> f64 {
        let lon = ra_deg.to_radians();
        let lat = dec_deg.to_radians();
        let idx = hash(self.depth, lon, lat);
        self.probability_at(idx)
    }

    /// Position (RA, Dec in degrees) of the maximum probability pixel.
    pub fn max_prob_position(&self) -> (f64, f64) {
        let (idx, _) = self
            .pixels
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .copied()
            .unwrap_or((0, 0.0));
        let (lon, lat) = center(self.depth, idx);
        (lon.to_degrees(), lat.to_degrees())
    }

    /// Number of non-zero pixels.
    pub fn nnz(&self) -> usize {
        self.pixels.len()
    }

    /// Total number of pixels at this NSIDE.
    pub fn npix(&self) -> usize {
        12 * (self.nside as usize) * (self.nside as usize)
    }
}

fn is_valid_nside(nside: i64) -> bool {
    nside > 0 && (nside & (nside - 1)) == 0
}

fn read_prob_column(fptr: &mut FitsFile) -> Result<Vec<f64>> {
    let column_names = ["PROB", "PROBABILITY", "PROBDENSITY"];
    for col_name in &column_names {
        if let Ok(hdu) = fptr.hdu(1) {
            if let Ok(data) = hdu.read_col::<f64>(fptr, col_name) {
                return Ok(data);
            }
        }
    }
    Err(Error::MissingColumn(
        "PROB/PROBABILITY/PROBDENSITY".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_from_dense() {
        let nside = 4;
        let npix = 12 * nside * nside;
        let mut probs = vec![0.0; npix as usize];
        probs[0] = 0.5;
        probs[10] = 0.3;
        probs[100] = 0.2;

        let skymap = SparseSkymap::from_dense(nside, &probs);
        assert_eq!(skymap.nnz(), 3);
        assert!((skymap.probability_at(0) - 0.5).abs() < 1e-10);
        assert!((skymap.probability_at(10) - 0.3).abs() < 1e-10);
        assert!((skymap.probability_at(100) - 0.2).abs() < 1e-10);
        assert_eq!(skymap.probability_at(50), 0.0);
    }

    #[test]
    fn test_sparse_roundtrip() {
        let nside = 8u32;
        let npix = 12 * nside * nside;
        let mut probs = vec![0.0; npix as usize];
        // Put all probability in one pixel
        probs[42] = 1.0;

        let skymap = SparseSkymap::from_dense(nside, &probs);
        assert_eq!(skymap.nnz(), 1);
        assert!((skymap.probability_at(42) - 1.0).abs() < 1e-10);
    }
}
