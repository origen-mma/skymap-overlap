//! Overlap integral and empirical p-value via random skymap rotations.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::rotation::rotate_skymap;
use crate::skymap::SparseSkymap;

/// Result of an empirical p-value computation.
#[derive(Debug, Clone)]
pub struct PvalueResult {
    /// Overlap integral of the observed (unrotated) skymap pair.
    pub observed_overlap: f64,
    /// Empirical p-value: fraction of trials with overlap >= observed.
    pub p_value: f64,
    /// Number of rotation trials performed.
    pub n_trials: usize,
    /// Number of trials with overlap >= observed.
    pub n_above: usize,
    /// Raw overlap values from each trial (for diagnostics).
    pub trial_overlaps: Vec<f64>,
}

/// Overlap integral I_Ω = Σ_i p_gw(i) × p_grb(i).
///
/// For two sparse skymaps at the same NSIDE, this performs a merge-join
/// on sorted pixel indices — O(nnz_a + nnz_b) instead of O(npix).
///
/// For different NSIDEs, the coarser map is resampled up by distributing
/// each pixel's probability equally among its sub-pixels.
pub fn overlap_integral(a: &SparseSkymap, b: &SparseSkymap) -> f64 {
    if a.nside == b.nside {
        merge_join_overlap(&a.pixels, &b.pixels)
    } else if a.nside < b.nside {
        // Upsample a to b's resolution
        let a_up = upsample_pixels(&a.pixels, a.depth, b.depth);
        merge_join_overlap(&a_up, &b.pixels)
    } else {
        let b_up = upsample_pixels(&b.pixels, b.depth, a.depth);
        merge_join_overlap(&a.pixels, &b_up)
    }
}

/// Merge-join two sorted sparse pixel lists and compute the dot product.
fn merge_join_overlap(a: &[(u64, f64)], b: &[(u64, f64)]) -> f64 {
    let mut overlap = 0.0;
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        if a[i].0 == b[j].0 {
            overlap += a[i].1 * b[j].1;
            i += 1;
            j += 1;
        } else if a[i].0 < b[j].0 {
            i += 1;
        } else {
            j += 1;
        }
    }
    overlap
}

/// Upsample sparse pixels from `src_depth` to `tgt_depth` (tgt > src).
/// Each pixel at depth d maps to 4^(tgt-src) sub-pixels at depth tgt,
/// each receiving prob / 4^(tgt-src).
fn upsample_pixels(pixels: &[(u64, f64)], src_depth: u8, tgt_depth: u8) -> Vec<(u64, f64)> {
    let depth_diff = tgt_depth - src_depth;
    let ratio = 4u64.pow(depth_diff as u32);
    let sub_factor = 1.0 / ratio as f64;

    let mut result = Vec::with_capacity(pixels.len() * ratio as usize);
    for &(idx, prob) in pixels {
        let base = idx * ratio;
        let sub_prob = prob * sub_factor;
        for k in 0..ratio {
            result.push((base + k, sub_prob));
        }
    }
    result.sort_by_key(|&(idx, _)| idx);
    result
}

/// Generate a uniform random point on the sphere.
/// Returns (RA, Dec) in degrees.
pub fn random_sky_position(rng: &mut impl Rng) -> (f64, f64) {
    let ra = rng.gen::<f64>() * 360.0;
    // Uniform in cos(dec) → uniform on sphere
    let dec = (rng.gen::<f64>() * 2.0 - 1.0).asin() * 180.0 / PI;
    (ra, dec)
}

/// Compute an empirical p-value for the spatial overlap of two skymaps.
///
/// Rotates `grb` to `n_trials` random sky positions and computes the
/// overlap integral with `gw` each time. The p-value is the fraction
/// of trials where the overlap >= the observed overlap.
///
/// Uses rayon for parallel execution across all available cores.
///
/// # Arguments
/// * `gw` - Gravitational wave skymap (held fixed).
/// * `grb` - Gamma-ray burst skymap (rotated randomly).
/// * `n_trials` - Number of random rotation trials.
/// * `seed` - Optional RNG seed for reproducibility.
pub fn empirical_pvalue(
    gw: &SparseSkymap,
    grb: &SparseSkymap,
    n_trials: usize,
    seed: Option<u64>,
) -> PvalueResult {
    let (grb_ra, grb_dec) = grb.max_prob_position();
    let observed_overlap = overlap_integral(gw, grb);

    let base_seed = seed.unwrap_or(42);

    let trial_overlaps: Vec<f64> = (0..n_trials)
        .into_par_iter()
        .map(|i| {
            let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(i as u64));
            let (rand_ra, rand_dec) = random_sky_position(&mut rng);
            let rotated = rotate_skymap(grb, grb_ra, grb_dec, rand_ra, rand_dec);
            overlap_integral(gw, &rotated)
        })
        .collect();

    let n_above = trial_overlaps
        .iter()
        .filter(|&&v| v >= observed_overlap)
        .count();

    let p_value = n_above as f64 / n_trials as f64;

    PvalueResult {
        observed_overlap,
        p_value,
        n_trials,
        n_above,
        trial_overlaps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_blob_skymap(nside: u32, center_idx: u64, n_pixels: u64) -> SparseSkymap {
        let npix = 12 * nside as u64 * nside as u64;
        let prob = 1.0 / n_pixels as f64;
        let mut pixels = Vec::new();
        for i in 0..n_pixels {
            let idx = (center_idx + i) % npix;
            pixels.push((idx, prob));
        }
        pixels.sort_by_key(|&(idx, _)| idx);
        SparseSkymap {
            nside,
            depth: (nside as f64).log2() as u8,
            pixels,
        }
    }

    #[test]
    fn test_self_overlap() {
        let sky = make_blob_skymap(64, 1000, 50);
        let ov = overlap_integral(&sky, &sky);
        // Self-overlap = sum of p_i^2 = n * (1/n)^2 = 1/n
        let expected = 1.0 / 50.0;
        assert!((ov - expected).abs() < 1e-10);
    }

    #[test]
    fn test_disjoint_overlap() {
        let a = make_blob_skymap(64, 0, 50);
        let b = make_blob_skymap(64, 10000, 50);
        let ov = overlap_integral(&a, &b);
        assert!(ov < 1e-15);
    }

    #[test]
    fn test_pvalue_identical_skymaps() {
        // Identical skymaps should have p-value close to 0 (observed overlap
        // is very high — only a coincident rotation would match it).
        let sky = make_blob_skymap(32, 500, 20);
        let result = empirical_pvalue(&sky, &sky, 100, Some(12345));
        assert!(result.p_value < 0.3);
    }
}
