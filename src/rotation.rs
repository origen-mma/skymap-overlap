//! Spherical rotation for sparse HEALPix skymaps via Rodrigues' formula.

use cdshealpix::nested::{center, hash};
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::skymap::SparseSkymap;

/// Compute a 3×3 rotation matrix that maps `(src_ra, src_dec)` to
/// `(tgt_ra, tgt_dec)` using Rodrigues' rotation formula.
pub fn rotation_matrix(
    src_ra: f64,
    src_dec: f64,
    tgt_ra: f64,
    tgt_dec: f64,
) -> [[f64; 3]; 3] {
    let src = spherical_to_cartesian(src_ra, src_dec);
    let tgt = spherical_to_cartesian(tgt_ra, tgt_dec);

    // Rotation axis = cross product
    let axis = [
        src[1] * tgt[2] - src[2] * tgt[1],
        src[2] * tgt[0] - src[0] * tgt[2],
        src[0] * tgt[1] - src[1] * tgt[0],
    ];

    let axis_len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();

    if axis_len < 1e-10 {
        // Source and target coincide (or are antipodal); return identity.
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    let n = [axis[0] / axis_len, axis[1] / axis_len, axis[2] / axis_len];

    // Rotation angle = dot product
    let cos_a = (src[0] * tgt[0] + src[1] * tgt[1] + src[2] * tgt[2]).clamp(-1.0, 1.0);
    let angle = cos_a.acos();
    let sin_a = angle.sin();
    let omc = 1.0 - cos_a;

    [
        [
            cos_a + n[0] * n[0] * omc,
            n[0] * n[1] * omc - n[2] * sin_a,
            n[0] * n[2] * omc + n[1] * sin_a,
        ],
        [
            n[1] * n[0] * omc + n[2] * sin_a,
            cos_a + n[1] * n[1] * omc,
            n[1] * n[2] * omc - n[0] * sin_a,
        ],
        [
            n[2] * n[0] * omc - n[1] * sin_a,
            n[2] * n[1] * omc + n[0] * sin_a,
            cos_a + n[2] * n[2] * omc,
        ],
    ]
}

/// Apply a 3×3 rotation matrix to a vector.
#[inline]
pub fn apply_rotation(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Convert (RA, Dec) in degrees to a Cartesian unit vector.
#[inline]
pub fn spherical_to_cartesian(ra: f64, dec: f64) -> [f64; 3] {
    let ra_r = ra * PI / 180.0;
    let dec_r = dec * PI / 180.0;
    [dec_r.cos() * ra_r.cos(), dec_r.cos() * ra_r.sin(), dec_r.sin()]
}

/// Convert a Cartesian unit vector to (RA, Dec) in degrees.
#[inline]
pub fn cartesian_to_spherical(v: &[f64; 3]) -> (f64, f64) {
    let dec = v[2].asin() * 180.0 / PI;
    let mut ra = v[1].atan2(v[0]) * 180.0 / PI;
    if ra < 0.0 {
        ra += 360.0;
    }
    (ra, dec)
}

/// Rotate a sparse skymap from `(src_ra, src_dec)` to `(tgt_ra, tgt_dec)`.
///
/// Only iterates over non-zero pixels, making this much faster than a
/// dense rotation. Probabilities are re-normalized after rotation.
pub fn rotate_skymap(
    skymap: &SparseSkymap,
    src_ra: f64,
    src_dec: f64,
    tgt_ra: f64,
    tgt_dec: f64,
) -> SparseSkymap {
    let rot = rotation_matrix(src_ra, src_dec, tgt_ra, tgt_dec);
    let depth = skymap.depth;

    // Accumulate rotated probabilities (multiple source pixels may land
    // on the same target pixel).
    let mut acc: HashMap<u64, f64> = HashMap::with_capacity(skymap.pixels.len());

    for &(idx, prob) in &skymap.pixels {
        let (lon, lat) = center(depth, idx);
        let src_vec = spherical_to_cartesian(lon.to_degrees(), lat.to_degrees());
        let rot_vec = apply_rotation(&rot, &src_vec);
        let (rot_ra, rot_dec) = cartesian_to_spherical(&rot_vec);
        let tgt_idx = hash(depth, rot_ra.to_radians(), rot_dec.to_radians());
        *acc.entry(tgt_idx).or_insert(0.0) += prob;
    }

    // Normalize
    let sum: f64 = acc.values().sum();
    let norm = if sum > 0.0 { sum } else { 1.0 };

    let mut pixels: Vec<(u64, f64)> = acc
        .into_iter()
        .map(|(idx, p)| (idx, p / norm))
        .collect();
    pixels.sort_by_key(|&(idx, _)| idx);

    SparseSkymap {
        nside: skymap.nside,
        depth: skymap.depth,
        pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spherical_roundtrip() {
        let ra = 123.456;
        let dec = -34.567;
        let v = spherical_to_cartesian(ra, dec);
        let (ra2, dec2) = cartesian_to_spherical(&v);
        assert!((ra - ra2).abs() < 1e-10);
        assert!((dec - dec2).abs() < 1e-10);
    }

    #[test]
    fn test_identity_rotation() {
        let m = rotation_matrix(100.0, 30.0, 100.0, 30.0);
        let v = spherical_to_cartesian(200.0, -10.0);
        let r = apply_rotation(&m, &v);
        assert!((v[0] - r[0]).abs() < 1e-10);
        assert!((v[1] - r[1]).abs() < 1e-10);
        assert!((v[2] - r[2]).abs() < 1e-10);
    }

    #[test]
    fn test_rotate_skymap_preserves_probability() {
        let nside = 32u32;
        let npix = 12 * nside * nside;
        let mut probs = vec![0.0; npix as usize];
        // Create a small blob of probability
        for i in 100..120 {
            probs[i] = 1.0;
        }
        let skymap = SparseSkymap::from_dense(nside, &probs);

        let rotated = rotate_skymap(&skymap, 0.0, 0.0, 90.0, 45.0);

        let sum: f64 = rotated.pixels.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
