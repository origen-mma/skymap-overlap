//! Integration tests for skymap-overlap using synthetic skymaps.

use skymap_overlap::{
    empirical_pvalue, far_remapped, far_raven, far_temporal, overlap_integral, GBM_RATE_HZ,
    SparseSkymap,
};

/// Create a uniform blob skymap with `n_pixels` contiguous pixels starting at `center_idx`.
fn make_blob(nside: u32, center_idx: u64, n_pixels: u64) -> SparseSkymap {
    let npix = 12 * nside as u64 * nside as u64;
    let prob = 1.0 / n_pixels as f64;
    let mut pixels: Vec<(u64, f64)> = (0..n_pixels)
        .map(|i| ((center_idx + i) % npix, prob))
        .collect();
    pixels.sort_by_key(|&(idx, _)| idx);
    SparseSkymap {
        nside,
        depth: (nside as f64).log2() as u8,
        pixels,
    }
}

/// Create a skymap from a dense probability array.
fn make_dense(nside: u32, nonzero: &[(usize, f64)]) -> SparseSkymap {
    let npix = 12 * (nside as usize) * (nside as usize);
    let mut probs = vec![0.0; npix];
    for &(idx, p) in nonzero {
        probs[idx] = p;
    }
    SparseSkymap::from_dense(nside, &probs)
}

// --- Overlap integral tests ---

#[test]
fn overlap_self_is_sum_of_squares() {
    let sky = make_blob(64, 1000, 100);
    let ov = overlap_integral(&sky, &sky);
    // Each pixel has prob 1/100, so self-overlap = 100 * (1/100)^2 = 0.01
    assert!((ov - 0.01).abs() < 1e-12);
}

#[test]
fn overlap_disjoint_is_zero() {
    let a = make_blob(64, 0, 50);
    let b = make_blob(64, 20000, 50);
    let ov = overlap_integral(&a, &b);
    assert!(ov.abs() < 1e-15);
}

#[test]
fn overlap_is_symmetric() {
    let a = make_blob(32, 0, 30);
    let b = make_blob(32, 20, 30);
    let ov_ab = overlap_integral(&a, &b);
    let ov_ba = overlap_integral(&b, &a);
    assert!((ov_ab - ov_ba).abs() < 1e-15);
}

#[test]
fn overlap_partial_exact() {
    // 100 pixels each, 50 overlap
    let a = make_blob(32, 0, 100);
    let b = make_blob(32, 50, 100);
    let ov = overlap_integral(&a, &b);
    // 50 shared pixels, each with prob 1/100
    let expected = 50.0 * (1.0 / 100.0) * (1.0 / 100.0);
    assert!((ov - expected).abs() < 1e-12);
}

#[test]
fn overlap_single_pixel_maps() {
    let a = make_dense(8, &[(42, 1.0)]);
    let b = make_dense(8, &[(42, 1.0)]);
    let ov = overlap_integral(&a, &b);
    assert!((ov - 1.0).abs() < 1e-12);
}

#[test]
fn overlap_single_pixel_disjoint() {
    let a = make_dense(8, &[(0, 1.0)]);
    let b = make_dense(8, &[(1, 1.0)]);
    let ov = overlap_integral(&a, &b);
    assert!(ov.abs() < 1e-15);
}

// --- P-value tests ---

#[test]
fn pvalue_returns_valid_range() {
    let gw = make_blob(32, 500, 20);
    let grb = make_blob(32, 500, 20);
    let result = empirical_pvalue(&gw, &grb, 200, Some(42));

    assert!(result.p_value >= 0.0);
    assert!(result.p_value <= 1.0);
    assert_eq!(result.n_trials, 200);
    assert!(result.n_above <= result.n_trials);
    assert_eq!(result.trial_overlaps.len(), 200);
}

#[test]
fn pvalue_observed_overlap_matches_direct() {
    let gw = make_blob(32, 100, 30);
    let grb = make_blob(32, 110, 30);

    let direct = overlap_integral(&gw, &grb);
    let result = empirical_pvalue(&gw, &grb, 50, Some(1));

    assert!((result.observed_overlap - direct).abs() < 1e-15);
}

#[test]
fn pvalue_reproducible_with_seed() {
    let gw = make_blob(16, 0, 20);
    let grb = make_blob(16, 10, 20);

    let r1 = empirical_pvalue(&gw, &grb, 100, Some(999));
    let r2 = empirical_pvalue(&gw, &grb, 100, Some(999));

    assert!((r1.p_value - r2.p_value).abs() < 1e-15);
    for (a, b) in r1.trial_overlaps.iter().zip(r2.trial_overlaps.iter()) {
        assert!((a - b).abs() < 1e-15);
    }
}

#[test]
fn pvalue_different_seeds_differ() {
    let gw = make_blob(32, 0, 30);
    let grb = make_blob(32, 15, 30);

    let r1 = empirical_pvalue(&gw, &grb, 100, Some(1));
    let r2 = empirical_pvalue(&gw, &grb, 100, Some(2));

    // Trial overlaps should differ (extremely unlikely to match by chance)
    let same = r1
        .trial_overlaps
        .iter()
        .zip(r2.trial_overlaps.iter())
        .all(|(a, b)| (a - b).abs() < 1e-15);
    assert!(!same, "Different seeds should produce different trials");
}

#[test]
fn pvalue_disjoint_high_pvalue() {
    // With well-separated skymaps, most random rotations will also give
    // zero overlap, so p-value should be high.
    let gw = make_blob(16, 0, 5);
    let grb = make_blob(16, 1000, 5);
    let result = empirical_pvalue(&gw, &grb, 200, Some(42));
    // The observed overlap is 0, and many rotated positions will also give 0
    assert!(result.p_value >= 0.5);
}

// --- FAR tests ---

#[test]
fn far_raven_positive() {
    let far = far_raven(1e-7, GBM_RATE_HZ, 600.0, 0.01);
    assert!(far > 0.0);
    assert!(far.is_finite());
}

#[test]
fn far_raven_zero_overlap_is_infinity() {
    let far = far_raven(1e-7, GBM_RATE_HZ, 600.0, 0.0);
    assert!(far.is_infinite());
}

#[test]
fn far_raven_higher_overlap_lower_far() {
    let far_low_ov = far_raven(1e-7, GBM_RATE_HZ, 600.0, 0.001);
    let far_high_ov = far_raven(1e-7, GBM_RATE_HZ, 600.0, 0.1);
    assert!(far_high_ov < far_low_ov);
}

#[test]
fn far_remapped_positive() {
    let far = far_remapped(1e-7, GBM_RATE_HZ, 600.0, 0.05, 2.0 / 86400.0);
    assert!(far > 0.0);
    assert!(far.is_finite());
}

#[test]
fn far_remapped_lower_p_lower_far() {
    let far_high_p = far_remapped(1e-7, GBM_RATE_HZ, 600.0, 0.1, 2.0 / 86400.0);
    let far_low_p = far_remapped(1e-7, GBM_RATE_HZ, 600.0, 0.001, 2.0 / 86400.0);
    assert!(far_low_p < far_high_p);
}

#[test]
fn far_remapped_zero_pvalue_is_zero() {
    let far = far_remapped(1e-7, GBM_RATE_HZ, 600.0, 0.0, 2.0 / 86400.0);
    assert!((far - 0.0).abs() < 1e-30);
}

#[test]
fn far_temporal_exact() {
    let expected = 1e-7 * GBM_RATE_HZ * 600.0;
    let far = far_temporal(1e-7, GBM_RATE_HZ, 600.0);
    assert!((far - expected).abs() < 1e-25);
}

#[test]
fn far_remapped_exceeds_temporal_at_p1() {
    let ft = far_temporal(1e-7, GBM_RATE_HZ, 600.0);
    let fc = far_remapped(1e-7, GBM_RATE_HZ, 600.0, 1.0, 2.0 / 86400.0);
    assert!(fc >= ft);
}

// --- SparseSkymap construction tests ---

#[test]
fn from_dense_normalization() {
    let sky = make_dense(4, &[(0, 2.0), (1, 3.0)]);
    assert!((sky.probability_at(0) - 0.4).abs() < 1e-10);
    assert!((sky.probability_at(1) - 0.6).abs() < 1e-10);
}

// --- FITS fixture tests ---

#[test]
fn load_moc_gw_skymap() {
    let gw = SparseSkymap::from_fits("tests/fixtures/gw_skymap.fits").unwrap();
    assert_eq!(gw.nside, 64);
    assert!(gw.nnz() > 0);
}

#[test]
fn load_flat_grb_skymap() {
    let grb = SparseSkymap::from_fits("tests/fixtures/grb_skymap.fits").unwrap();
    assert_eq!(grb.nside, 32);
    assert!(grb.nnz() > 0);
}

#[test]
fn overlap_real_fits_skymaps() {
    let gw = SparseSkymap::from_fits("tests/fixtures/gw_skymap.fits").unwrap();
    let grb = SparseSkymap::from_fits("tests/fixtures/grb_skymap.fits").unwrap();
    let ov = overlap_integral(&gw, &grb);
    assert!(ov > 0.0);
    assert!(ov < 1.0);
}

#[test]
fn pvalue_real_fits_skymaps() {
    let gw = SparseSkymap::from_fits("tests/fixtures/gw_skymap.fits").unwrap();
    let grb = SparseSkymap::from_fits("tests/fixtures/grb_skymap.fits").unwrap();
    let result = empirical_pvalue(&gw, &grb, 50, Some(42));
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    assert_eq!(result.trial_overlaps.len(), 50);
}

#[test]
fn from_dense_zero_pixels_excluded() {
    let sky = make_dense(4, &[(0, 1.0), (100, 0.0)]);
    // pixel 100 had 0.0, so it should not appear
    assert_eq!(sky.nnz(), 1);
}

#[test]
fn npix_correct() {
    let sky = make_blob(64, 0, 10);
    assert_eq!(sky.npix(), 12 * 64 * 64);
}

#[test]
fn probability_at_missing_pixel_is_zero() {
    let sky = make_dense(8, &[(42, 1.0)]);
    assert_eq!(sky.probability_at(0), 0.0);
    assert_eq!(sky.probability_at(100), 0.0);
}

// --- End-to-end integration test ---

#[test]
fn end_to_end_pvalue_to_far() {
    let gw = make_blob(32, 200, 40);
    let grb = make_blob(32, 210, 40);

    let result = empirical_pvalue(&gw, &grb, 500, Some(42));
    assert!(result.observed_overlap > 0.0);

    let far = far_remapped(1e-7, GBM_RATE_HZ, 600.0, result.p_value, 2.0 / 86400.0);

    // FAR should be positive and finite when p > 0
    if result.p_value > 0.0 {
        assert!(far > 0.0);
        assert!(far.is_finite());
    }
}

#[test]
fn end_to_end_comparison_raven_vs_piotrzkowski() {
    let gw = make_blob(32, 300, 50);
    let grb = make_blob(32, 310, 50);

    let ov = overlap_integral(&gw, &grb);
    let result = empirical_pvalue(&gw, &grb, 500, Some(42));

    let far_old = far_raven(1e-7, GBM_RATE_HZ, 600.0, ov);
    let far_new = far_remapped(1e-7, GBM_RATE_HZ, 600.0, result.p_value, 2.0 / 86400.0);

    // Both should be positive and finite
    assert!(far_old > 0.0 && far_old.is_finite());
    if result.p_value > 0.0 {
        assert!(far_new > 0.0 && far_new.is_finite());
    }
}
