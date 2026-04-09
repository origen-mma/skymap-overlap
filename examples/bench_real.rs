//! Benchmark with real FITS skymaps.

use skymap_overlap::{empirical_pvalue, far_remapped, overlap_integral, SparseSkymap, GBM_RATE_HZ};
use std::time::Instant;

fn main() {
    let gw_path = std::env::args()
        .nth(1)
        .expect("Usage: bench_real <gw.fits> <grb.fits> [n_trials]");
    let grb_path = std::env::args()
        .nth(2)
        .expect("Usage: bench_real <gw.fits> <grb.fits> [n_trials]");
    let n_trials: usize = std::env::args()
        .nth(3)
        .map(|s| s.parse().unwrap())
        .unwrap_or(1000);

    println!("Loading GW skymap: {}", gw_path);
    let t0 = Instant::now();
    let gw = SparseSkymap::from_fits(&gw_path).unwrap();
    println!(
        "  NSIDE={}, nnz={}/{} ({:.1}%), loaded in {:.1}ms",
        gw.nside,
        gw.nnz(),
        gw.npix(),
        100.0 * gw.nnz() as f64 / gw.npix() as f64,
        t0.elapsed().as_secs_f64() * 1000.0
    );

    println!("Loading GRB skymap: {}", grb_path);
    let t0 = Instant::now();
    let grb = SparseSkymap::from_fits(&grb_path).unwrap();
    println!(
        "  NSIDE={}, nnz={}/{} ({:.1}%), loaded in {:.1}ms",
        grb.nside,
        grb.nnz(),
        grb.npix(),
        100.0 * grb.nnz() as f64 / grb.npix() as f64,
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Single overlap
    let t0 = Instant::now();
    let ov = overlap_integral(&gw, &grb);
    println!(
        "\nOverlap integral: {:.6e} (computed in {:.3}ms)",
        ov,
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Monte Carlo p-value
    println!("\nRunning {} parallel rotation trials...", n_trials);
    let t0 = Instant::now();
    let result = empirical_pvalue(&gw, &grb, n_trials, Some(42));
    let elapsed = t0.elapsed().as_secs_f64();

    println!("\n=== Results ===");
    println!("Total time:       {:.2}s", elapsed);
    println!(
        "Time per trial:   {:.2}ms",
        elapsed * 1000.0 / n_trials as f64
    );
    println!(
        "Throughput:       {:.0} trials/sec",
        n_trials as f64 / elapsed
    );
    println!("Observed overlap: {:.6e}", result.observed_overlap);
    println!("Trials >= obs:    {}", result.n_above);
    println!("p-value:          {:.4e}", result.p_value);

    // Joint FAR
    let far_gw = 1e-7; // Hz
    let dt = 600.0; // seconds
    let far_max = 2.0 / 86400.0; // 2/day

    let far = far_remapped(far_gw, GBM_RATE_HZ, dt, result.p_value, far_max);
    println!(
        "\nJoint FAR (remapped): {:.4e} Hz ({:.2e} /year)",
        far,
        far * 365.25 * 86400.0
    );
}
