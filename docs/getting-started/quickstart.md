# Quick Start

## Python

### Loading Skymaps

```python
import skymap_overlap

# Load from FITS files (flat HEALPix or multi-order MOC)
gw = skymap_overlap.Skymap.from_fits("gw_skymap.fits")
grb = skymap_overlap.Skymap.from_fits("grb_skymap.fits")

print(f"GW skymap:  nside={gw.nside}, nnz={gw.nnz}, npix={gw.npix}")
print(f"GRB skymap: nside={grb.nside}, nnz={grb.nnz}, npix={grb.npix}")
```

### Computing the Overlap Integral

```python
ov = skymap_overlap.overlap(gw, grb)
print(f"Overlap integral: {ov:.6e}")
```

### Computing the Empirical P-value

```python
result = skymap_overlap.pvalue(gw, grb, n_trials=1000, seed=42)
print(f"p-value:          {result.p_value:.4f}")
print(f"Observed overlap: {result.observed_overlap:.6e}")
print(f"Trials above:     {result.n_above} / {result.n_trials}")

# Trial overlaps are returned as a numpy array
import matplotlib.pyplot as plt
plt.hist(result.trial_overlaps, bins=50)
plt.axvline(result.observed_overlap, color='r', label='observed')
plt.xlabel('Overlap integral')
plt.ylabel('Count')
plt.legend()
plt.show()
```

### Computing the Joint FAR

```python
# Corrected method (remapped FAR)
far = skymap_overlap.far_remapped(
    far_gw=1e-7,                          # GW FAR in Hz
    grb_rate=skymap_overlap.GBM_RATE_HZ,  # GRB rate in Hz
    time_window=600.0,                     # coincidence window (s)
    p_value=result.p_value,
    far_gw_max=2.0 / 86400.0,             # pipeline FAR threshold
)
print(f"Joint FAR (corrected): {far:.2e} Hz")

# Original RAVEN method (for comparison)
far_old = skymap_overlap.far_raven(
    far_gw=1e-7,
    grb_rate=skymap_overlap.GBM_RATE_HZ,
    time_window=600.0,
    overlap=ov,
)
print(f"Joint FAR (RAVEN):     {far_old:.2e} Hz")
```

## Rust

```rust
use skymap_overlap::{
    SparseSkymap, overlap_integral, empirical_pvalue,
    far_remapped, GBM_RATE_HZ,
};

fn main() -> Result<(), skymap_overlap::Error> {
    let gw = SparseSkymap::from_fits("gw_skymap.fits")?;
    let grb = SparseSkymap::from_fits("grb_skymap.fits")?;

    // Overlap integral
    let ov = overlap_integral(&gw, &grb);
    println!("Overlap: {:.6e}", ov);

    // Empirical p-value (1000 trials, reproducible)
    let result = empirical_pvalue(&gw, &grb, 1000, Some(42));
    println!("p-value: {:.4}", result.p_value);

    // Joint FAR
    let far = far_remapped(
        1e-7,
        GBM_RATE_HZ,
        600.0,
        result.p_value,
        2.0 / 86400.0,
    );
    println!("Joint FAR: {:.2e} Hz", far);

    Ok(())
}
```
