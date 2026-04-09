# skymap-overlap

Fast joint GW-GRB False Alarm Rate computation using empirical skymap overlap
p-values from random rotation trials.

[![Tests](https://github.com/origen-mma/skymap-overlap/actions/workflows/tests.yml/badge.svg)](https://github.com/origen-mma/skymap-overlap/actions/workflows/tests.yml)
[![Docs](https://github.com/origen-mma/skymap-overlap/actions/workflows/docs.yml/badge.svg)](https://origen-mma.github.io/skymap-overlap/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

![Overlap method visualization](docs/overlap_method.png)

## Why?

The current RAVEN method divides the temporal FAR by the skymap overlap
integral, which produces biased FAR distributions. Replacing the overlap
integral with an empirical p-value from random rotation trials, combined
with a log-remapping formula, yields a properly uniform joint FAR distribution.

The main blocker for adoption has been computational cost. This crate provides
a high-performance Rust implementation with Python bindings — fast enough
for real-time use in gwcelery / RAVEN.

## Performance

Sparse HEALPix operations scale with the localization area, not the full sky:

| Operation | Time |
|---|---|
| Overlap integral | ~0.05 ms |
| Single rotation trial | ~5.7 ms |
| 100 rotation trials (8 cores) | ~0.6 s |
| 1,000 rotation trials (8 cores) | ~5.7 s |
| 10,000 rotation trials (8 cores) | ~57 s |

## Installation

### Python (recommended)

Build from source with [maturin](https://www.maturin.rs/):

```bash
pip install maturin
git clone https://github.com/origen-mma/skymap-overlap.git
cd skymap-overlap
maturin develop --release --features python
```

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
skymap-overlap = { git = "https://github.com/origen-mma/skymap-overlap" }
```

## Quick Start

### Python

```python
import skymap_overlap

# Load skymaps
gw = skymap_overlap.Skymap.from_fits("gw_skymap.fits")
grb = skymap_overlap.Skymap.from_fits("grb_skymap.fits")
print(f"GW: nside={gw.nside}, nnz={gw.nnz}")

# Compute overlap integral
ov = skymap_overlap.overlap(gw, grb)
print(f"Overlap: {ov:.6e}")

# Compute empirical p-value (parallel across all cores)
result = skymap_overlap.pvalue(gw, grb, n_trials=1000, seed=42)
print(f"p-value: {result.p_value:.4f}")
print(f"Observed overlap: {result.observed_overlap:.6e}")
print(f"Trial overlaps shape: {result.trial_overlaps.shape}")

# Compute joint FAR (corrected remapped method)
far = skymap_overlap.far_remapped(
    far_gw=1e-7,
    grb_rate=skymap_overlap.GBM_RATE_HZ,
    time_window=600.0,
    p_value=result.p_value,
    far_gw_max=2.0 / 86400.0,
)
print(f"Joint FAR: {far:.2e} Hz")
```

### Rust

```rust,no_run
use skymap_overlap::{SparseSkymap, empirical_pvalue, far_remapped, GBM_RATE_HZ};

let gw = SparseSkymap::from_fits("gw_skymap.fits").unwrap();
let grb = SparseSkymap::from_fits("grb_skymap.fits").unwrap();

let result = empirical_pvalue(&gw, &grb, 1000, Some(42));
println!("p-value: {:.4}", result.p_value);

let far = far_remapped(
    1e-7,
    GBM_RATE_HZ,
    600.0,
    result.p_value,
    2.0 / 86400.0,
);
println!("Joint FAR: {:.2e} Hz", far);
```

## API Overview

### Python

| Function / Class | Description |
|---|---|
| `Skymap.from_fits(path)` | Load a HEALPix skymap from FITS |
| `Skymap.from_dense(nside, probs)` | Create from a dense probability array |
| `overlap(gw, grb)` | Compute the overlap integral |
| `pvalue(gw, grb, n_trials, seed)` | Empirical p-value via rotation trials |
| `far_remapped(...)` | Corrected joint FAR with remapping |
| `far_raven(...)` | Original RAVEN FAR (biased) |
| `far_temporal(...)` | Temporal-only FAR (no spatial info) |
| `GBM_RATE_HZ` | Fermi GBM detection rate (~325/yr) |

### Rust

The Rust API mirrors the Python API. See the
[API documentation](https://origen-mma.github.io/skymap-overlap/api/) for
full details.

## Method

The corrected joint FAR:

```
FAR_c = FAR_gw * R_grb * dt * p * [1 - ln(FAR_gw * p / FAR_gw_max)]
```

where `p` is the empirical p-value obtained by rotating the GRB skymap to
`N` random positions and computing the fraction of trials where the overlap
integral exceeds the observed value.

## License

This project is licensed under the GNU General Public License v3.0 - see the
[LICENSE](LICENSE) file for details.
