# skymap-overlap

Fast joint GW-GRB False Alarm Rate computation using empirical skymap overlap
p-values, implementing the method of
[Piotrzkowski (2023)](https://doi.org/10.3847/1538-4357/acd3f2).

[![Tests](https://github.com/mcoughlin/skymap-overlap/actions/workflows/tests.yml/badge.svg)](https://github.com/mcoughlin/skymap-overlap/actions/workflows/tests.yml)
[![Docs](https://github.com/mcoughlin/skymap-overlap/actions/workflows/docs.yml/badge.svg)](https://mcoughlin.github.io/skymap-overlap/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Why?

The current RAVEN method divides the temporal FAR by the skymap overlap
integral, which produces biased FAR distributions. Piotrzkowski (2023) showed
that replacing the overlap integral with an empirical p-value from random
rotation trials, combined with a log-remapping formula, yields a properly
uniform joint FAR distribution.

This crate provides a high-performance Rust implementation with Python bindings,
suitable for integration into gwcelery / RAVEN pipelines.

## Performance

Sparse HEALPix operations scale with the localization area, not the full sky:

| Operation | Time |
|---|---|
| Overlap integral | ~0.05 ms |
| Single rotation trial (nside=256) | ~5.7 ms |
| 1000 rotation trials (8 cores) | ~0.8 s |

## Installation

### Python (recommended)

```bash
pip install skymap-overlap
```

Or build from source with [maturin](https://www.maturin.rs/):

```bash
pip install maturin
maturin develop --release --features python
```

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
skymap-overlap = { git = "https://github.com/mcoughlin/skymap-overlap" }
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

# Compute joint FAR (Piotrzkowski 2023, Eq 3)
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

```rust
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
| `far_remapped(...)` | Corrected joint FAR (Eq 3) |
| `far_raven(...)` | Original RAVEN FAR (Eq 1, biased) |
| `far_temporal(...)` | Temporal-only FAR (no spatial info) |
| `GBM_RATE_HZ` | Fermi GBM detection rate (~325/yr) |

### Rust

The Rust API mirrors the Python API. See the
[API documentation](https://mcoughlin.github.io/skymap-overlap/api/) for
full details.

## Method

The corrected joint FAR (Eq 3 of Piotrzkowski 2023):

```
FAR_c = FAR_gw * R_grb * dt * p * [1 - ln(FAR_gw * p / FAR_gw_max)]
```

where `p` is the empirical p-value obtained by rotating the GRB skymap to
`N` random positions and computing the fraction of trials where the overlap
integral exceeds the observed value (Eq 4).

## License

This project is licensed under the GNU General Public License v3.0 - see the
[LICENSE](LICENSE) file for details.

## Citation

If you use this software, please cite:

```bibtex
@article{Piotrzkowski2023,
  author  = {Piotrzkowski, Brandon},
  title   = {A Revised Method for Joint GW-GRB Detection},
  journal = {The Astrophysical Journal},
  year    = {2023},
  doi     = {10.3847/1538-4357/acd3f2}
}
```
