# API Reference

## Python API

### Classes

#### `skymap_overlap.Skymap`

A sparse HEALPix skymap storing only non-zero pixels.

**Static methods:**

| Method | Description |
|---|---|
| `Skymap.from_fits(path: str) -> Skymap` | Load from a FITS file (flat HEALPix or multi-order MOC) |
| `Skymap.from_dense(nside: int, probs: list[float]) -> Skymap` | Create from a dense probability array of length $12 \times \text{nside}^2$ |

**Properties:**

| Property | Type | Description |
|---|---|---|
| `nside` | `int` | HEALPix NSIDE parameter |
| `depth` | `int` | HEALPix depth ($\log_2$ NSIDE) |
| `nnz` | `int` | Number of non-zero pixels |
| `npix` | `int` | Total pixels at this NSIDE ($12 \times \text{nside}^2$) |

**Methods:**

| Method | Description |
|---|---|
| `probability_at(nested_idx: int) -> float` | Probability at a nested pixel index |
| `probability_at_position(ra_deg: float, dec_deg: float) -> float` | Probability at a sky position |
| `max_prob_position() -> tuple[float, float]` | (RA, Dec) of the peak pixel |

---

#### `skymap_overlap.PvalueResult`

Result of an empirical p-value computation.

**Properties:**

| Property | Type | Description |
|---|---|---|
| `observed_overlap` | `float` | Overlap integral of the unrotated pair |
| `p_value` | `float` | Fraction of trials with overlap >= observed |
| `n_trials` | `int` | Number of rotation trials |
| `n_above` | `int` | Trials with overlap >= observed |
| `trial_overlaps` | `numpy.ndarray` | 1-D float64 array of trial overlap values |

---

### Functions

#### `skymap_overlap.overlap(gw, grb) -> float`

Compute the overlap integral $I_\Omega = \sum_i p_{\text{gw}}(i) \times p_{\text{grb}}(i)$.

Automatically handles resolution mismatch by upsampling the coarser map.

**Parameters:**

- `gw` (`Skymap`): Gravitational wave skymap
- `grb` (`Skymap`): Gamma-ray burst skymap

---

#### `skymap_overlap.pvalue(gw, grb, n_trials, seed=None) -> PvalueResult`

Compute an empirical p-value via random rotation trials. Uses all available
CPU cores.

**Parameters:**

- `gw` (`Skymap`): GW skymap (held fixed)
- `grb` (`Skymap`): GRB skymap (rotated randomly)
- `n_trials` (`int`): Number of random rotation trials
- `seed` (`int`, optional): RNG seed for reproducibility (default: 42)

---

#### `skymap_overlap.far_remapped(far_gw, grb_rate, time_window, p_value, far_gw_max) -> float`

Corrected joint FAR with remapping:

$$
\text{FAR}_c = \text{FAR}_{\text{gw}} \times R_{\text{grb}} \times \Delta t \times p \times \left[1 - \ln\left(\frac{\text{FAR}_{\text{gw}} \times p}{\text{FAR}_{\text{gw,max}}}\right)\right]
$$

**Parameters:**

- `far_gw` (`float`): GW FAR in Hz
- `grb_rate` (`float`): GRB detection rate in Hz
- `time_window` (`float`): Coincidence window in seconds
- `p_value` (`float`): Empirical spatial p-value
- `far_gw_max` (`float`): Pipeline FAR threshold in Hz

**Returns:** Joint FAR in Hz

---

#### `skymap_overlap.far_raven(far_gw, grb_rate, time_window, overlap) -> float`

Original RAVEN joint FAR (Eq 1, biased):

$$
\text{FAR}_c = \frac{\text{FAR}_{\text{gw}} \times R_{\text{grb}} \times \Delta t}{I_\Omega}
$$

!!! warning
    This method produces biased FAR distributions. Use `far_remapped` for
    production analysis.

---

#### `skymap_overlap.far_temporal(far_gw, grb_rate, time_window) -> float`

Temporal-only joint FAR (no spatial information):

$$
\text{FAR}_t = \text{FAR}_{\text{gw}} \times R_{\text{grb}} \times \Delta t
$$

---

### Constants

| Constant | Value | Description |
|---|---|---|
| `GBM_RATE_HZ` | ~1.03e-5 | Fermi GBM rate in Hz (~325/year) |
| `GBM_RATE_PER_YEAR` | 325.0 | Fermi GBM rate per year |

---

## Rust API

The Rust API is documented via `cargo doc`. The main public items are:

| Item | Description |
|---|---|
| `SparseSkymap` | Sparse HEALPix skymap struct |
| `SparseSkymap::from_fits(path)` | Load from FITS |
| `SparseSkymap::from_dense(nside, probs)` | Create from dense array |
| `overlap_integral(a, b) -> f64` | Overlap integral |
| `empirical_pvalue(gw, grb, n, seed) -> PvalueResult` | Empirical p-value |
| `far_remapped(...)` | Corrected joint FAR |
| `far_raven(...)` | Original RAVEN FAR |
| `far_temporal(...)` | Temporal-only FAR |
| `GBM_RATE_HZ` | GBM rate constant (Hz) |

Generate the full Rust docs with:

```bash
cargo doc --open
```
