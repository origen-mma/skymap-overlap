# skymap-overlap

Fast joint GW-GRB False Alarm Rate computation using empirical skymap overlap
p-values, implementing the method of
[Piotrzkowski (2023)](https://doi.org/10.3847/1538-4357/acd3f2).

## Overview

When a gravitational wave (GW) candidate is detected in coincidence with a
gamma-ray burst (GRB), the significance of the spatial coincidence is
quantified by computing the overlap between their sky localization maps.

The current RAVEN method divides the temporal FAR by the overlap integral
$I_\Omega$, but this produces biased FAR distributions.
**skymap-overlap** implements the corrected method from Piotrzkowski (2023):

1. Rotate the GRB skymap to $N$ random sky positions
2. Compute the overlap integral at each position
3. The p-value is the fraction of trials where the overlap exceeds the
   observed value
4. Combine with the temporal FAR using a log-remapping formula

## Key Features

- **Sparse HEALPix representation**: Operations scale with localization area,
  not the full sky
- **Parallel Monte Carlo**: Rotation trials run across all CPU cores via rayon
- **FITS support**: Loads flat HEALPix and multi-order MOC formats (LIGO,
  Fermi GBM, Swift)
- **Python bindings**: PyO3-based bindings for use in gwcelery/RAVEN pipelines
- **Dual API**: Use from Rust or Python with the same interface

## Performance

| Operation | Time |
|---|---|
| Overlap integral | ~0.05 ms |
| Single rotation trial (nside=256) | ~5.7 ms |
| 1000 rotation trials (8 cores) | ~0.8 s |

## Getting Started

See the [Installation](getting-started/installation.md) and
[Quick Start](getting-started/quickstart.md) guides.
