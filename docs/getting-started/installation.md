# Installation

## Python

### From PyPI

```bash
pip install skymap-overlap
```

### From source

Building from source requires a Rust toolchain (1.70+) and
[maturin](https://www.maturin.rs/):

```bash
git clone https://github.com/mcoughlin/skymap-overlap.git
cd skymap-overlap
pip install maturin
maturin develop --release --features python
```

### Dependencies

The Python package requires:

- Python >= 3.9
- NumPy (for trial overlap arrays)

## Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
skymap-overlap = { git = "https://github.com/mcoughlin/skymap-overlap" }
```

### System Requirements

The `fitsio` crate requires the CFITSIO C library. On most systems, the
`fitsio-src` feature (enabled by default) will build it from source
automatically. If you prefer to use a system installation:

```bash
# Ubuntu/Debian
sudo apt-get install libcfitsio-dev

# macOS
brew install cfitsio

# Fedora/RHEL
sudo dnf install cfitsio-devel
```

## Verifying the Installation

### Python

```python
import skymap_overlap
print(skymap_overlap.GBM_RATE_HZ)
```

### Rust

```rust
use skymap_overlap::GBM_RATE_HZ;
println!("GBM rate: {:.2e} Hz", GBM_RATE_HZ);
```
