//! Error types for skymap-overlap.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("FITS I/O: {0}")]
    Fits(String),

    #[error("Invalid NSIDE {0} (must be a power of 2)")]
    InvalidNside(i64),

    #[error("Missing FITS column: {0}")]
    MissingColumn(String),

    #[error("Empty probability map")]
    EmptyMap,

    #[error("Probability sum invalid: {0}")]
    InvalidProbSum(f64),

    #[error("NSIDE mismatch: {0} vs {1}")]
    NsideMismatch(u32, u32),

    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
