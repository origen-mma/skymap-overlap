//! Joint GW-GRB False Alarm Rate calculations.
//!
//! Implements both the original RAVEN method (Urban et al. 2016) and the
//! corrected p-value method (Piotrzkowski 2023).

/// Typical Fermi GBM gamma-ray burst detection rate (~325/year).
pub const GBM_RATE_PER_YEAR: f64 = 325.0;
/// GBM rate in Hz.
pub const GBM_RATE_HZ: f64 = GBM_RATE_PER_YEAR / (365.25 * 24.0 * 3600.0);

/// Original RAVEN joint FAR (Eq 1 of Piotrzkowski 2023).
///
/// FAR_c = FAR_gw × R_grb × Δt / I_Ω
///
/// This formula is known to be biased: it under-represents low-significance
/// candidates because dividing by I_Ω (which follows an unknown distribution)
/// does not yield a uniformly distributed FAR.
///
/// # Arguments
/// * `far_gw_hz` - GW false alarm rate in Hz.
/// * `grb_rate_hz` - GRB detection rate in Hz.
/// * `time_window_s` - Coincidence time window in seconds.
/// * `overlap_integral` - Skymap overlap integral I_Ω.
pub fn far_raven(
    far_gw_hz: f64,
    grb_rate_hz: f64,
    time_window_s: f64,
    overlap_integral: f64,
) -> f64 {
    if overlap_integral <= 0.0 {
        return f64::INFINITY;
    }
    far_gw_hz * grb_rate_hz * time_window_s / overlap_integral
}

/// Corrected joint FAR using empirical p-values (Eq 3 of Piotrzkowski 2023).
///
/// FAR_c = FAR_gw × R_grb × Δt × p × [1 − ln(FAR_gw × p / FAR_gw_max)]
///
/// The `1 − ln(AB / (A_max × B_max))` remapping ensures the joint FAR
/// follows a uniform distribution when both inputs (temporal FAR and
/// spatial p-value) are uniform, correcting the bias in [`far_raven`].
///
/// # Arguments
/// * `far_gw_hz` - GW false alarm rate in Hz.
/// * `grb_rate_hz` - GRB detection rate in Hz.
/// * `time_window_s` - Coincidence time window in seconds.
/// * `p_value` - Empirical spatial p-value from rotation trials.
/// * `far_gw_max_hz` - Maximum GW FAR threshold of the pipeline in Hz
///   (e.g., 2/day ≈ 2.3e-5 Hz).
pub fn far_remapped(
    far_gw_hz: f64,
    grb_rate_hz: f64,
    time_window_s: f64,
    p_value: f64,
    far_gw_max_hz: f64,
) -> f64 {
    if p_value <= 0.0 || far_gw_max_hz <= 0.0 {
        return 0.0;
    }

    let far_temporal = far_gw_hz * grb_rate_hz * time_window_s;
    let ratio = (far_gw_hz * p_value) / far_gw_max_hz;

    // Clamp ratio to avoid ln of negative or zero
    if ratio <= 0.0 {
        return 0.0;
    }

    far_temporal * p_value * (1.0 - ratio.ln())
}

/// Temporal-only joint FAR (no spatial information).
///
/// FAR_t = FAR_gw × R_grb × Δt
pub fn far_temporal(far_gw_hz: f64, grb_rate_hz: f64, time_window_s: f64) -> f64 {
    far_gw_hz * grb_rate_hz * time_window_s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_far_raven_basic() {
        let far = far_raven(1e-7, GBM_RATE_HZ, 600.0, 0.01);
        assert!(far > 0.0);
        assert!(far.is_finite());
    }

    #[test]
    fn test_far_remapped_basic() {
        let far_gw = 1e-7; // Hz
        let dt = 600.0; // seconds
        let p = 0.05;
        let far_max = 2.0 / 86400.0; // 2 per day

        let far = far_remapped(far_gw, GBM_RATE_HZ, dt, p, far_max);
        assert!(far > 0.0);
        assert!(far.is_finite());

        // Lower p-value should give lower FAR
        let far_low_p = far_remapped(far_gw, GBM_RATE_HZ, dt, 0.001, far_max);
        assert!(far_low_p < far);
    }

    #[test]
    fn test_far_remapped_vs_temporal() {
        // With p=1 the spatial info adds no information, so the corrected FAR
        // should be >= the temporal FAR (the log term adds a penalty).
        let far_gw = 1e-7;
        let dt = 600.0;
        let far_max = 2.0 / 86400.0;

        let ft = far_temporal(far_gw, GBM_RATE_HZ, dt);
        let fc = far_remapped(far_gw, GBM_RATE_HZ, dt, 1.0, far_max);
        // fc = ft × 1.0 × [1 - ln(far_gw / far_max)]
        // Since far_gw << far_max, ln(...) is very negative, so 1-ln > 1, fc > ft
        assert!(fc >= ft);
    }
}
