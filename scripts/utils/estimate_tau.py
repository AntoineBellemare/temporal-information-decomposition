"""
Characteristic Timescale (τ) Estimation
========================================

Auto-estimate the intrinsic timescale of a signal for normalized comparisons.

Methods:
1. τ_autocorr: Time for autocorrelation to decay to 1/e
2. τ_zero: First zero-crossing of autocorrelation  
3. τ_period: Dominant oscillation period (1/peak_frequency)
4. τ_halflife: Time for autocorrelation to decay to 0.5
5. τ_integral: Integral timescale (area under |ACF|)

Usage:
    from estimate_tau import estimate_tau, TauEstimator
    
    tau = estimate_tau(signal, fs)
    # or
    estimator = TauEstimator(signal, fs)
    print(estimator.summary())
"""

import numpy as np
from scipy import signal as sig
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import warnings


@dataclass
class TauEstimates:
    """Container for multiple τ estimates."""
    tau_autocorr: float      # 1/e decay time
    tau_halflife: float      # Half-life (decay to 0.5)
    tau_zero: float          # First zero-crossing
    tau_period: float        # Dominant period
    tau_integral: float      # Integral timescale
    tau_best: float          # Recommended single value
    method_used: str         # Which method for tau_best
    fs: float                # Sampling rate
    
    def to_dict(self) -> Dict:
        return {
            'tau_autocorr': self.tau_autocorr,
            'tau_halflife': self.tau_halflife,
            'tau_zero': self.tau_zero,
            'tau_period': self.tau_period,
            'tau_integral': self.tau_integral,
            'tau_best': self.tau_best,
            'method_used': self.method_used,
            'fs': self.fs
        }
    
    def summary(self) -> str:
        return f"""
Timescale Estimates (fs={self.fs:.1f} Hz)
=========================================
τ_autocorr (1/e decay):    {self.tau_autocorr*1000:.2f} ms  ({self.tau_autocorr*self.fs:.1f} samples)
τ_halflife (0.5 decay):    {self.tau_halflife*1000:.2f} ms  ({self.tau_halflife*self.fs:.1f} samples)
τ_zero (first zero):       {self.tau_zero*1000:.2f} ms  ({self.tau_zero*self.fs:.1f} samples)
τ_period (dominant freq):  {self.tau_period*1000:.2f} ms  ({self.tau_period*self.fs:.1f} samples)
τ_integral (ACF integral): {self.tau_integral*1000:.2f} ms  ({self.tau_integral*self.fs:.1f} samples)
-----------------------------------------
τ_best ({self.method_used}):     {self.tau_best*1000:.2f} ms  ({self.tau_best*self.fs:.1f} samples)
"""


class TauEstimator:
    """
    Estimate characteristic timescale of a signal.
    
    The characteristic timescale τ represents the signal's "natural clock" -
    the typical time over which the signal maintains memory of its past.
    
    Parameters
    ----------
    signal : array-like
        1D time series
    fs : float
        Sampling frequency in Hz
    max_lag_seconds : float, optional
        Maximum lag to consider (default: 10 seconds or half signal length)
    detrend : bool, optional
        Whether to detrend the signal first (default: True)
    """
    
    def __init__(self, signal: np.ndarray, fs: float, 
                 max_lag_seconds: Optional[float] = None,
                 detrend: bool = True):
        
        self.signal = np.asarray(signal).flatten()
        self.fs = fs
        self.n = len(self.signal)
        
        # Remove NaNs
        mask = np.isfinite(self.signal)
        if not np.all(mask):
            self.signal = self.signal[mask]
            self.n = len(self.signal)
        
        # Detrend
        if detrend:
            self.signal = sig.detrend(self.signal)
        
        # Normalize
        self.signal = (self.signal - np.mean(self.signal)) / (np.std(self.signal) + 1e-10)
        
        # Max lag
        if max_lag_seconds is None:
            self.max_lag = min(int(10 * fs), self.n // 2)
        else:
            self.max_lag = min(int(max_lag_seconds * fs), self.n // 2)
        
        # Compute autocorrelation
        self._compute_autocorrelation()
        
        # Compute power spectrum
        self._compute_spectrum()
    
    def _compute_autocorrelation(self):
        """Compute normalized autocorrelation function."""
        # Use FFT-based correlation for efficiency
        n = self.n
        
        # Zero-pad for FFT
        nfft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        
        # FFT-based autocorrelation
        fft_sig = np.fft.fft(self.signal, nfft)
        acf_full = np.fft.ifft(fft_sig * np.conj(fft_sig)).real
        
        # Normalize by number of overlapping points
        acf_full = acf_full[:n] / (n - np.arange(n))
        
        # Normalize to start at 1
        acf_full = acf_full / acf_full[0]
        
        # Trim to max_lag
        self.acf = acf_full[:self.max_lag]
        self.lags = np.arange(self.max_lag)
        self.lags_seconds = self.lags / self.fs
    
    def _compute_spectrum(self):
        """Compute power spectrum using Welch's method."""
        # Use Welch's method for robust spectrum
        nperseg = min(self.n // 4, int(4 * self.fs))  # ~4 second windows
        nperseg = max(nperseg, 64)  # At least 64 samples
        
        self.freqs, self.psd = sig.welch(
            self.signal, fs=self.fs, nperseg=nperseg,
            noverlap=nperseg // 2, scaling='density'
        )
    
    def tau_from_autocorr_decay(self, threshold: float = 1/np.e) -> float:
        """
        Find time for ACF to decay to threshold (default 1/e ≈ 0.368).
        
        For exponentially decaying ACF: ACF(τ) = exp(-t/τ)
        At t=τ: ACF = 1/e
        """
        # Find first crossing below threshold
        below_threshold = self.acf < threshold
        
        if not np.any(below_threshold):
            # Never decays below threshold - use max lag
            return self.lags_seconds[-1]
        
        idx = np.argmax(below_threshold)
        
        if idx == 0:
            # Immediate decay - use 1 sample
            return 1 / self.fs
        
        # Linear interpolation for sub-sample precision
        t1, t2 = self.lags_seconds[idx-1], self.lags_seconds[idx]
        a1, a2 = self.acf[idx-1], self.acf[idx]
        
        # Interpolate: find t where acf = threshold
        tau = t1 + (threshold - a1) * (t2 - t1) / (a2 - a1)
        
        return max(tau, 1/self.fs)
    
    def tau_from_halflife(self) -> float:
        """Find time for ACF to decay to 0.5."""
        return self.tau_from_autocorr_decay(threshold=0.5)
    
    def tau_from_zero_crossing(self) -> float:
        """
        Find first zero-crossing of ACF.
        
        For oscillatory signals, this gives ~quarter period.
        For monotonic decay, it may not exist.
        """
        # Find sign changes
        signs = np.sign(self.acf)
        sign_changes = np.diff(signs) != 0
        
        if not np.any(sign_changes):
            # No zero crossing - return NaN or fallback
            return np.nan
        
        idx = np.argmax(sign_changes) + 1
        
        # Linear interpolation
        t1, t2 = self.lags_seconds[idx-1], self.lags_seconds[idx]
        a1, a2 = self.acf[idx-1], self.acf[idx]
        
        # Interpolate: find t where acf = 0
        tau = t1 - a1 * (t2 - t1) / (a2 - a1)
        
        return max(tau, 1/self.fs)
    
    def tau_from_dominant_period(self, freq_range: Tuple[float, float] = None) -> float:
        """
        Find dominant oscillation period from power spectrum.
        
        Parameters
        ----------
        freq_range : tuple, optional
            (min_freq, max_freq) to consider. Default: (0.1, fs/4)
        """
        if freq_range is None:
            freq_range = (0.1, self.fs / 4)
        
        # Mask to frequency range
        mask = (self.freqs >= freq_range[0]) & (self.freqs <= freq_range[1])
        
        if not np.any(mask):
            return np.nan
        
        freqs_masked = self.freqs[mask]
        psd_masked = self.psd[mask]
        
        # Find peak
        peak_idx = np.argmax(psd_masked)
        peak_freq = freqs_masked[peak_idx]
        
        if peak_freq < 1e-10:
            return np.nan
        
        return 1 / peak_freq
    
    def tau_from_integral(self) -> float:
        """
        Compute integral timescale: ∫|ACF(t)|dt from 0 to first zero or max_lag.
        
        This represents the "effective memory" of the signal.
        """
        # Find first zero crossing for integration limit
        zero_idx = None
        signs = np.sign(self.acf)
        sign_changes = np.diff(signs) != 0
        
        if np.any(sign_changes):
            zero_idx = np.argmax(sign_changes) + 1
        
        if zero_idx is None:
            # Integrate to max lag or where ACF < 0.05
            small_acf = np.abs(self.acf) < 0.05
            if np.any(small_acf):
                zero_idx = np.argmax(small_acf)
            else:
                zero_idx = len(self.acf)
        
        # Integrate using trapezoidal rule
        tau_int = np.trapz(np.abs(self.acf[:zero_idx]), self.lags_seconds[:zero_idx])
        
        return max(tau_int, 1/self.fs)
    
    def tau_from_exponential_fit(self) -> Tuple[float, float]:
        """
        Fit exponential decay to ACF: ACF(t) = exp(-t/τ).
        
        Returns
        -------
        tau : float
            Fitted decay constant
        r_squared : float
            Goodness of fit
        """
        # Only fit positive part of ACF
        positive_mask = self.acf > 0.05
        if not np.any(positive_mask):
            return np.nan, 0.0
        
        # Find contiguous positive region from start
        first_negative = np.argmax(~positive_mask)
        if first_negative == 0:
            first_negative = len(positive_mask)
        
        t_fit = self.lags_seconds[:first_negative]
        acf_fit = self.acf[:first_negative]
        
        if len(t_fit) < 3:
            return np.nan, 0.0
        
        # Log-linear fit: log(ACF) = -t/τ
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                log_acf = np.log(acf_fit + 1e-10)
                coeffs = np.polyfit(t_fit, log_acf, 1)
                tau = -1 / coeffs[0] if coeffs[0] < 0 else np.nan
                
                # R-squared
                pred = np.exp(coeffs[0] * t_fit + coeffs[1])
                ss_res = np.sum((acf_fit - pred) ** 2)
                ss_tot = np.sum((acf_fit - np.mean(acf_fit)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                
                return tau, r_squared
            except:
                return np.nan, 0.0
    
    def estimate_all(self, preferred_method: str = 'auto') -> TauEstimates:
        """
        Compute all τ estimates and select best.
        
        Parameters
        ----------
        preferred_method : str
            Which method to use for tau_best:
            - 'auto': Auto-detect (oscillatory→period, else→autocorr)
            - 'autocorr': Force 1/e decay time (good for aperiodic/1/f signals)
            - 'period': Force dominant period (good for oscillatory signals)
            - 'halflife': Force half-life decay
            - 'integral': Force integral timescale
            - 'zero': Force first zero-crossing
        
        Selection logic for 'auto':
        1. If signal is oscillatory (clear spectral peak): use τ_period
        2. If ACF decays smoothly: use τ_autocorr (1/e)
        3. Fallback: use τ_integral
        """
        tau_autocorr = self.tau_from_autocorr_decay(1/np.e)
        tau_halflife = self.tau_from_halflife()
        tau_zero = self.tau_from_zero_crossing()
        tau_period = self.tau_from_dominant_period()
        tau_integral = self.tau_from_integral()
        
        # Select based on preferred method
        if preferred_method == 'autocorr':
            tau_best = tau_autocorr
            method = "autocorr"
        elif preferred_method == 'period':
            tau_best = tau_period if np.isfinite(tau_period) else tau_autocorr
            method = "period" if np.isfinite(tau_period) else "autocorr (fallback)"
        elif preferred_method == 'halflife':
            tau_best = tau_halflife
            method = "halflife"
        elif preferred_method == 'integral':
            tau_best = tau_integral
            method = "integral"
        elif preferred_method == 'zero':
            tau_best = tau_zero if np.isfinite(tau_zero) else tau_autocorr
            method = "zero" if np.isfinite(tau_zero) else "autocorr (fallback)"
        else:  # 'auto'
            # Determine signal type and select best τ
            is_oscillatory = self._is_oscillatory()
            
            if is_oscillatory and np.isfinite(tau_period):
                tau_best = tau_period
                method = "period"
            elif np.isfinite(tau_autocorr) and tau_autocorr > 0:
                tau_best = tau_autocorr
                method = "autocorr"
            elif np.isfinite(tau_integral):
                tau_best = tau_integral
                method = "integral"
            else:
                # Fallback: use 10 samples
                tau_best = 10 / self.fs
                method = "fallback"
        
        # Ensure valid tau_best
        if not np.isfinite(tau_best) or tau_best <= 0:
            tau_best = 10 / self.fs
            method = "fallback"
        
        return TauEstimates(
            tau_autocorr=tau_autocorr,
            tau_halflife=tau_halflife,
            tau_zero=tau_zero if np.isfinite(tau_zero) else -1,
            tau_period=tau_period if np.isfinite(tau_period) else -1,
            tau_integral=tau_integral,
            tau_best=tau_best,
            method_used=method,
            fs=self.fs
        )
    
    def _is_oscillatory(self) -> bool:
        """
        Detect if signal is oscillatory based on spectral properties.
        
        Criteria: Peak prominence > 2x median power
        """
        if len(self.psd) < 5:
            return False
        
        # Find peak in relevant frequency range
        mask = (self.freqs > 0.1) & (self.freqs < self.fs / 4)
        if not np.any(mask):
            return False
        
        psd_masked = self.psd[mask]
        peak_power = np.max(psd_masked)
        median_power = np.median(psd_masked)
        
        # Oscillatory if peak is prominent
        return peak_power > 3 * median_power
    
    def get_acf(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lags_seconds, acf) for plotting."""
        return self.lags_seconds, self.acf
    
    def get_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (frequencies, psd) for plotting."""
        return self.freqs, self.psd
    
    def summary(self) -> str:
        """Return summary string of all estimates."""
        return self.estimate_all().summary()


def estimate_tau(signal: np.ndarray, fs: float, 
                 method: str = 'auto',
                 **kwargs) -> float:
    """
    Convenience function to estimate characteristic timescale.
    
    Parameters
    ----------
    signal : array-like
        1D time series
    fs : float
        Sampling frequency in Hz
    method : str
        'auto' (default), 'autocorr', 'halflife', 'zero', 'period', 'integral'
    
    Returns
    -------
    tau : float
        Characteristic timescale in seconds
    """
    estimator = TauEstimator(signal, fs, **kwargs)
    
    # Use estimate_all with preferred_method for consistent behavior
    return estimator.estimate_all(preferred_method=method).tau_best


def create_normalized_lags(tau: float, fs: float,
                           tau_multiples: list = None) -> np.ndarray:
    """
    Create lag values at normalized τ multiples.
    
    Parameters
    ----------
    tau : float
        Characteristic timescale in seconds
    fs : float
        Sampling frequency
    tau_multiples : list, optional
        Multiples of τ to use. Default: [0.1, 0.25, 0.5, 1, 2, 5, 10]
    
    Returns
    -------
    lags : ndarray
        Lag values in samples (integers)
    """
    if tau_multiples is None:
        tau_multiples = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Convert to samples
    lags_seconds = np.array(tau_multiples) * tau
    lags_samples = np.round(lags_seconds * fs).astype(int)
    
    # Ensure at least 1 sample and unique
    lags_samples = np.maximum(lags_samples, 1)
    lags_samples = np.unique(lags_samples)
    
    return lags_samples


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("CHARACTERISTIC TIMESCALE (τ) ESTIMATION DEMO")
    print("=" * 70)
    
    fs = 256  # Hz
    duration = 30  # seconds
    t = np.arange(0, duration, 1/fs)
    
    # Test signals
    signals = {}
    
    # 1. White noise (τ ≈ 0)
    np.random.seed(42)
    signals['White Noise'] = np.random.randn(len(t))
    
    # 2. AR(1) process (exponential ACF decay)
    ar_coef = 0.95
    ar_signal = np.zeros(len(t))
    ar_signal[0] = np.random.randn()
    for i in range(1, len(t)):
        ar_signal[i] = ar_coef * ar_signal[i-1] + np.sqrt(1 - ar_coef**2) * np.random.randn()
    signals['AR(1) φ=0.95'] = ar_signal
    
    # 3. Sinusoid (oscillatory, τ = period)
    freq = 10  # Hz
    signals['10 Hz Sine'] = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
    
    # 4. Alpha-like oscillation (8-13 Hz bandpass noise)
    noise = np.random.randn(len(t))
    b, a = sig.butter(4, [8, 13], btype='band', fs=fs)
    signals['Alpha Band'] = sig.filtfilt(b, a, noise)
    
    # 5. Slow oscillation + noise
    signals['Slow Osc (1Hz)'] = np.sin(2 * np.pi * 1 * t) + 0.5 * np.random.randn(len(t))
    
    # 6. Random walk (integrated noise)
    signals['Random Walk'] = np.cumsum(np.random.randn(len(t))) / np.sqrt(len(t))
    
    # Analyze each
    fig, axes = plt.subplots(len(signals), 3, figsize=(15, 3*len(signals)))
    
    results = {}
    
    for idx, (name, signal) in enumerate(signals.items()):
        print(f"\n{'='*50}")
        print(f"Signal: {name}")
        print("=" * 50)
        
        estimator = TauEstimator(signal, fs)
        estimates = estimator.estimate_all()
        results[name] = estimates
        
        print(estimates.summary())
        
        # Plot signal
        ax1 = axes[idx, 0]
        ax1.plot(t[:int(2*fs)], signal[:int(2*fs)], 'b-', linewidth=0.5)
        ax1.set_title(f'{name}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.axvline(estimates.tau_best, color='r', linestyle='--', 
                   label=f'τ={estimates.tau_best*1000:.1f}ms')
        ax1.legend(fontsize=8)
        
        # Plot ACF
        ax2 = axes[idx, 1]
        lags_s, acf = estimator.get_acf()
        ax2.plot(lags_s * 1000, acf, 'b-')
        ax2.axhline(1/np.e, color='gray', linestyle=':', alpha=0.5, label='1/e')
        ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax2.axvline(estimates.tau_best * 1000, color='r', linestyle='--', 
                   label=f'τ_best')
        if estimates.tau_autocorr > 0:
            ax2.axvline(estimates.tau_autocorr * 1000, color='g', linestyle=':', 
                       label=f'τ_autocorr', alpha=0.7)
        ax2.set_xlabel('Lag (ms)')
        ax2.set_ylabel('ACF')
        ax2.set_title('Autocorrelation')
        ax2.legend(fontsize=8)
        ax2.set_xlim(0, min(500, lags_s[-1] * 1000))
        
        # Plot spectrum
        ax3 = axes[idx, 2]
        freqs, psd = estimator.get_spectrum()
        ax3.semilogy(freqs, psd, 'b-')
        if estimates.tau_period > 0:
            peak_freq = 1 / estimates.tau_period
            ax3.axvline(peak_freq, color='r', linestyle='--', 
                       label=f'f_peak={peak_freq:.1f}Hz')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('PSD')
        ax3.set_title('Power Spectrum')
        ax3.set_xlim(0, 50)
        ax3.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('tau_estimation_demo.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: tau_estimation_demo.png")
    plt.show()
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Signal':<20} {'τ_best (ms)':<12} {'Method':<12} {'τ in samples':<12}")
    print("-" * 60)
    for name, est in results.items():
        print(f"{name:<20} {est.tau_best*1000:>10.2f}   {est.method_used:<12} {est.tau_best*fs:>10.1f}")
