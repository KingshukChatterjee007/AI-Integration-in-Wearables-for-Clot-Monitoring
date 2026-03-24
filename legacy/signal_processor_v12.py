import numpy as np
import pandas as pd
import pywt
from scipy.signal import medfilt, butter, filtfilt
import matplotlib.pyplot as plt
from pathlib import Path

class SignalProcessorV12:
    def __init__(self, fs=64):
        self.fs = fs

    def remove_baseline_wandering(self, signal, cutoff=0.5):
        """
        Removes low-frequency drift using a high-pass Butterworth filter.
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = butter(1, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, signal)

    def denoise_wavelet(self, signal, wavelet='db4', level=4):
        """
        Denoises signal using Discrete Wavelet Transform (DWT).
        """
        # Fix: Ensure signal is a writable numpy array
        sig = np.array(signal, copy=True).astype(np.float64)
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        # Calculate threshold
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        # Thresholding
        coeffs[1:] = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet)

def run_order_comparison():
    # Load 10 seconds of raw BVP data (640 points)
    bvp_path = Path("raw_data_stress/S1/Final/BVP.csv")
    df = pd.read_csv(bvp_path, skiprows=2, header=None)
    raw_signal = df[0].values[1000:2000] # Get a stable segment
    
    processor = SignalProcessorV12(fs=64)
    
    # Order A: DWT -> BW Removal
    dwt_first = processor.denoise_wavelet(raw_signal)
    order_a = processor.remove_baseline_wandering(dwt_first)
    
    # Order B: BW Removal -> DWT
    bw_first = processor.remove_baseline_wandering(raw_signal)
    order_b = processor.denoise_wavelet(bw_first)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(raw_signal, color='gray', alpha=0.5, label='Raw BVP')
    plt.title("Original Raw BVP Signal (S1 Final)")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(order_a, color='blue', label='DWT -> Baseline Removal')
    plt.title("Order A: DWT (Denoise) then Baseline Removal (Drift)")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(order_b, color='green', label='Baseline Removal -> DWT')
    plt.title("Order B: Baseline Removal (Drift) then DWT (Denoise)")
    plt.legend()
    
    plt.tight_layout()
    Path("model_comparison_plots_CLEAN/visualizations").mkdir(parents=True, exist_ok=True)
    plt.savefig("model_comparison_plots_CLEAN/visualizations/v12_order_comparison.png")
    print("Pre-processing comparison saved to model_comparison_plots_CLEAN/visualizations/v12_order_comparison.png")

if __name__ == "__main__":
    run_order_comparison()
