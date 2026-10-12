"""Period estimation and cycle counting for repeated motion in real footage.

Optical-flow and frame-difference features are computed per image cell at three
cell sizes, so the most periodic region of the frame dominates without being
named. Autocorrelation gives the period, robust to tempo drift and to slow
camera or lighting trends. The dominant band-passed component is then peak
counted.
"""
import numpy as np

CELL_SIZES = (8, 16, 32)
MIN_PERIOD_SEC = 0.25
MAX_PERIOD_FRACTION = 0.6
PEAK_RELATIVE = 0.7
MIN_PERIODICITY = 0.05
BAND = (0.6, 1.6)
PEAK_AMPLITUDE = 0.4


def _blur(frames):
    kernel = np.array([1, 2, 1], np.float32) / 4
    padded = np.pad(frames, ((0, 0), (0, 0), (1, 1)), mode="edge")
    frames = kernel[0] * padded[:, :, :-2] + kernel[1] * frames + kernel[2] * padded[:, :, 2:]
    padded = np.pad(frames, ((0, 0), (1, 1), (0, 0)), mode="edge")
    return kernel[0] * padded[:, :-2] + kernel[1] * frames + kernel[2] * padded[:, 2:]


def _pool(values, size):
    frames, height, width = values.shape
    trimmed = values[:, :height // size * size, :width // size * size]
    return trimmed.reshape(frames, height // size, size, width // size, size).mean(axis=(2, 4))


def _flow(frames, size):
    """Lucas-Kanade motion per size x size cell, as one row per frame."""
    blurred = _blur(_blur(frames))
    before, after = blurred[:-1], blurred[1:]
    mean = 0.5 * (before + after)
    gx = np.zeros_like(mean)
    gy = np.zeros_like(mean)
    gx[:, :, 1:-1] = (mean[:, :, 2:] - mean[:, :, :-2]) / 2
    gy[:, 1:-1, :] = (mean[:, 2:, :] - mean[:, :-2, :]) / 2
    dt = after - before
    sxx, syy, sxy = _pool(gx * gx, size), _pool(gy * gy, size), _pool(gx * gy, size)
    sxt, syt = _pool(gx * dt, size), _pool(gy * dt, size)
    ridge = 1e-4 + 0.02 * float((sxx + syy).mean())
    det = (sxx + ridge) * (syy + ridge) - sxy ** 2
    u = -((syy + ridge) * sxt - sxy * syt) / det
    v = -((sxx + ridge) * syt - sxy * sxt) / det
    u, v = np.concatenate([u[:1], u]), np.concatenate([v[:1], v])
    return np.concatenate([u.reshape(len(u), -1), v.reshape(len(v), -1)], axis=1)


def _features(frames):
    frames = np.asarray(frames, dtype=np.float32) / 255
    groups = []
    for size in (s for s in CELL_SIZES if s <= min(frames.shape[1:])):
        groups.append(_flow(frames, size))
        pooled = _pool(frames, size)
        groups.append(np.abs(np.diff(pooled, axis=0, prepend=pooled[:1])).reshape(len(frames), -1))
    return groups


def _detrend(values):
    """Remove drift slower than half the window with a moving average."""
    window = max(3, (len(values) // 2) | 1)
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="reflect")
    sums = np.cumsum(np.vstack([np.zeros((1, values.shape[1])), padded]), axis=0)
    return values - ((sums[window:] - sums[:-window]) / window)[:len(values)]


def _autocorrelation(values):
    n = len(values)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(values, n=size, axis=0)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), axis=0)[:n]
    return corr / (n - np.arange(n))[:, None]


def _local_peaks(curve, low, high):
    return [(i, float(curve[i])) for i in range(max(low, 1), min(high, len(curve) - 1))
            if curve[i] > curve[i - 1] and curve[i] >= curve[i + 1]]


def _refine(curve, i):
    if 1 <= i < len(curve) - 1:
        left, mid, right = curve[i - 1], curve[i], curve[i + 1]
        denominator = left - 2 * mid + right
        if denominator < 0:
            return i + 0.5 * (left - right) / denominator
    return float(i)


def _separated_peaks(signal, spacing, threshold):
    candidates = [i for i in range(1, len(signal) - 1)
                  if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]
                  and signal[i] > 0 and signal[i] >= threshold]
    kept = []
    for i in sorted(candidates, key=lambda j: -signal[j]):
        if all(abs(i - j) >= spacing for j in kept):
            kept.append(i)
    return sorted(kept)


def estimate(frames, fps, start_sec=0.0):
    """Return period and cycles of the dominant periodic motion, or None if there is none.

    frames: grayscale uint8 array (time, height, width) sampled at fps.
    """
    n = len(frames)
    if n < 8 or min(np.shape(frames)[1:]) < min(CELL_SIZES):
        return None
    groups = []
    for values in _features(frames):
        values = _detrend(values)
        groups.append(values - values.mean(axis=0))
    curves = []
    for values in groups:
        corr = _autocorrelation(values)
        curves.append(corr.sum(axis=1) / max(float(corr[0].sum()), 1e-12))
    curve = np.mean(curves, axis=0)
    low, high = max(2, int(MIN_PERIOD_SEC * fps)), max(4, int(n * MAX_PERIOD_FRACTION))
    peaks = _local_peaks(curve, low, high)
    if not peaks:
        return None
    best = max(v for _, v in peaks)
    if best < MIN_PERIODICITY:
        return None
    lag = next(i for i, v in peaks if v >= PEAK_RELATIVE * best)
    period = _refine(curve, lag) / fps
    combined = np.concatenate([v / max(float(np.sqrt((v ** 2).sum())), 1e-12) for v in groups], axis=1)
    spectrum = np.fft.rfft(combined, axis=0)
    freqs = np.fft.rfftfreq(n, 1 / fps)
    keep = (freqs >= BAND[0] / period) & (freqs <= BAND[1] / period)
    banded = np.fft.irfft(spectrum * keep[:, None], n=n, axis=0)
    u, s, _ = np.linalg.svd(banded, full_matrices=False)
    component = u[:, 0] * s[0]
    analytic = np.fft.ifft(np.fft.fft(component) * _analytic_mask(n))
    threshold = PEAK_AMPLITUDE * float(np.percentile(np.abs(analytic), 90))
    spacing = 0.6 * period * fps
    maxima = _separated_peaks(component, spacing, threshold)
    minima = _separated_peaks(-component, spacing, threshold)
    chosen = maxima if len(maxima) >= len(minima) else minima
    half = period / 2
    end_limit = start_sec + n / fps
    cycles = [{"index": k + 1,
               "start_sec": round(max(start_sec, start_sec + i / fps - half), 3),
               "end_sec": round(min(end_limit, start_sec + i / fps + half), 3),
               "peak_sec": round(start_sec + i / fps, 3)}
              for k, i in enumerate(chosen)]
    return {"count": len(cycles), "cycles": cycles, "period_sec": round(period, 3),
            "periodicity": round(best, 3)}


def _analytic_mask(n):
    mask = np.zeros(n)
    mask[0] = 1
    if n % 2 == 0:
        mask[n // 2] = 1
        mask[1:n // 2] = 2
    else:
        mask[1:(n + 1) // 2] = 2
    return mask
