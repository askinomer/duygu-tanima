"""Veri artırma ve gürültülü test sinyali üretimi.

Dış gürültü veri seti (MUSAN, DEMAND) gerektirmesin diye renkli gürültü
(beyaz, pembe, kahverengi) sentetik olarak üretilir. Hız değişimi Kaldi'deki
"speed perturbation" gibidir: hem tempo hem perde değişir.
"""
from __future__ import annotations

import librosa
import numpy as np

from common import SR


def colored_noise(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    if kind == "white":
        x = white
    else:
        spec = np.fft.rfft(white)
        f = np.fft.rfftfreq(n, d=1.0 / SR)
        f[0] = f[1]
        exponent = {"pink": 0.5, "brown": 1.0}[kind]  # genlik ~ 1/f^exponent
        x = np.fft.irfft(spec / f ** exponent, n=n)
    x = x - x.mean()
    return (x / (np.std(x) + 1e-12)).astype(np.float32)


def add_noise(y: np.ndarray, snr_db: float, kind: str, rng: np.random.Generator) -> np.ndarray:
    noise = colored_noise(len(y), kind, rng)
    p_sig = np.mean(y ** 2) + 1e-12
    p_noise = p_sig / (10 ** (snr_db / 10))
    out = y + noise * np.sqrt(p_noise)
    peak = np.max(np.abs(out))
    return (out / peak * 0.99 if peak > 0.99 else out).astype(np.float32)


def speed_perturb(y: np.ndarray, factor: float) -> np.ndarray:
    return librosa.resample(y, orig_sr=int(round(SR * factor)), target_sr=SR,
                            res_type="soxr_hq").astype(np.float32)


def random_augment(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Eğitim için rastgele artırma: hız (p=0.5), gürültü (p=0.8), kazanç."""
    if rng.random() < 0.5:
        y = speed_perturb(y, float(rng.uniform(0.9, 1.1)))
    if rng.random() < 0.8:
        y = add_noise(y, float(rng.uniform(5, 20)), str(rng.choice(["white", "pink", "brown"])), rng)
    y = y * (10 ** (rng.uniform(-6, 6) / 20))
    peak = np.max(np.abs(y))
    return (y / peak * 0.99 if peak > 0.99 else y).astype(np.float32)


def noisy_test(y: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Tekrar üretilebilir gürültülü test sinyali (pembe gürültü)."""
    return add_noise(y, snr_db, "pink", np.random.default_rng(seed))
