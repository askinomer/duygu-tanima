"""Ortak ayarlar: yollar, etiketler, katlar (folds), metrikler ve yardımcı fonksiyonlar.

Bu dosyadaki sabitler projedeki tüm betikler tarafından kullanılır; böylece
A, B ve C aynı etiket sırası, aynı bölme ve aynı metriklerle çalışır.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
FEATS = DATA / "features"
SPLITS = ROOT / "splits"
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"
PRETRAINED = ROOT / "pretrained"

SR = 16_000  # WavLM / wav2vec2 16 kHz bekler

# ---------------------------------------------------------------------------
# Etiketler
# ---------------------------------------------------------------------------
# RAVDESS dosya adı: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
RAVDESS_EMO = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
RAVDESS_CLASSES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# EmoDB dosya adı: SS TTT E V  (konuşmacı, metin kodu, duygu harfi, sürüm)
EMODB_EMO = {
    "W": "angry",    # Ärger (Wut)
    "L": "boredom",  # Langeweile
    "E": "disgust",  # Ekel
    "A": "fearful",  # Angst
    "F": "happy",    # Freude
    "T": "sad",      # Trauer
    "N": "neutral",
}
EMODB_CLASSES = ["neutral", "boredom", "happy", "sad", "angry", "fearful", "disgust"]
EMODB_GENDER = {"03": "m", "08": "f", "09": "f", "10": "m", "11": "m",
                "12": "m", "13": "f", "14": "f", "15": "m", "16": "f"}

# İki veri setinin ortak 6 sınıfı (çapraz veri seti deneyleri için)
COMMON6 = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]

TR_NAMES = {
    "neutral": "nötr", "calm": "sakin", "happy": "mutlu", "sad": "üzgün",
    "angry": "öfkeli", "fearful": "korkmuş", "disgust": "tiksinmiş",
    "surprised": "şaşkın", "boredom": "sıkılmış",
}

# ---------------------------------------------------------------------------
# Konuşmacıdan bağımsız katlar
# ---------------------------------------------------------------------------
# RAVDESS: 24 oyuncu -> 6 kat x 4 oyuncu. Tek numaralılar erkek, çiftler kadın;
# ardışık 4 oyuncu her zaman 2 erkek + 2 kadın verir.
RAVDESS_FOLDS = [[4 * k + i for i in range(1, 5)] for k in range(6)]
# EmoDB: 10 konuşmacı -> 5 kat x (1 erkek + 1 kadın)
EMODB_FOLDS = [["03", "08"], ["10", "09"], ["11", "13"], ["12", "14"], ["15", "16"]]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def metrics(y_true, y_pred, n_classes: int) -> dict:
    """Accuracy (WA), UAR (= ağırlıksız ortalama duyarlılık) ve macro-F1."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(n_classes))
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "uar": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
