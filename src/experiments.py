"""Deney tanımları: her deney ve kat için eğitim / doğrulama / test indeksleri.

Deneyler (yol haritasındaki E1-E6):
  ravdess8_spk : RAVDESS, 8 sınıf, konuşmacıdan bağımsız 6 kat      -> E1, E2, E6
  ravdess8_rnd : RAVDESS, 8 sınıf, konuşmacı karışık rastgele 6 kat -> E3
  cross6_<SRC> : ortak 6 sınıf, eğitim kaynağı SRC in {R, E, RE},
                 5 eşleştirilmiş kat, test hem RAVDESS hem EmoDB      -> E4, E5
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import COMMON6, DATA, RAVDESS_CLASSES, SPLITS


def load_table() -> pd.DataFrame:
    meta = pd.read_csv(DATA / "metadata.csv")
    sp = pd.read_csv(SPLITS / "splits.csv")[["utt_id", "corpus", "spk_fold", "rnd_fold"]]
    df = meta.merge(sp, on=["utt_id", "corpus"], how="left", validate="one_to_one")
    df["emotion6"] = df["emotion6"].fillna("")
    return df.reset_index(drop=True)


@dataclass
class Split:
    classes: list
    train: np.ndarray
    val: np.ndarray
    tests: dict = field(default_factory=dict)  # ad -> indeksler
    labels: np.ndarray = None                   # tüm tablo için etiket (-1 = kullanılmaz)


def n_folds(exp: str) -> int:
    return 6 if exp.startswith("ravdess8") else 5


def get_split(df: pd.DataFrame, exp: str, k: int) -> Split:
    if exp.startswith("ravdess8"):
        classes = RAVDESS_CLASSES
        is_r = (df.corpus == "ravdess").values
        fold = (df.spk_fold if exp == "ravdess8_spk" else df.rnd_fold).values
        K = 6
        te = np.where(is_r & (fold == k))[0]
        va = np.where(is_r & (fold == (k + 1) % K))[0]
        tr = np.where(is_r & (fold != k) & (fold != (k + 1) % K))[0]
        lab = np.array([classes.index(e) if r else -1 for e, r in zip(df.emotion, is_r)])
        return Split(classes, tr, va, {"ravdess": te}, lab)

    if exp.startswith("cross6_"):
        src = exp.split("_")[1]
        classes = COMMON6
        ok = (df.emotion6 != "").values
        lab = np.array([classes.index(e) if e else -1 for e in df.emotion6])
        r = ok & (df.corpus == "ravdess").values
        e = ok & (df.corpus == "emodb").values
        f = df.spk_fold.values
        r_te, r_va = r & (f == k), r & (f == (k + 1) % 6)
        e_te, e_va = e & (f == k), e & (f == (k + 1) % 5)
        r_tr, e_tr = r & ~r_te & ~r_va, e & ~e_te & ~e_va
        tr = np.zeros(len(df), bool)
        va = np.zeros(len(df), bool)
        if "R" in src:
            tr |= r_tr
            va |= r_va
        if "E" in src:
            tr |= e_tr
            va |= e_va
        tests = {"ravdess": np.where(r_te)[0], "emodb": np.where(e_te)[0]}
        return Split(classes, np.where(tr)[0], np.where(va)[0], tests, lab)

    raise ValueError(exp)


ALL_EXPS = ["ravdess8_spk", "ravdess8_rnd", "cross6_R", "cross6_E", "cross6_RE"]


if __name__ == "__main__":
    df = load_table()
    for exp in ALL_EXPS:
        for k in range(n_folds(exp)):
            s = get_split(df, exp, k)
            spk_tr = set(df.speaker[s.train])
            leak = {n: len(spk_tr & set(df.speaker[idx])) for n, idx in s.tests.items()}
            print(f"{exp:14s} k={k} train={len(s.train):4d} val={len(s.val):4d} "
                  + " ".join(f"test[{n}]={len(i)}" for n, i in s.tests.items())
                  + f"  ortak_konuşmacı={leak}")
