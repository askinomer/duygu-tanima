"""Veri hazırlığı: RAVDESS ve EmoDB -> 16 kHz mono, sessizliği kırpılmış wav + metadata.csv

Kullanım:
    python src/prepare_data.py --ravdess_dir data/raw/ravdess --emodb_dir data/raw/emodb

Klasörlerin içinde alt klasörler olabilir; betik *.wav dosyalarını özyinelemeli bulur.
Resmi indirme adresleri (bkz. README):
    RAVDESS: https://zenodo.org/records/1188976  (Audio_Speech_Actors_01-24.zip)
    EmoDB  : http://emodb.bilderbar.info/  (veya Interspeech 2025 EmoDB 2.0 sayfası)
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from common import (EMODB_EMO, EMODB_GENDER, COMMON6, PROC, DATA, RAVDESS_EMO, ROOT, SR)

RAVDESS_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav$")
EMODB_RE = re.compile(r"^(\d{2})([ab]\d{2})([WLEAFTN])([a-z])\.wav$")


def load_trim_resample(path: Path, top_db: float = 30.0):
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    dur_raw = len(y) / sr
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    yt, _ = librosa.effects.trim(y, top_db=top_db, frame_length=1024, hop_length=256)
    if len(yt) < SR // 2:  # çok agresif kırpma olursa orijinale dön
        yt = y
    return yt.astype(np.float32), sr, dur_raw


def collect_ravdess(root: Path):
    rows = []
    for p in sorted(root.rglob("*.wav")):
        m = RAVDESS_RE.match(p.name)
        if not m:
            continue
        modality, channel, emo, inten, stmt, rep, actor = m.groups()
        if modality != "03" or channel != "01":  # yalnız ses, yalnız konuşma
            continue
        a = int(actor)
        rows.append(dict(src=p, corpus="ravdess", utt_id=p.stem, speaker=f"R{a:02d}",
                         gender="m" if a % 2 == 1 else "f", emotion=RAVDESS_EMO[emo],
                         intensity="normal" if inten == "01" else "strong",
                         text=f"s{stmt}", rep=int(rep)))
    return rows


def collect_emodb(root: Path):
    rows = []
    for p in sorted(root.rglob("*.wav")):
        m = EMODB_RE.match(p.name)
        if not m:
            continue
        spk, text, emo, ver = m.groups()
        rows.append(dict(src=p, corpus="emodb", utt_id=p.stem, speaker=f"E{spk}",
                         gender=EMODB_GENDER[spk], emotion=EMODB_EMO[emo],
                         intensity="", text=text, rep=0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess_dir", type=Path, default=DATA / "raw" / "ravdess")
    ap.add_argument("--emodb_dir", type=Path, default=DATA / "raw" / "emodb")
    ap.add_argument("--top_db", type=float, default=30.0)
    args = ap.parse_args()

    rows = collect_ravdess(args.ravdess_dir) + collect_emodb(args.emodb_dir)
    n_r = sum(r["corpus"] == "ravdess" for r in rows)
    n_e = sum(r["corpus"] == "emodb" for r in rows)
    print(f"Bulunan: RAVDESS={n_r} (beklenen 1440), EmoDB={n_e} (beklenen 535)")

    out_rows, srs = [], Counter()
    for i, r in enumerate(rows):
        y, sr, dur_raw = load_trim_resample(r["src"], args.top_db)
        srs[(r["corpus"], sr)] += 1
        out = PROC / "16k" / r["corpus"] / f"{r['utt_id']}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), y, SR, subtype="PCM_16")
        rr = {k: v for k, v in r.items() if k != "src"}
        rr.update(path=str(out.relative_to(ROOT)), orig_sr=sr,
                  dur_raw=round(dur_raw, 3), dur=round(len(y) / SR, 3),
                  emotion6=r["emotion"] if r["emotion"] in COMMON6 else "")
        out_rows.append(rr)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(rows)}")

    df = pd.DataFrame(out_rows)
    df.to_csv(DATA / "metadata.csv", index=False)
    print("Örnekleme hızları (orijinal):", dict(srs))
    print(df.groupby(["corpus", "emotion"]).size().unstack(0).fillna(0).astype(int))
    print(df.groupby("corpus")[["dur_raw", "dur"]].describe().round(2).T)
    print("Yazıldı:", DATA / "metadata.csv")


if __name__ == "__main__":
    main()
