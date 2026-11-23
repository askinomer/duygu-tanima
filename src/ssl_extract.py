"""Önceden eğitilmiş SSL modelinden (WavLM / wav2vec2 / HuBERT) özellik çıkarımı.

Her kayıt için modelin TÜM katman çıktıları (CNN sonrası + 24 transformer katmanı = 25)
zaman ekseninde ortalama ve standart sapma ile havuzlanır -> (25, 2*D) vektör.
İsteğe bağlı olarak bir katmanın kare-düzeyi (frame-level) çıktısı da saklanır;
finetune_ssl.py üst katmanları bu önbellekten başlayarak eğitir.

Kullanım:
    python src/ssl_extract.py --model microsoft/wavlm-large --variants clean snr10 snr0 aug1 aug2
    # İnternetsiz ortamda dönüştürülmüş yerel klasör:
    python src/ssl_extract.py --model pretrained/wavlm-large-hf ...

Çıktı: data/features/<etiket>/<varyant>.npz  (kesintiye dayanıklı: parça parça kaydeder)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

from augment import noisy_test, random_augment
from common import FEATS, ROOT
from experiments import load_table


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_backbone(name: str):
    model = AutoModel.from_pretrained(name)
    model.eval()
    return model.to(DEVICE)


def normalize(y: np.ndarray) -> np.ndarray:
    # WavLM-Large / wav2vec2-large: sıfır ortalama, birim varyans (orijinal kodda layer_norm)
    return (y - y.mean()) / (y.std() + 1e-7)


@torch.inference_mode()
def forward_all(model, y: np.ndarray, cache_layers: list[int]):
    x = torch.from_numpy(normalize(y)).float()[None].to(DEVICE)
    out = model(x, output_hidden_states=True)
    # transformers>=5: hidden_states[0] = transformer girişi, [i] = i. katmanın çıkışı (i=1..24)
    hs = torch.stack(out.hidden_states, 0)[:, 0].float().cpu()  # (L+1, T, D)
    pooled = torch.cat([hs.mean(1), hs.std(1)], -1)  # (L+1, 2D)
    frames = {l: hs[l].half().numpy() for l in cache_layers}
    assert torch.isfinite(hs).all() and hs.abs().max() < 6e4, "fp16 taşması riski"
    return pooled.float().numpy(), frames


def variant_wave(df, i: int, variant: str) -> np.ndarray:
    y = sf.read(str(ROOT / df.path[i]), dtype="float32")[0]
    if variant == "clean":
        return y
    if variant.startswith("snr"):
        return noisy_test(y, float(variant[3:]), seed=1000 + i)
    if variant.startswith("aug"):
        j = int(variant[3:])
        return random_augment(y, np.random.default_rng(10_000 * j + i))
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/wavlm-large")
    ap.add_argument("--tag", default="wavlm_large")
    ap.add_argument("--variants", nargs="+", default=["clean"])
    ap.add_argument("--cache_layers", type=int, nargs="*", default=[12, 16, 20, 22],
                    help="kare-düzeyi saklanacak katmanlar (yalnız 'clean' için); boş = saklama")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="hata ayıklama: yalnız ilk N kayıt")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    df = load_table()
    model = load_backbone(args.model)
    out_dir = FEATS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        # gürültülü ve artırılmış varyantlar yalnız RAVDESS için gerekiyor
        sel = np.arange(len(df)) if variant == "clean" else np.where(df.corpus == "ravdess")[0]
        if args.limit:
            sel = sel[:args.limit]
        final = out_dir / f"{variant}.npz"
        if final.exists():
            print(f"{final} zaten var, atlanıyor")
            continue
        part_dir = out_dir / f"_parts_{variant}"
        part_dir.mkdir(exist_ok=True)
        cache = args.cache_layers if variant == "clean" else []
        t_start = time.time()
        for c0 in range(0, len(sel), args.chunk):
            pf = part_dir / f"{c0:05d}.npz"
            if pf.exists():
                continue
            idx = sel[c0:c0 + args.chunk]
            pooled, frames = [], {l: [] for l in cache}
            for i in idx:
                p, f = forward_all(model, variant_wave(df, i, variant), cache)
                pooled.append(p)
                for l in cache:
                    frames[l].append(f[l])
            extra = {f"frames_L{l}": np.array(frames[l] + [None], dtype=object)[:-1] for l in cache}
            tmp = pf.with_name(pf.stem + "_tmp.npz")
            np.savez(tmp, idx=idx, pooled=np.stack(pooled), **extra)
            tmp.rename(pf)  # yarım yazılmış parça kalmasın
            done = min(c0 + args.chunk, len(sel))
            el = time.time() - t_start
            print(f"[{variant}] {done}/{len(sel)}  {el / 60:.1f} dk", flush=True)
        parts = sorted(p for p in part_dir.glob("*.npz") if not p.stem.endswith("_tmp"))
        data = [np.load(p, allow_pickle=True) for p in parts]
        idx = np.concatenate([d["idx"] for d in data])
        pooled = np.concatenate([d["pooled"] for d in data])
        for l in cache:  # önce kare önbellekleri, en son ana dosya (yarıda kesilirse tekrar üretilir)
            frames = [f for d in data for f in d[f"frames_L{l}"]]
            np.savez(out_dir / f"frames_L{l}.npz", idx=idx,
                     frames=np.array(frames + [None], dtype=object)[:-1])
        tmp = out_dir / f"{variant}_tmp.npz"
        np.savez(tmp, idx=idx, utt_id=df.utt_id.values[idx], pooled=pooled)
        tmp.rename(final)
        for p in parts:
            p.unlink()
        part_dir.rmdir()
        print(f"Yazıldı: {final}  pooled={pooled.shape}")


if __name__ == "__main__":
    main()
