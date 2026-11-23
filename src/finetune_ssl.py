"""WavLM kısmi fine-tune: alt katmanlar dondurulur, üst katmanlar + sınıflandırıcı eğitilir (E2).

Neden kısmi? Sampath vd. (2025) kısmi fine-tuning'in tam fine-tuning'e yakın sonuç verip
çok daha hızlı eğitildiğini gösteriyor. Alt katmanlar dondurulduğu için onların çıktısı
(ssl_extract.py --cache_layers) bir kez hesaplanıp önbelleğe alınır; eğitim yalnız üst
katmanlardan geçer. Bu sayede GPU'suz bir makinede bile 6 katlı deney yapılabilir.

Kullanım (GPU'da da aynı komut çalışır):
    python src/finetune_ssl.py --exp ravdess8_spk --cache_layer 22 --n_layers 2

Model: önbellekteki katman K çıktısı -> WavLM katmanları [K, K+n) (eğitilir)
       -> LayerNorm -> Linear(1024->256) -> zaman ortalaması -> Linear(256->C)
"""
from __future__ import annotations

import argparse
import copy
import gc
import math
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

from common import FEATS, PRETRAINED, RESULTS, metrics, save_json, seed_everything
from experiments import get_split, load_table, n_folds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FrameStore:
    """Kare önbelleğini diskten bellek eşlemeli (memory-map) okur; RAM'e tümüyle yüklemez."""

    def __init__(self, mm, idx, offsets):
        self.mm = mm
        self.pos = {int(i): (int(offsets[j]), int(offsets[j + 1])) for j, i in enumerate(idx)}

    def __getitem__(self, i):
        a, b = self.pos[int(i)]
        return self.mm[a:b]


def load_frames(tag: str, layer: int):
    base = FEATS / tag
    npy = base / f"frames_L{layer}.npy"
    if not npy.exists():  # ssl_extract.py'nin npz çıktısını bir kez dönüştür
        z = np.load(base / f"frames_L{layer}.npz", allow_pickle=True)  # kendi ürettiğimiz dosya
        fr = z["frames"]
        off = np.concatenate([[0], np.cumsum([len(f) for f in fr])])
        mm = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16, shape=(int(off[-1]), int(fr[0].shape[1])))
        for j, f in enumerate(fr):
            mm[off[j]:off[j + 1]] = f
        mm.flush()
        np.savez(base / f"frames_L{layer}_index.npz", idx=z["idx"], offsets=off)
        del mm, z, fr
    ix = np.load(base / f"frames_L{layer}_index.npz")
    return FrameStore(np.load(npy, mmap_mode="r"), ix["idx"], ix["offsets"])


class TopLayers(nn.Module):
    def __init__(self, backbone, start: int, n: int, n_cls: int, drop: float = 0.1):
        super().__init__()
        enc = backbone.encoder
        self.layers = nn.ModuleList([copy.deepcopy(enc.layers[i]) for i in range(start, start + n)])
        self.norm = copy.deepcopy(enc.layer_norm)
        # WavLM göreli konum yanlılığı ilk katmanda hesaplanır ve tüm katmanlara aktarılır
        self.rel = copy.deepcopy(enc.layers[0].attention)
        for p in self.rel.parameters():
            p.requires_grad = False
        self.heads = enc.layers[0].attention.num_heads
        for m in self.layers.modules():
            if isinstance(m, nn.Dropout):
                m.p = drop
        self.proj = nn.Linear(enc.layer_norm.normalized_shape[0], 256)
        self.drop = nn.Dropout(0.2)
        self.cls = nn.Linear(256, n_cls)

    def forward(self, h, mask):  # h: (B, T, D) ; mask: (B, T) 1=geçerli
        B, T, _ = h.shape
        with torch.no_grad():
            pb = self.rel.compute_bias(T, T).to(h.device)  # (H, T, T)
        pb = pb.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.heads, T, T)
        h = h * mask[..., None]
        for i, layer in enumerate(self.layers):
            h = layer(h, attention_mask=mask, position_bias=pb)[0]
        h = self.norm(h)
        z = self.proj(h)
        m = mask[..., None].float()
        pooled = (z * m).sum(1) / m.sum(1).clamp(min=1)
        return self.cls(self.drop(pooled))


def make_batch(frames, idx, crop, rng=None):
    xs = []
    for i in idx:
        f = frames[i]
        if rng is not None and len(f) > crop:
            s = int(rng.integers(0, len(f) - crop + 1))
            f = f[s:s + crop]
        xs.append(torch.from_numpy(f.astype(np.float32)))
    L = torch.tensor([len(x) for x in xs])
    x = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    mask = (torch.arange(x.shape[1])[None] < L[:, None]).long()
    return x, mask


@torch.no_grad()
def predict(model, frames, idx, bs=16):
    model.eval()
    order = sorted(idx, key=lambda i: len(frames[i]))  # benzer uzunluklar -> az dolgu
    probs = {}
    for b in range(0, len(order), bs):
        bi = order[b:b + bs]
        x, m = make_batch(frames, bi, crop=10 ** 9)
        p = torch.softmax(model(x.to(DEVICE), m.to(DEVICE)), -1).cpu().numpy()
        probs.update(zip(bi, p))
    return np.stack([probs[i] for i in idx])


def bucket_batches(idx, frames, bs, rng, pool=8):
    """Karıştır, bs*pool'luk gruplar içinde uzunluğa göre sırala, yığınları karıştır.
    Benzer uzunluktaki kayıtlar aynı yığına düşer; dolgu (padding) hesabı azalır."""
    perm = rng.permutation(idx)
    batches = []
    for g in range(0, len(perm), bs * pool):
        grp = sorted(perm[g:g + bs * pool], key=lambda i: len(frames[i]))
        batches += [np.array(grp[b:b + bs]) for b in range(0, len(grp), bs)]
    order = rng.permutation(len(batches))
    return [batches[j] for j in order]


def run_fold(df, frames, template, exp, k, args):
    sp = get_split(df, exp, k)
    seed_everything(args.seed + k)
    rng = np.random.default_rng(args.seed + k)
    y = sp.labels
    n_cls = len(sp.classes)
    model = copy.deepcopy(template)  # önceden eğitilmiş üst katmanlar
    model.proj = nn.Linear(model.proj.in_features, model.proj.out_features)  # yeni, rastgele başlık
    model.cls = nn.Linear(model.proj.out_features, n_cls)
    model = model.to(DEVICE)
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(("layers", "norm"))]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(("proj", "cls"))]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                             {"params": head_params, "lr": args.lr_head}], weight_decay=0.01)
    steps = args.epochs * math.ceil(len(sp.train) / args.bs)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)) if s >= warm
        else (s + 1) / warm)
    cnt = np.bincount(y[sp.train], minlength=n_cls).astype(float)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (n_cls * np.maximum(cnt, 1)),
                                                   dtype=torch.float32).to(DEVICE))
    best, state, bad, hist = -1, None, 0, []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        for bi in bucket_batches(sp.train, frames, args.bs, rng):
            x, m = make_batch(frames, bi, args.crop, rng)
            loss = crit(model(x.to(DEVICE), m.to(DEVICE)), torch.from_numpy(y[bi]).to(DEVICE))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(bi)
        pv = predict(model, frames, sp.val)
        mv = metrics(y[sp.val], pv.argmax(1), n_cls)
        hist.append({"epoch": ep + 1, "train_loss": tot / len(sp.train),
                     **{f"val_{q}": v for q, v in mv.items()}, "sec": round(time.time() - t0, 1)})
        print(f"  kat {k} epoch {ep + 1}: kayıp={tot / len(sp.train):.3f} val UAR={mv['uar']:.3f} "
              f"({time.time() - t0:.0f} sn)", flush=True)
        if mv["uar"] > best:
            best, bad = mv["uar"], 0
            state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict({n: p.to(DEVICE) for n, p in state.items()})
    res = {"exp": exp, "fold": k, "model": args.name, "classes": sp.classes, "best_val_uar": best,
           "epochs_run": len(hist), "history": hist, "config": vars(args) | {"name": args.name}, "tests": {}}
    for name, idx in sp.tests.items():
        p = predict(model, frames, idx)
        res["tests"][name] = {"metrics": metrics(y[idx], p.argmax(1), n_cls),
                              "utt_id": df.utt_id[idx].tolist(), "y": y[idx].tolist(),
                              "pred": p.argmax(1).tolist(), "prob": np.round(p, 4).tolist()}
    return res, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="ravdess8_spk")
    ap.add_argument("--model", default=str(PRETRAINED / "wavlm-large-hf"))
    ap.add_argument("--tag", default="wavlm_large")
    ap.add_argument("--cache_layer", type=int, default=22)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=150, help="eğitimde en fazla kare (20 ms) = 3 sn")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()
    args.name = args.name or f"{args.tag}_ft{args.n_layers}"

    df = load_table()
    backbone = AutoModel.from_pretrained(args.model)
    template = TopLayers(backbone, args.cache_layer, args.n_layers, n_cls=2)
    del backbone  # tam model (1,2 GB) bellekte tutulmaz; yalnız gereken katmanlar kopyalandı
    gc.collect()
    frames = load_frames(args.tag, args.cache_layer)
    out_dir = RESULTS / args.exp / args.name
    for k in (args.folds if args.folds is not None else range(n_folds(args.exp))):
        t0 = time.time()
        res, model = run_fold(df, frames, template, args.exp, k, args)
        res["train_seconds"] = round(time.time() - t0, 1)
        save_json(res, out_dir / f"fold{k}.json")
        summ = " ".join(f"{n}: UAR={v['metrics']['uar']:.3f}" for n, v in res["tests"].items())
        print(f"[{args.name} {args.exp} kat {k}] {res['epochs_run']} epoch, {res['train_seconds']:.0f} sn | {summ}",
              flush=True)
        if args.save_model:
            torch.save({"state": model.state_dict(), "classes": res["classes"], "config": res["config"]},
                       out_dir / f"model_fold{k}.pt")


if __name__ == "__main__":
    main()
