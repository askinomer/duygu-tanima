"""Dondurulmuş SSL özellikleri üzerinde sınıflandırıcı eğitimi.

İki mod:
  --mode wls   : 25 katmanın öğrenilebilir ağırlıklı toplamı + MLP (SUPERB tarzı).
                 E2 (dondurulmuş), E3, E4/E5, E6 deneylerinde kullanılır.
  --mode probe : her katman için ayrı lojistik regresyon (katman analizi).

Kullanım:
    python src/ssl_heads.py --exp ravdess8_spk --mode wls
    python src/ssl_heads.py --exp ravdess8_spk --mode wls --augment
    python src/ssl_heads.py --exp ravdess8_spk --mode probe
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from common import FEATS, RESULTS, metrics, save_json, seed_everything
from experiments import get_split, load_table, n_folds


def load_feats(tag: str, variant: str, n_rows: int):
    path = FEATS / tag / f"{variant}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    full = np.full((n_rows,) + d["pooled"].shape[1:], np.nan, dtype=np.float32)
    full[d["idx"]] = d["pooled"].astype(np.float32)
    return full


class WLSHead(nn.Module):
    def __init__(self, n_layers: int, dim: int, n_cls: int, hid: int = 256):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))
        self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, hid), nn.GELU(),
                                 nn.Dropout(0.3), nn.Linear(hid, n_cls))

    def forward(self, x):  # x: (B, L, D) katman başına standartlaştırılmış
        a = torch.softmax(self.w, 0)
        return self.net((a[None, :, None] * x).sum(1))


def train_wls(X_tr, y_tr, X_va, y_va, n_cls, seed, epochs=300, patience=30, bs=64):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model = WLSHead(X_tr.shape[1], X_tr.shape[2], n_cls)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    cnt = np.bincount(y_tr, minlength=n_cls).astype(float)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (n_cls * np.maximum(cnt, 1)),
                                                   dtype=torch.float32))
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    Xv = torch.from_numpy(X_va)
    best, state, bad, hist = -1, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(yt))
        tot = 0.0
        for b in range(0, len(perm), bs):
            bi = perm[b:b + bs]
            loss = crit(model(Xt[bi]), yt[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
        model.eval()
        with torch.no_grad():
            pv = model(Xv).argmax(1).numpy()
        mv = metrics(y_va, pv, n_cls)
        hist.append({"epoch": ep + 1, "train_loss": tot / len(yt),
                     **{f"val_{m}": v for m, v in mv.items()}})
        if mv["uar"] > best:
            best, bad = mv["uar"], 0
            state = {n: p.detach().clone() for n, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model, best, hist


def run_wls(df, feats, exp, k, args):
    sp = get_split(df, exp, k)
    y = sp.labels
    n_cls = len(sp.classes)
    clean = feats["clean"]
    tr_rows = [clean[sp.train]]
    tr_y = [y[sp.train]]
    if args.augment:
        augs = [v for v in ("aug1", "aug2") if feats.get(v) is not None]
        assert augs, "aug1.npz yok; önce ssl_extract.py --variants aug1"
        for v in augs:
            tr_rows.append(feats[v][sp.train])
            tr_y.append(y[sp.train])
    X_tr = np.concatenate(tr_rows)
    y_tr = np.concatenate(tr_y)
    mu = clean[sp.train].mean(0, keepdims=True)  # katman ve boyut başına z-skor
    sd = clean[sp.train].std(0, keepdims=True) + 1e-5
    z = lambda a: ((a - mu) / sd).astype(np.float32)
    model, best, hist = train_wls(z(X_tr), y_tr, z(clean[sp.val]), y[sp.val], n_cls, args.seed + k)
    res = {"exp": exp, "fold": k, "model": args.name, "classes": sp.classes,
           "best_val_uar": best, "epochs_run": len(hist), "history": hist,
           "layer_weights": torch.softmax(model.w, 0).tolist(), "tests": {}}

    def evaluate(X, idx):
        with torch.no_grad():
            p = torch.softmax(model(torch.from_numpy(z(X[idx]))), -1).numpy()
        return p

    for name, idx in sp.tests.items():
        p = evaluate(clean, idx)
        res["tests"][name] = {"metrics": metrics(y[idx], p.argmax(1), n_cls),
                              "utt_id": df.utt_id[idx].tolist(), "y": y[idx].tolist(),
                              "pred": p.argmax(1).tolist(), "prob": np.round(p, 4).tolist()}
        if exp == "ravdess8_spk":
            for v in ("snr10", "snr0"):
                if feats.get(v) is not None:
                    p = evaluate(feats[v], idx)
                    res["tests"][f"{name}_{v}"] = {"metrics": metrics(y[idx], p.argmax(1), n_cls),
                                                   "y": y[idx].tolist(), "pred": p.argmax(1).tolist()}
    return res, model, (mu, sd)


def run_probe(df, feats, exp, k, args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sp = get_split(df, exp, k)
    y = sp.labels
    clean = feats["clean"]
    out = {"exp": exp, "fold": k, "model": args.name, "layers": []}
    for layer in range(clean.shape[1]):
        sc = StandardScaler().fit(clean[sp.train, layer])
        clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(clean[sp.train, layer]), y[sp.train])
        te = sp.tests["ravdess"]
        m = metrics(y[te], clf.predict(sc.transform(clean[te, layer])), len(sp.classes))
        out["layers"].append({"layer": layer, **m})
        print(f"  kat {k} katman {layer:2d}: UAR={m['uar']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="ravdess8_spk")
    ap.add_argument("--tag", default="wavlm_large")
    ap.add_argument("--mode", choices=["wls", "probe"], default="wls")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()
    args.name = f"{args.tag}_frozen_wls" + ("_aug" if args.augment else "") if args.mode == "wls" \
        else f"{args.tag}_probe"

    df = load_table()
    feats = {v: load_feats(args.tag, v, len(df)) for v in ("clean", "aug1", "aug2", "snr10", "snr0")}
    assert feats["clean"] is not None, "Önce ssl_extract.py ile 'clean' özelliklerini çıkarın."
    out_dir = RESULTS / args.exp / args.name
    for k in range(n_folds(args.exp)):
        t0 = time.time()
        if args.mode == "probe":
            res = run_probe(df, feats, args.exp, k, args)
            save_json(res, out_dir / f"fold{k}.json")
            continue
        res, model, norm = run_wls(df, feats, args.exp, k, args)
        res["train_seconds"] = round(time.time() - t0, 1)
        save_json(res, out_dir / f"fold{k}.json")
        summ = " ".join(f"{n}: UAR={v['metrics']['uar']:.3f}" for n, v in res["tests"].items())
        print(f"[{args.name} {args.exp} kat {k}] {res['epochs_run']} epoch | {summ}", flush=True)
        if args.save_model:
            torch.save({"state": model.state_dict(), "mu": norm[0], "sd": norm[1],
                        "classes": res["classes"], "tag": args.tag}, out_dir / f"model_fold{k}.pt")


if __name__ == "__main__":
    main()
