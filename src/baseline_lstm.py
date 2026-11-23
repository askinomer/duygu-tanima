"""Baseline: log-Mel spektrogram + CNN ön katmanı + 2 katmanlı BiLSTM + dikkat havuzlaması (E1, E3, E4/E5, E6).

Kullanım:
    python src/baseline_lstm.py --exp ravdess8_spk            # E1
    python src/baseline_lstm.py --exp ravdess8_rnd            # E3 (baseline)
    python src/baseline_lstm.py --exp cross6_R                # E4/E5 matrisi
    python src/baseline_lstm.py --exp ravdess8_spk --augment  # E6 (baseline)

Her kat için en iyi doğrulama UAR'ına göre erken durdurma yapılır ve test
tahminleri results/<exp>/<model>/fold<k>.json içine yazılır.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

from augment import noisy_test, random_augment
from common import RESULTS, ROOT, SR, metrics, save_json, seed_everything
from experiments import get_split, load_table, n_folds


N_MELS = 64
CROP = 300  # eğitimde 3 sn rastgele kesit (10 ms adım)

_mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=400, win_length=400,
                                            hop_length=160, n_mels=N_MELS)


def logmel(y: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        m = _mel(torch.from_numpy(y).float())
    return torch.log(m + 1e-6).T.numpy()  # (T, n_mels)


class BiLSTM(nn.Module):
    """log-Mel -> 2 katmanlı 1B evrişim (zamanı 4 kat kısaltır) -> 2 katmanlı BiLSTM -> dikkat havuzlaması.

    Evrişim ön katmanı yerel spektral desenleri yakalar ve LSTM'in işleyeceği adım sayısını
    dörtte bire indirir; CPU'da eğitimi yaklaşık 4 kat hızlandırır.
    """

    def __init__(self, n_in: int, n_cls: int, hid: int = 128, drop: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_in, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.lstm = nn.LSTM(128, hid, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=drop)
        self.att = nn.Linear(2 * hid, 1)
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(2 * hid, n_cls)

    def forward(self, x, lengths):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, ~T/4, 128)
        lengths = torch.div(torch.div(lengths + 1, 2, rounding_mode="floor") + 1, 2,
                            rounding_mode="floor").clamp(min=1, max=h.shape[1])
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=int(lengths.max()))
        mask = torch.arange(h.shape[1])[None, :] < lengths[:, None]
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e9)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        return self.out(self.drop((w * h).sum(1)))


def spec_augment(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    x = x.clone()
    T, F = x.shape
    for _ in range(2):
        f = int(rng.integers(0, 9))
        f0 = int(rng.integers(0, F - f + 1))
        x[:, f0:f0 + f] = 0
        t = int(rng.integers(0, 21))
        t0 = int(rng.integers(0, max(1, T - t + 1)))
        x[t0:t0 + t, :] = 0
    return x


def batchify(feats, idx, mean, std, rng=None, train=False):
    xs = []
    for i in idx:
        f = feats[i]
        if train and len(f) > CROP:
            s = int(rng.integers(0, len(f) - CROP + 1))
            f = f[s:s + CROP]
        x = torch.from_numpy((f - mean) / std).float()
        if train:
            x = spec_augment(x, rng)
        xs.append(x)
    lengths = torch.tensor([len(x) for x in xs])
    return nn.utils.rnn.pad_sequence(xs, batch_first=True), lengths


def predict(model, feats, idx, mean, std, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(idx), bs):
            x, L = batchify(feats, idx[b:b + bs], mean, std)
            out.append(torch.softmax(model(x, L), -1).numpy())
    return np.concatenate(out) if out else np.zeros((0, model.out.out_features))


def run_fold(df, waves, clean_feats, exp, k, args, noisy_feats):
    sp = get_split(df, exp, k)
    seed_everything(args.seed + k)
    rng = np.random.default_rng(args.seed + k)
    y = sp.labels
    tr, va = sp.train, sp.val
    stack = np.concatenate([clean_feats[i] for i in tr])
    mean, std = stack.mean(0), stack.std(0) + 1e-5

    model = BiLSTM(N_MELS, len(sp.classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # sınıf dengesizliği için ağırlıklı kayıp (RAVDESS'te nötr sınıf yarı boyutta)
    cnt = np.bincount(y[tr], minlength=len(sp.classes)).astype(float)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (len(cnt) * np.maximum(cnt, 1)),
                                                   dtype=torch.float32))
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(tr)
        tot = 0.0
        t0 = time.time()
        for b in range(0, len(perm), args.bs):
            bi = perm[b:b + args.bs]
            if args.augment:  # her adımda yeni artırılmış dalga biçimi
                feats_b = {i: logmel(random_augment(waves[i], rng)) for i in bi}
            else:
                feats_b = clean_feats
            x, L = batchify(feats_b, bi, mean, std, rng, train=True)
            loss = crit(model(x, L), torch.from_numpy(y[bi]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(bi)
        pv = predict(model, clean_feats, va, mean, std)
        mv = metrics(y[va], pv.argmax(1), len(sp.classes))
        hist.append({"epoch": ep + 1, "train_loss": tot / len(tr), **{f"val_{m}": v for m, v in mv.items()},
                     "sec": round(time.time() - t0, 1)})
        if args.verbose:
            print(f"  kat {k} epoch {ep + 1}: kayıp={tot / len(tr):.3f} val UAR={mv['uar']:.3f} "
                  f"({time.time() - t0:.0f} sn)", flush=True)
        if mv["uar"] > best:
            best, bad = mv["uar"], 0
            best_state = {n: p.detach().clone() for n, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)

    res = {"exp": exp, "fold": k, "model": args.name, "classes": sp.classes,
           "best_val_uar": best, "epochs_run": len(hist), "history": hist, "tests": {}}
    for name, idx in sp.tests.items():
        p = predict(model, clean_feats, idx, mean, std)
        res["tests"][name] = {"metrics": metrics(y[idx], p.argmax(1), len(sp.classes)),
                              "utt_id": df.utt_id[idx].tolist(), "y": y[idx].tolist(),
                              "pred": p.argmax(1).tolist(), "prob": np.round(p, 4).tolist()}
        for snr, nf in noisy_feats.items():
            p = predict(model, nf, idx, mean, std)
            res["tests"][f"{name}_snr{snr}"] = {
                "metrics": metrics(y[idx], p.argmax(1), len(sp.classes)),
                "y": y[idx].tolist(), "pred": p.argmax(1).tolist()}
    return res, model, (mean, std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="ravdess8_spk")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noisy_snr", type=float, nargs="*", default=[10, 0])
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="her epoch'u yazdır")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.name = "baseline_bilstm" + ("_aug" if args.augment else "")

    df = load_table()
    print("Dalga biçimleri ve log-Mel özellikleri yükleniyor...", flush=True)
    waves = {i: sf.read(str(ROOT / p), dtype="float32")[0] for i, p in enumerate(df.path)}
    clean = {i: logmel(w) for i, w in waves.items()}
    noisy = {}
    if args.exp == "ravdess8_spk":
        for snr in args.noisy_snr:
            noisy[int(snr)] = {i: logmel(noisy_test(w, snr, seed=1000 + i))
                               for i, w in waves.items() if df.corpus[i] == "ravdess"}
    print(f"Hazır: {len(clean)} kayıt. Eğitim başlıyor.", flush=True)

    out_dir = RESULTS / args.exp / args.name
    folds = args.folds if args.folds is not None else range(n_folds(args.exp))
    for k in folds:
        t0 = time.time()
        res, model, norm = run_fold(df, waves, clean, args.exp, k, args, noisy)
        res["train_seconds"] = round(time.time() - t0, 1)
        save_json(res, out_dir / f"fold{k}.json")
        summ = " ".join(f"{n}: UAR={v['metrics']['uar']:.3f}" for n, v in res["tests"].items())
        print(f"[{args.name} {args.exp} kat {k}] {res['epochs_run']} epoch, {res['train_seconds']} sn | {summ}",
              flush=True)
        if args.save_model:
            torch.save({"state": model.state_dict(), "mean": norm[0], "std": norm[1],
                        "classes": res["classes"]}, out_dir / f"model_fold{k}.pt")


if __name__ == "__main__":
    main()
