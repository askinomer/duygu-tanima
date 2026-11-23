"""Demo için son model: RAVDESS + EmoDB ortak 6 sınıf, dondurulmuş WavLM + ağırlıklı katman toplamı.

Doğrulama (erken durdurma) için RAVDESS'ten 4 oyuncu (kat 5) ve EmoDB'den 2 konuşmacı (kat 4)
ayrılır; geri kalan herkes eğitimde. RAVDESS'in iki artırılmış kopyası da eğitime eklenir
(mikrofon gürültüsüne dayanıklılık için).

Kullanım:
    python src/train_demo_model.py
Çıktı: models/demo_wavlm_wls_common6.pt
"""
from __future__ import annotations

import numpy as np
import torch

from common import COMMON6, ROOT, metrics, save_json
from experiments import load_table
from ssl_heads import load_feats, train_wls


def main(tag: str = "wavlm_large", seed: int = 42):
    df = load_table()
    feats = {v: load_feats(tag, v, len(df)) for v in ("clean", "aug1", "aug2")}
    y = np.array([COMMON6.index(e) if e else -1 for e in df.emotion6])
    ok = y >= 0
    val = ok & (((df.corpus == "ravdess") & (df.spk_fold == 5)) | ((df.corpus == "emodb") & (df.spk_fold == 4))).values
    tr = ok & ~val
    tr_i, va_i = np.where(tr)[0], np.where(val)[0]
    X = [feats["clean"][tr_i]]
    Y = [y[tr_i]]
    rav_tr = tr_i[(df.corpus.values[tr_i] == "ravdess")]
    for v in ("aug1", "aug2"):
        if feats[v] is not None:
            X.append(feats[v][rav_tr])
            Y.append(y[rav_tr])
    X, Y = np.concatenate(X), np.concatenate(Y)
    mu = feats["clean"][tr_i].mean(0, keepdims=True)
    sd = feats["clean"][tr_i].std(0, keepdims=True) + 1e-5
    z = lambda a: ((a - mu) / sd).astype(np.float32)
    model, best, hist = train_wls(z(X), Y, z(feats["clean"][va_i]), y[va_i], len(COMMON6), seed)
    with torch.no_grad():
        pv = model(torch.from_numpy(z(feats["clean"][va_i]))).argmax(1).numpy()
    m = {c: metrics(y[va_i][df.corpus.values[va_i] == c], pv[df.corpus.values[va_i] == c], 6)
         for c in ("ravdess", "emodb")}
    print("Doğrulama:", m)
    out = ROOT / "models" / "demo_wavlm_wls_common6.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "classes": COMMON6, "tag": tag}, out)
    save_json({"val": m, "n_train": int(len(Y)), "epochs": len(hist)}, out.with_suffix(".json"))
    print("Kaydedildi:", out)


if __name__ == "__main__":
    main()
