"""Sonuçları toplar: kat ortalaması ± std tabloları, confusion matrix'ler ve rapor grafikleri.

Kullanım:
    python src/analyze_results.py
Çıktılar:
    results/summary.csv, results/summary.md
    report/figures/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from common import DATA, FIGURES, RESULTS, TR_NAMES

# --- Görsel kimlik (dataviz referans paleti) ---------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
    "legend.frameon": False,
})

MODEL_TR = {
    "baseline_bilstm": "Baseline (log-Mel + CNN-BiLSTM)",
    "baseline_bilstm_aug": "Baseline + artırma",
    "wavlm_large_frozen_wls": "WavLM-L dondurulmuş (ağırlıklı katman)",
    "wavlm_large_frozen_wls_aug": "WavLM-L dondurulmuş + artırma",
    "wavlm_large_ft2": "WavLM-L kısmi fine-tune (üst 2 katman)",
}


def load_runs():
    rows, runs = [], {}
    for f in sorted(RESULTS.glob("*/*/fold*.json")):
        exp, model = f.parent.parent.name, f.parent.name
        r = json.loads(f.read_text())
        if "tests" not in r:
            continue
        runs.setdefault((exp, model), []).append(r)
        for test, v in r["tests"].items():
            rows.append({"exp": exp, "model": model, "fold": r["fold"], "test": test, **v["metrics"]})
    return pd.DataFrame(rows), runs


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["exp", "model", "test"])
    s = g.agg(n=("fold", "nunique"), acc=("acc", "mean"), acc_sd=("acc", "std"),
              uar=("uar", "mean"), uar_sd=("uar", "std"), f1=("f1", "mean"), f1_sd=("f1", "std"))
    return s.reset_index()


def fmt(m, s):
    return f"{100 * m:.1f} ± {100 * s:.1f}"


def write_tables(s: pd.DataFrame):
    s.to_csv(RESULTS / "summary.csv", index=False)
    lines = ["| Deney | Model | Test | Kat | Accuracy (%) | UAR (%) | Macro-F1 (%) |",
             "|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        lines.append(f"| {r.exp} | {r.model} | {r.test} | {r.n} | {fmt(r.acc, r.acc_sd)} | "
                     f"{fmt(r.uar, r.uar_sd)} | {fmt(r.f1, r.f1_sd)} |")
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def pooled_cm(runs, test="ravdess"):
    ys, ps = [], []
    for r in runs:
        ys += r["tests"][test]["y"]
        ps += r["tests"][test]["pred"]
    return np.array(ys), np.array(ps)


def fig_confusion(runs, model, exp="ravdess8_spk", test="ravdess"):
    rs = runs.get((exp, model))
    if not rs:
        return
    classes = rs[0]["classes"]
    y, p = pooled_cm(rs, test)
    cm = confusion_matrix(y, p, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#ffffff"] + BLUES)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    names = [TR_NAMES[c] for c in classes]
    ax.set_xticks(range(len(classes)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), names)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.5 else INK)
    ax.set_xlabel("Tahmin edilen")
    ax.set_ylabel("Gerçek")
    ax.set_title(f"{MODEL_TR.get(model, model)}\nsatır yüzdesi, 6 katın birleşik test tahminleri",
                 fontsize=9.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig_confusion_{model}.png")
    plt.close(fig)
    return cm


def bar_groups(ax, groups, series, values, errors, colors, labels, fmt_pct=True):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    for i, (sname, col) in enumerate(zip(series, colors)):
        v = np.array(values[i])
        e = np.array(errors[i])
        pos = x - 0.4 + w / 2 + i * w
        ax.bar(pos, v, w * 0.92, color=col, label=labels[i], zorder=2)
        ax.errorbar(pos, v, yerr=e, fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
        for xx, vv, ee in zip(pos, v, e):
            if np.isfinite(vv):
                ax.text(xx, vv + ee + 1.0, f"{vv:.1f}".replace(".", ","), ha="center", va="bottom",
                        fontsize=8, color=INK)
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)


def get(s, exp, model, test, col="uar"):
    q = s[(s.exp == exp) & (s.model == model) & (s.test == test)]
    if q.empty:
        return np.nan, np.nan
    return 100 * q[col].iloc[0], 100 * q[col + "_sd"].iloc[0]


def fig_main(s):
    models = [m for m in ["baseline_bilstm", "wavlm_large_frozen_wls", "wavlm_large_ft2"]
              if not s[(s.model == m) & (s.exp == "ravdess8_spk")].empty]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    metrics_ = [("acc", "Accuracy"), ("uar", "UAR"), ("f1", "Macro-F1")]
    vals = [[get(s, "ravdess8_spk", m, "ravdess", c)[0] for c, _ in metrics_] for m in models]
    errs = [[get(s, "ravdess8_spk", m, "ravdess", c)[1] for c, _ in metrics_] for m in models]
    bar_groups(ax, [l for _, l in metrics_], models, vals, errs, [MUTED, S1, S2][:len(models)],
               [MODEL_TR[m] for m in models])
    ax.set_ylim(0, 100)
    ax.set_ylabel("%  (6 kat ortalaması ± std)")
    ax.axhline(12.5, color=AXIS, lw=1, ls="--", zorder=1)
    ax.set_title("RAVDESS, 8 sınıf, konuşmacıdan bağımsız test (kesikli çizgi: şans düzeyi %12,5)",
                 loc="left", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_main_comparison.png")
    plt.close(fig)


def fig_leakage(s):
    models = [m for m in ["baseline_bilstm", "wavlm_large_frozen_wls"]
              if not s[(s.model == m) & (s.exp == "ravdess8_rnd")].empty]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    vals = [[get(s, "ravdess8_spk", m, "ravdess")[0] for m in models],
            [get(s, "ravdess8_rnd", m, "ravdess")[0] for m in models]]
    errs = [[get(s, "ravdess8_spk", m, "ravdess")[1] for m in models],
            [get(s, "ravdess8_rnd", m, "ravdess")[1] for m in models]]
    bar_groups(ax, [MODEL_TR[m].split(" (")[0] for m in models], ["spk", "rnd"], vals, errs,
               [S1, S2], ["Konuşmacıdan bağımsız bölme", "Rastgele bölme (konuşmacı sızıntısı)"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("UAR %  (6 kat ort. ± std)")
    ax.set_title("Rastgele bölme başarıyı olduğundan yüksek gösterir", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_leakage.png")
    plt.close(fig)


def fig_cross(s):
    models = [m for m in ["baseline_bilstm", "wavlm_large_frozen_wls"]
              if not s[(s.model == m) & (s.exp == "cross6_R")].empty]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), squeeze=False)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#ffffff"] + BLUES)
    for ax, m in zip(axes[0], models):
        srcs = ["R", "E", "RE"]
        M = np.array([[get(s, f"cross6_{src}", m, t)[0] for t in ["ravdess", "emodb"]] for src in srcs])
        SD = np.array([[get(s, f"cross6_{src}", m, t)[1] for t in ["ravdess", "emodb"]] for src in srcs])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100)
        ax.grid(False)
        ax.set_xticks([0, 1], ["Test:\nRAVDESS", "Test:\nEmoDB"])
        ax.set_yticks([0, 1, 2], ["Eğitim: RAVDESS", "Eğitim: EmoDB", "Eğitim: ikisi"] if ax is axes[0][0]
                      else ["", "", ""])
        for i in range(3):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]:.1f}\n±{SD[i, j]:.1f}".replace(".", ","), ha="center", va="center",
                        fontsize=9, color="white" if M[i, j] > 55 else INK)
        ax.set_title(MODEL_TR[m].split(" (")[0], fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Ortak 6 sınıfta UAR (%), 5 kat ort. ± std", x=0.02, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_cross_corpus.png")
    plt.close(fig)


def fig_noise(s):
    models = [m for m in ["baseline_bilstm", "baseline_bilstm_aug", "wavlm_large_frozen_wls",
                          "wavlm_large_frozen_wls_aug"]
              if not s[(s.model == m) & (s.exp == "ravdess8_spk")].empty]
    if len(models) < 2:
        return
    conds = [("ravdess", "Temiz"), ("ravdess_snr10", "SNR 10 dB"), ("ravdess_snr0", "SNR 0 dB")]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    vals = [[get(s, "ravdess8_spk", m, c)[0] for c, _ in conds] for m in models]
    errs = [[get(s, "ravdess8_spk", m, c)[1] for c, _ in conds] for m in models]
    cols = {"baseline_bilstm": "#b7b5ad", "baseline_bilstm_aug": MUTED,
            "wavlm_large_frozen_wls": "#86b6ef", "wavlm_large_frozen_wls_aug": S1}
    bar_groups(ax, [l for _, l in conds], models, vals, errs, [cols[m] for m in models],
               [MODEL_TR[m] for m in models])
    ax.set_ylim(0, 100)
    ax.set_ylabel("UAR %  (6 kat ort. ± std)")
    ax.set_title("Pembe gürültülü testte veri artırmanın etkisi", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_noise_robustness.png")
    plt.close(fig)


def fig_layers(runs):
    files = sorted((RESULTS / "ravdess8_spk" / "wavlm_large_probe").glob("fold*.json"))
    if not files:
        return
    per = np.array([[l["uar"] for l in json.loads(f.read_text())["layers"]] for f in files]) * 100
    rs = runs.get(("ravdess8_spk", "wavlm_large_frozen_wls"))
    fig, axes = plt.subplots(2 if rs else 1, 1, figsize=(6.6, 4.6 if rs else 3.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]} if rs else None, squeeze=False)
    ax = axes[0, 0]
    L = np.arange(per.shape[1])
    m = per.mean(0)
    ax.plot(L, m, color=S1, lw=2, marker="o", ms=4)
    ax.fill_between(L, m - per.std(0), m + per.std(0), color=S1, alpha=0.15, lw=0)
    best = int(m.argmax())
    ax.annotate(f"en iyi: katman {best}, UAR %{m[best]:.1f}".replace(".", ","), (best, m[best]),
                xytext=(0.5, m.max() + 3), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_ylim(min(30, m.min() - 8), m.max() + 9)
    ax.set_ylabel("UAR %\n(katman başına lojistik\nregresyon, 6 kat)")
    ax.set_title("Duygu bilgisi WavLM'in orta katmanlarında yoğunlaşıyor", loc="left")
    if rs:
        w = np.mean([r["layer_weights"] for r in rs], 0)
        a2 = axes[1, 0]
        a2.bar(L, w, color=S2, width=0.7)
        a2.axhline(1 / len(w), color=INK2, lw=1, ls="--")
        a2.text(len(w) - 0.4, 1 / len(w) * 1.04, "eşit ağırlık (1/25)", ha="right", va="bottom",
                fontsize=8, color=INK2)
        a2.set_ylim(0, max(w) * 1.25)
        a2.set_ylabel("öğrenilen\nkatman ağırlığı")
        a2.grid(axis="x", visible=False)
    axes[-1, 0].set_xlabel("WavLM-Large katmanı (0 = transformer girişi, 24 = son katman)")
    axes[-1, 0].set_xticks(L[::2])
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_layer_analysis.png")
    plt.close(fig)
    return per


def fig_curves(runs):
    pairs = [("baseline_bilstm", MUTED), ("wavlm_large_ft2", S2), ("wavlm_large_frozen_wls", S1)]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for m, col in pairs:
        rs = runs.get(("ravdess8_spk", m))
        if not rs:
            continue
        r = rs[0]
        ep = [h["epoch"] for h in r["history"]]
        axes[0].plot(ep, [h["train_loss"] for h in r["history"]], color=col, lw=1.8,
                     label=MODEL_TR[m].split(" (")[0])
        axes[1].plot(ep, [100 * h["val_uar"] for h in r["history"]], color=col, lw=1.8,
                     label=MODEL_TR[m].split(" (")[0])
    axes[0].set_title("Eğitim kaybı (kat 0)", loc="left")
    axes[1].set_title("Doğrulama UAR % (kat 0)", loc="left")
    for a in axes:
        a.set_xlabel("epoch")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_learning_curves.png")
    plt.close(fig)


def fig_data():
    meta = pd.read_csv(DATA / "metadata.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
    order = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised", "boredom"]
    cnt = meta.groupby(["emotion", "corpus"]).size().unstack(fill_value=0).reindex(order).fillna(0)
    x = np.arange(len(order))
    axes[0].bar(x - 0.2, cnt["ravdess"], 0.38, color=S1, label="RAVDESS")
    axes[0].bar(x + 0.2, cnt["emodb"], 0.38, color=S2, label="EmoDB")
    axes[0].set_xticks(x, [TR_NAMES[o] for o in order], rotation=35, ha="right")
    axes[0].set_ylabel("kayıt sayısı")
    axes[0].set_title("Duygu dağılımı", loc="left")
    axes[0].legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    axes[0].set_ylim(0, 235)
    axes[0].grid(axis="x", visible=False)
    for c, col in [("ravdess", S1), ("emodb", S2)]:
        axes[1].hist(meta[meta.corpus == c].dur, bins=30, color=col, alpha=0.6,
                     label={"ravdess": "RAVDESS", "emodb": "EmoDB"}[c])
    axes[1].legend(fontsize=8.5)
    axes[1].set_ylabel("kayıt sayısı")
    axes[1].set_xlabel("süre (sn, sessizlik kırpıldıktan sonra)")
    axes[1].set_title("Süre dağılımı", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_dataset.png")
    plt.close(fig)


def paired(df: pd.DataFrame):
    """Aynı katlarda iki modelin UAR farkı: ortalama fark, kazanılan kat sayısı, Wilcoxon p."""
    from scipy.stats import wilcoxon
    pairs = [("ravdess8_spk", "ravdess", "wavlm_large_frozen_wls", "baseline_bilstm"),
             ("ravdess8_spk", "ravdess", "wavlm_large_ft2", "baseline_bilstm"),
             ("ravdess8_spk", "ravdess", "wavlm_large_ft2", "wavlm_large_frozen_wls"),
             ("ravdess8_spk", "ravdess", "wavlm_large_frozen_wls_aug", "wavlm_large_frozen_wls"),
             ("ravdess8_spk", "ravdess", "baseline_bilstm_aug", "baseline_bilstm"),
             ("ravdess8_spk", "ravdess", "wavlm_large_frozen_wls_aug", "baseline_bilstm_aug"),
             ("ravdess8_spk", "ravdess_snr10", "wavlm_large_frozen_wls_aug", "wavlm_large_frozen_wls"),
             ("ravdess8_spk", "ravdess_snr10", "baseline_bilstm_aug", "baseline_bilstm")]
    lines = ["| Deney | Test | A | B | Ort. UAR farkı A−B (puan) | A'nın kazandığı kat | Wilcoxon p |",
             "|---|---|---|---|---|---|---|"]
    for exp, test, a, b in pairs:
        qa = df[(df.exp == exp) & (df.test == test) & (df.model == a)].set_index("fold").uar
        qb = df[(df.exp == exp) & (df.test == test) & (df.model == b)].set_index("fold").uar
        common = qa.index.intersection(qb.index)
        if len(common) < 3:
            continue
        d = (qa[common] - qb[common]) * 100
        p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
        lines.append(f"| {exp} | {test} | {a} | {b} | {d.mean():+.1f} | {(d > 0).sum()}/{len(d)} | {p:.3f} |")
    (RESULTS / "paired.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def breakdown(runs):
    """Hata analizi: RAVDESS'te şiddet (normal/güçlü) ve cinsiyete göre UAR, sınıf başına duyarlılık."""
    from common import metrics as mtr
    meta = pd.read_csv(DATA / "metadata.csv").set_index("utt_id")
    lines = ["| Model | Alt grup | n | UAR (%) |", "|---|---|---|---|"]
    recall_rows = []
    for m in ["baseline_bilstm", "wavlm_large_frozen_wls", "wavlm_large_ft2"]:
        rs = runs.get(("ravdess8_spk", m))
        if not rs:
            continue
        utt = sum((r["tests"]["ravdess"]["utt_id"] for r in rs), [])
        y = np.array(sum((r["tests"]["ravdess"]["y"] for r in rs), []))
        p = np.array(sum((r["tests"]["ravdess"]["pred"] for r in rs), []))
        sub = meta.loc[utt]
        groups = {"tümü": np.ones(len(y), bool),
                  "normal şiddet": (sub.intensity == "normal").values,
                  "güçlü şiddet": (sub.intensity == "strong").values,
                  "kadın": (sub.gender == "f").values, "erkek": (sub.gender == "m").values}
        for g, mask in groups.items():
            # nötrün güçlü şiddeti yok; şiddet gruplarında UAR mevcut sınıflar üzerinden
            u = mtr(y[mask], p[mask], 8)["uar"] if g not in ("normal şiddet", "güçlü şiddet") else \
                np.mean([np.mean(p[mask & (y == c)] == c) for c in np.unique(y[mask])])
            lines.append(f"| {m} | {g} | {mask.sum()} | {100 * u:.1f} |")
        classes = rs[0]["classes"]
        recall_rows.append({"model": m, **{TR_NAMES[c]: round(100 * np.mean(p[y == i] == i), 1)
                                           for i, c in enumerate(classes)}})
    txt = "\n".join(lines)
    if recall_rows:
        rec = pd.DataFrame(recall_rows).set_index("model")
        txt += "\n\nSınıf başına duyarlılık (%):\n\n" + rec.to_markdown()
    (RESULTS / "breakdown.md").write_text(txt, encoding="utf-8")
    print(txt)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df, runs = load_runs()
    s = summarize(df)
    write_tables(s)
    paired(df)
    breakdown(runs)
    print((RESULTS / "summary.md").read_text())
    fig_data()
    fig_main(s)
    fig_leakage(s)
    fig_cross(s)
    fig_noise(s)
    fig_layers(runs)
    fig_curves(runs)
    for m in ["baseline_bilstm", "wavlm_large_frozen_wls", "wavlm_large_ft2"]:
        fig_confusion(runs, m)
    print("Grafikler:", sorted(p.name for p in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
