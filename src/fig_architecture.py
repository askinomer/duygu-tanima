"""Şekil: üç modelin mimari şeması (report/figures/fig_architecture.png)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import FIGURES

INK, INK2, AXIS = "#0b0b0b", "#52514e", "#c3c2b7"
FROZEN, TRAIN = "#cde2fb", "#fbd9c9"      # mavi = dondurulmuş, turuncu = eğitilen
FROZEN_E, TRAIN_E = "#2a78d6", "#eb6834"
NEUTRAL = "#f0efec"


def box(ax, x, y, w, h, text, kind="neutral", size=8.6, bold=False):
    fc, ec = {"neutral": (NEUTRAL, AXIS), "frozen": (FROZEN, FROZEN_E), "train": (TRAIN, TRAIN_E)}[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 color=INK2, lw=1.0, shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    box(ax, 0.1, 2.35, 1.25, 0.9, "Ses kaydı\n16 kHz, mono,\nsessizlik kırpılmış")
    # --- Baseline satırı
    ax.text(1.75, 5.35, "(a) Baseline", fontsize=9.5, fontweight="bold", color=INK)
    box(ax, 1.75, 4.35, 1.5, 0.75, "log-Mel\n64 bant, 10 ms", "neutral")
    box(ax, 3.6, 4.35, 1.6, 0.75, "2 × Conv1D\n(zaman / 4)", "train")
    box(ax, 5.55, 4.35, 1.6, 0.75, "2 × BiLSTM\n128 birim/yön", "train")
    box(ax, 7.5, 4.35, 1.5, 0.75, "dikkat\nhavuzlaması", "train")
    box(ax, 9.4, 4.35, 1.45, 0.75, "duygu\n(8 veya 6 sınıf)", "train")
    for x0, x1 in [(3.25, 3.6), (5.2, 5.55), (7.15, 7.5), (9.0, 9.4)]:
        arrow(ax, x0, 4.725, x1, 4.725)
    arrow(ax, 1.35, 3.0, 1.75, 4.5)

    # --- WavLM gövdesi
    ax.text(1.75, 3.62, "(b) WavLM-Large gövdesi", fontsize=9.5, fontweight="bold", color=INK)
    box(ax, 1.75, 2.45, 1.5, 0.95, "CNN özellik\nçıkarıcı\n(7 katman)", "frozen")
    box(ax, 3.6, 2.45, 2.0, 0.95, "Transformer\nkatman 1 … 24\n(1024 boyut)", "frozen")
    arrow(ax, 1.35, 2.8, 1.75, 2.9)
    arrow(ax, 3.25, 2.925, 3.6, 2.925)

    # (b1) dondurulmuş + ağırlıklı katman toplamı
    ax.text(5.95, 3.62, "(b1) Dondurulmuş + ağırlıklı katman toplamı", fontsize=9.5,
            fontweight="bold", color=INK)
    box(ax, 5.95, 2.45, 1.55, 0.95, "25 gizli durum\nort. + std\nhavuzlama", "frozen")
    box(ax, 7.8, 2.45, 1.3, 0.95, "öğrenilen\nkatman\nağırlıkları", "train")
    box(ax, 9.4, 2.45, 1.45, 0.95, "MLP\n2048→256→C", "train")
    arrow(ax, 5.6, 2.925, 5.95, 2.925)
    arrow(ax, 7.5, 2.925, 7.8, 2.925)
    arrow(ax, 9.1, 2.925, 9.4, 2.925)

    # (b2) kısmi fine-tune
    ax.text(1.75, 1.62, "(b2) Kısmi fine-tune", fontsize=9.5, fontweight="bold", color=INK)
    box(ax, 3.6, 0.45, 2.0, 0.95, "katman 22 çıktısı\n(bir kez hesaplanıp\nönbelleğe alınır)", "frozen")
    box(ax, 5.95, 0.45, 1.55, 0.95, "katman 23-24\n(ön-eğitilmiş\nağırlıklardan başlar)", "train")
    box(ax, 7.8, 0.45, 1.3, 0.95, "LayerNorm\n→ 256 →\nzaman ort.", "train")
    box(ax, 9.4, 0.45, 1.45, 0.95, "duygu\n(C sınıf)", "train")
    arrow(ax, 4.6, 2.45, 4.6, 1.4)
    arrow(ax, 5.6, 0.925, 5.95, 0.925)
    arrow(ax, 7.5, 0.925, 7.8, 0.925)
    arrow(ax, 9.1, 0.925, 9.4, 0.925)

    # açıklama
    ax.add_patch(FancyBboxPatch((0.1, 0.55), 0.3, 0.22, boxstyle="round,pad=0.01", fc=FROZEN, ec=FROZEN_E))
    ax.text(0.5, 0.66, "dondurulmuş", va="center", fontsize=8.5, color=INK2)
    ax.add_patch(FancyBboxPatch((0.1, 0.15), 0.3, 0.22, boxstyle="round,pad=0.01", fc=TRAIN, ec=TRAIN_E))
    ax.text(0.5, 0.26, "eğitilen", va="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_architecture.png", dpi=200)
    print("Kaydedildi:", FIGURES / "fig_architecture.png")


if __name__ == "__main__":
    main()
