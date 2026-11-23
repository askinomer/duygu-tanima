"""Gradio demosu: mikrofon veya dosya -> duygu olasılıkları.

Demo modeli: dondurulmuş WavLM-Large + ağırlıklı katman toplamı + MLP,
RAVDESS + EmoDB ortak 6 sınıf üzerinde eğitilmiş (bkz. train_demo_model.py).

Kullanım:
    python src/app_gradio.py                       # http://127.0.0.1:7860
    python src/app_gradio.py --share               # Colab'de herkese açık geçici link
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr  # noqa: E402
import librosa
import numpy as np
import torch

from common import PRETRAINED, ROOT, SR, TR_NAMES
from ssl_extract import forward_all, load_backbone
from ssl_heads import WLSHead

DEFAULT_HEAD = ROOT / "models" / "demo_wavlm_wls_common6.pt"


class EmotionRecognizer:
    def __init__(self, backbone: str, head_path):
        ck = torch.load(head_path, map_location="cpu", weights_only=False)  # kendi ürettiğimiz dosya
        self.classes = ck["classes"]
        self.mu, self.sd = ck["mu"], ck["sd"]
        self.head = WLSHead(self.mu.shape[1], self.mu.shape[2], len(self.classes))
        self.head.load_state_dict(ck["state"])
        self.head.eval()
        self.backbone = load_backbone(backbone)

    def __call__(self, y: np.ndarray, sr: int) -> dict:
        y = y.astype(np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if np.abs(y).max() > 1.0:  # Gradio int16 döndürebilir
            y = y / 32768.0
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        yt, _ = librosa.effects.trim(y, top_db=30)
        if len(yt) >= SR // 2:
            y = yt
        pooled, _ = forward_all(self.backbone, y, [])
        z = (pooled[None] - self.mu) / self.sd
        with torch.no_grad():
            p = torch.softmax(self.head(torch.from_numpy(z.astype(np.float32))), -1)[0].numpy()
        return {TR_NAMES[c]: float(v) for c, v in zip(self.classes, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=str(PRETRAINED / "wavlm-large-hf"),
                    help="yerel klasör ya da 'microsoft/wavlm-large'")
    ap.add_argument("--head", default=str(DEFAULT_HEAD))
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    rec = EmotionRecognizer(args.backbone, args.head)

    def predict(audio):
        if audio is None:
            return {}
        sr, y = audio
        return rec(y, sr)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Audio(sources=["microphone", "upload"], type="numpy",
                        label="Bir cümle söyleyin veya ses dosyası yükleyin"),
        outputs=gr.Label(num_top_classes=6, label="Tahmin edilen duygu"),
        title="Konuşmadan Duygu Tanıma",
        description=("WavLM-Large (dondurulmuş) + ağırlıklı katman toplamı. RAVDESS (İngilizce) ve "
                     "EmoDB (Almanca) oyuncu kayıtlarıyla eğitildi; 6 sınıf: nötr, mutlu, üzgün, "
                     "öfkeli, korkmuş, tiksinmiş. Doğal konuşmada ve Türkçede başarı düşebilir."),
        flagging_mode="never",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
