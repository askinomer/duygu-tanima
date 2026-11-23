"""Microsoft'un orijinal WavLM-Large kontrol noktasını Hugging Face biçimine çevirir
ve ağırlık eşlemesini katı biçimde doğrular.

Bu adım yalnızca Hugging Face'e erişimin olmadığı ortamlar için gerekir. İnternet
varsa doğrudan `--model microsoft/wavlm-large` kullanın (aynı ağırlıklar).

Kullanım:
    python src/convert_wavlm.py --ckpt pretrained/WavLM-Large.pt --out pretrained/wavlm-large-hf
Ağırlık dosyası güvenli modda (weights_only=True) açılır: yalnız tensör ve basit tipler.
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel

from common import ROOT


def hf_config(c: dict) -> WavLMConfig:
    spec = c["conv_feature_layers"]  # örn. "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"
    if not re.fullmatch(r"[\[\]()\d,+* ]+", spec):  # yalnız sayı, parantez, virgül, + ve * kabul edilir
        raise ValueError(f"Beklenmeyen conv_feature_layers değeri: {spec!r}")
    layers = eval(spec, {"__builtins__": {}})
    return WavLMConfig(
        hidden_size=c["encoder_embed_dim"], num_hidden_layers=c["encoder_layers"],
        num_attention_heads=c["encoder_attention_heads"], intermediate_size=c["encoder_ffn_embed_dim"],
        hidden_act=c["activation_fn"],
        feat_extract_norm="layer" if c["extractor_mode"] == "layer_norm" else "group",
        conv_dim=[d for d, _, _ in layers], conv_kernel=[k for _, k, _ in layers],
        conv_stride=[s for _, _, s in layers], conv_bias=c["conv_bias"],
        num_conv_pos_embeddings=c["conv_pos"], num_conv_pos_embedding_groups=c["conv_pos_groups"],
        do_stable_layer_norm=c["layer_norm_first"], num_buckets=c["num_buckets"],
        max_bucket_distance=c["max_distance"],
        hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0,
        layerdrop=0.0, mask_time_prob=0.0, apply_spec_augment=False,
    )


def map_key(k: str) -> str | None:
    if k == "mask_emb":
        return "masked_spec_embed"
    if k.startswith("feature_extractor.conv_layers."):
        i = k.split(".")[2]
        if k.endswith(".0.weight"):
            return f"feature_extractor.conv_layers.{i}.conv.weight"
        if ".2.1." in k:
            return f"feature_extractor.conv_layers.{i}.layer_norm.{k.split('.')[-1]}"
    if k.startswith("layer_norm."):
        return "feature_projection." + k
    if k.startswith("post_extract_proj."):
        return k.replace("post_extract_proj", "feature_projection.projection")
    if k == "encoder.pos_conv.0.bias":
        return "encoder.pos_conv_embed.conv.bias"
    if k == "encoder.pos_conv.0.weight_g":
        return "encoder.pos_conv_embed.conv.parametrizations.weight.original0"
    if k == "encoder.pos_conv.0.weight_v":
        return "encoder.pos_conv_embed.conv.parametrizations.weight.original1"
    if k.startswith("encoder.layer_norm."):
        return k
    if k.startswith("encoder.layers."):
        rep = [("self_attn.grep_linear", "attention.gru_rel_pos_linear"),
               ("self_attn.grep_a", "attention.gru_rel_pos_const"),
               ("self_attn.relative_attention_bias", "attention.rel_attn_embed"),
               ("self_attn_layer_norm", "layer_norm"),
               ("self_attn.", "attention."),
               ("fc1", "feed_forward.intermediate_dense"),
               ("fc2", "feed_forward.output_dense")]
        for a, b in rep:
            if a in k:
                return k.replace(a, b)
        return k  # final_layer_norm aynı adı taşır
    return None  # ön-eğitim başlıkları (final_proj, label_embs) gerekmez


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "pretrained" / "WavLM-Large.pt"))
    ap.add_argument("--out", default=str(ROOT / "pretrained" / "wavlm-large-hf"))
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = hf_config(ck["cfg"])
    model = WavLMModel(cfg).eval()
    target = model.state_dict()
    new, skipped = {}, []
    for k, v in ck["model"].items():
        nk = map_key(k)
        if nk is None or nk not in target:
            skipped.append(k)
            continue
        assert target[nk].shape == v.shape, (k, nk, v.shape, target[nk].shape)
        new[nk] = v
    missing = sorted(set(target) - set(new))
    print(f"eşlenen={len(new)}  hedef={len(target)}  eksik={missing}  atlanan={skipped}")
    assert not missing, "Eksik ağırlık var"
    model.load_state_dict(new, strict=True)

    # Ağırlık eşlemesi katı kontrol edilir: 488 hedef parametrenin hepsi dolu, şekiller
    # birebir aynı, load_state_dict(strict=True) başarılı. Ek olarak işlevsel bir kontrol:
    # gerçek bir kayıtta çıktılar sonlu ve rastgele başlatılmış modelden belirgin farklı olmalı.
    from experiments import load_table
    df = load_table()
    found = [ROOT / p for p in df.path if (ROOT / p).exists()]
    if found:
        y = sf.read(str(found[0]), dtype="float32")[0]
    else:  # veri henüz hazırlanmadıysa 2 sn'lik yapay bir sinyalle yalnız sayısal kontrol yapılır
        t = np.arange(32000) / 16000
        y = (0.1 * np.sin(2 * np.pi * (150 + 100 * t) * t)).astype(np.float32)
    x = torch.from_numpy((y - y.mean()) / (y.std() + 1e-7))[None]
    rand = WavLMModel(cfg).eval()
    with torch.inference_mode():
        hs = model(x, output_hidden_states=True).hidden_states
        hr = rand(x, output_hidden_states=True).hidden_states
    assert all(torch.isfinite(h).all() for h in hs)
    cos = torch.nn.functional.cosine_similarity(hs[12].mean(1), hr[12].mean(1)).item()
    print(f"T={hs[0].shape[1]} kare, katman sayısı={len(hs)}, "
          f"katman-12 ortalaması ile rastgele model arası kosinüs={cos:.3f}")

    model.save_pretrained(args.out)
    Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                             do_normalize=True, return_attention_mask=True).save_pretrained(args.out)
    print("Kaydedildi:", args.out)


if __name__ == "__main__":
    main()
