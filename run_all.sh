#!/usr/bin/env bash
# Tüm deney hattını baştan sona çalıştırır. GPU'lu bir makinede (Colab/Kaggle) birkaç saat,
# yalnız CPU'da (2 çekirdek) yaklaşık 6-7 saat sürer. Adımlar kesintiye dayanıklıdır.
set -euo pipefail
cd "$(dirname "$0")"

# 0) Veri: RAVDESS (Audio_Speech_Actors_01-24.zip) ve EmoDB wav dosyalarını
#    data/raw/ravdess ve data/raw/emodb altına açın (bkz. README).
python src/prepare_data.py --ravdess_dir data/raw/ravdess --emodb_dir data/raw/emodb
python src/make_splits.py

# 1) Önceden eğitilmiş model. İnternet varsa doğrudan Hugging Face adı kullanılabilir:
MODEL=${MODEL:-microsoft/wavlm-large}
#    İnternetsiz ortamda: python src/convert_wavlm.py && MODEL=pretrained/wavlm-large-hf

# 2) SSL özellik çıkarımı (temiz + gürültülü test + 2 artırılmış kopya, üst katman önbelleği)
python src/ssl_extract.py --model "$MODEL" --tag wavlm_large \
    --variants clean snr10 snr0 aug1 --cache_layers 22

# 3) Baseline (log-Mel + BiLSTM)
python src/baseline_lstm.py --exp ravdess8_spk --save_model       # E1 (+ gürültülü test)
python src/baseline_lstm.py --exp ravdess8_rnd                    # E3
python src/baseline_lstm.py --exp ravdess8_spk --augment          # E6
for s in R E RE; do python src/baseline_lstm.py --exp cross6_$s; done   # E4/E5

# 4) Dondurulmuş WavLM + ağırlıklı katman toplamı
python src/ssl_heads.py --exp ravdess8_spk --mode probe           # katman analizi
python src/ssl_heads.py --exp ravdess8_spk                        # E2 (dondurulmuş)
python src/ssl_heads.py --exp ravdess8_rnd                        # E3
python src/ssl_heads.py --exp ravdess8_spk --augment              # E6
for s in R E RE; do python src/ssl_heads.py --exp cross6_$s; done     # E4/E5

# 5) Kısmi fine-tune (üst 2 katman)                                 # E2 (fine-tune)
python src/finetune_ssl.py --exp ravdess8_spk --model "$MODEL" --cache_layer 22 --n_layers 2 --epochs 10 --patience 3

# 6) Tablolar, grafikler, demo modeli
python src/analyze_results.py
python src/train_demo_model.py
# python src/app_gradio.py --backbone "$MODEL"
