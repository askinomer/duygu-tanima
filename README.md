# Konuşmadan Duygu Tanıma (Speech Emotion Recognition)

Ses İşleme dersi dönem projesi, 2026-2027 (Konu 8). Kısa bir konuşma kaydına bakarak
konuşmacının duygusunu tahmin eden bir sistem. Önceden eğitilmiş, kendi kendine öğrenen
(self-supervised) bir konuşma modeli olan **WavLM-Large**'ı, klasik **log-Mel + BiLSTM**
baseline'ı ile karşılaştırıyoruz. Deneyler İngilizce **RAVDESS** ve Almanca **EmoDB**
üzerinde, konuşmacıdan bağımsız çapraz doğrulama ile yapıldı.

Sonuçlar `results/summary.md` dosyasında, grafikler `report/figures/` klasöründe.

## Özet sonuçlar

RAVDESS, 8 sınıf, konuşmacıdan bağımsız 6 kat (UAR %, ortalama ± std; şans %12,5):

| Model | Temiz | 10 dB gürültü | 0 dB gürültü |
| --- | --- | --- | --- |
| Baseline (log-Mel + CNN-BiLSTM) | 43,5 ± 4,8 | 17,3 ± 3,5 | 13,7 ± 1,0 |
| Baseline + veri artırma | 53,0 ± 8,2 | 47,9 ± 8,0 | 36,1 ± 6,3 |
| WavLM-Large dondurulmuş + ağırlıklı katman toplamı | 79,2 ± 6,2 | 61,9 ± 5,2 | 27,9 ± 3,9 |
| WavLM-Large dondurulmuş + veri artırma | 79,7 ± 6,6 | 74,5 ± 8,0 | 42,6 ± 15,5 |
| WavLM-Large kısmi fine-tune (üst 2 katman) | 71,5 ± 5,5 | - | - |

- Rastgele (konuşmacı karışık) bölme UAR'ı baseline'da 26,4, WavLM'de 12,4 puan şişiriyor.
- Çapraz veri seti (ortak 6 sınıf, WavLM): RAVDESS→EmoDB %58,4, EmoDB→RAVDESS %55,8; iki veri setiyle birlikte eğitim RAVDESS %82,7, EmoDB %95,1.
- Duygu bilgisi WavLM'in orta katmanlarında yoğun (en iyi tek katman: 9).

## Araştırma soruları

1. Önceden eğitilmiş WavLM, log-Mel + BiLSTM baseline'ına göre ne kadar kazandırıyor? (E1, E2)
2. Konuşmacıları karıştıran rastgele bölme başarıyı ne kadar şişiriyor? (E3)
3. Bir veri setinde eğitilen model diğerinde (başka dil, başka kayıt koşulu) ne kadar çalışıyor? (E4, E5)
4. Veri artırma gürültülü sese dayanıklılığı artırıyor mu? (E6)

## Klasör yapısı

```text
src/
  common.py            ortak sabitler: etiketler, katlar, metrikler
  prepare_data.py      16 kHz mono, sessizlik kırpma, data/metadata.csv
  make_splits.py       splits/splits.csv (herkesin ortak bölme dosyası)
  experiments.py       deney tanımları (eğitim/doğrulama/test indeksleri)
  augment.py           gürültü ekleme, hız değişimi
  baseline_lstm.py     E1, E3, E4/E5, E6: log-Mel + BiLSTM
  convert_wavlm.py     orijinal WavLM ağırlığını Hugging Face biçimine çevirir (internetsiz ortam için)
  ssl_extract.py       WavLM katman özelliklerini çıkarır ve önbelleğe alır
  ssl_heads.py         E2 (dondurulmuş), E3, E4/E5, E6 ve katman analizi
  finetune_ssl.py      E2: üst katmanların kısmi fine-tune'u
  analyze_results.py   tablolar ve grafikler
  train_demo_model.py  demo modeli (RAVDESS + EmoDB, 6 sınıf)
  app_gradio.py        mikrofon/dosya -> duygu demosu
splits/splits.csv      konuşmacı katları ve rastgele katlar
results/               her deney ve kat için JSON (metrikler, tahminler, eğitim geçmişi)
report/figures/        rapor grafikleri
run_all.sh             tüm hattı sırayla çalıştırır
```

## Kurulum

```bash
pip install -r requirements.txt
```

GPU şart değil. Bu projedeki tüm deneyler 2 çekirdekli, GPU'suz bir makinede koşuldu
(kısmi fine-tune ve özellik önbelleği sayesinde). Colab veya Kaggle'da GPU varsa aynı komutlar
daha hızlı çalışır.

## Veri

| Veri seti | Resmi kaynak | Lisans |
| --- | --- | --- |
| RAVDESS (yalnız ses, konuşma: `Audio_Speech_Actors_01-24.zip`) | <https://zenodo.org/records/1188976> | CC BY-NC-SA 4.0 |
| EmoDB | <http://emodb.bilderbar.info/> | Araştırma için serbest |

Zip dosyalarını `data/raw/ravdess/` ve `data/raw/emodb/` altına açın. Betikler wav dosyalarını
alt klasörlerde de bulur. Kontrol için beklenen sayılar: RAVDESS 1.440 kayıt (24 oyuncu, 8 duygu),
EmoDB 535 kayıt (10 konuşmacı, 7 duygu).

## Çalıştırma

```bash
bash run_all.sh                       # her şey (internet varsa MODEL=microsoft/wavlm-large)
```

veya adım adım (bkz. `run_all.sh`). wav2vec2 ile denemek için:

```bash
python src/ssl_extract.py --model facebook/wav2vec2-base --tag w2v2_base --variants clean
python src/ssl_heads.py --tag w2v2_base --exp ravdess8_spk
```

## Değerlendirme protokolü

- **Konuşmacıdan bağımsız 6 kat (RAVDESS):** her kat 4 oyuncu (2 kadın, 2 erkek). Kat *k* test,
  kat *k+1* doğrulama (erken durdurma), kalan 16 oyuncu eğitim. Her oyuncu tam bir kez test edilir.
- **Rastgele 6 kat (E3):** aynı boyutlar, ama duyguya göre tabakalı ve konuşmacıyı yok sayan bölme.
- **Çapraz veri seti (E4/E5):** ortak 6 sınıf (nötr, mutlu, üzgün, öfkeli, korkmuş, tiksinmiş).
  EmoDB 5 kat (her katta 1 kadın + 1 erkek), RAVDESS katlarıyla eşleştirilir.
- **Metrikler:** accuracy, UAR (sınıf başına duyarlılığın ortalaması; dengesiz sınıflar için asıl
  metrik) ve macro-F1. Kat ortalaması ± standart sapma raporlanır.

## Demo

```bash
python src/train_demo_model.py
python src/app_gradio.py --backbone microsoft/wavlm-large   # veya pretrained/wavlm-large-hf
```

## Kaynaklar

Kaynakça raporda. Temel çalışmalar: WavLM (Chen vd., IEEE JSTSP 2022), RAVDESS (Livingstone ve
Russo, PLoS ONE 2018), EmoDB (Burkhardt vd., Interspeech 2005; EmoDB 2.0, Interspeech 2025),
kısmi fine-tune (Sampath vd., arXiv 2025), SUPERB ağırlıklı katman toplamı (Yang vd., Interspeech 2021).
