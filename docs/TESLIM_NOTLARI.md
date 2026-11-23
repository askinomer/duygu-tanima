# Teslimden önce yapılacaklar

Teslim tarihi 22.12.2026, sunum tarihi 29.12.2026.

1. **İsimler.** Raporun başındaki "Hazırlayanlar" satırına, sunumun kapak slaytına ve YouTube açıklamasına ad, soyad ve numaralar işlendi (Ömer Faruk Aşkın 23181616758, Göktuğ Ercedoğan 21181616014). Videonun kapağında isim yok; isimler eklenmek istenirse yalnız kapak sahnesi yeniden üretilir.
2. **Rapor formatı.** Hoca bir rapor şablonu paylaşacağını söyledi. Şablon gelince `report/Konusmadan_Duygu_Tanima_Rapor.docx` içeriğini o şablona aktarın; bölüm başlıkları zaten istenen beş başlıkla aynı (Giriş, Materyal ve Metot, Deneysel Sonuçlar, Sonuç ve Öneriler, Kaynakça).
3. **Video.** 10 dakikalık video hazır (`Konusmadan_Duygu_Tanima_Video.mp4`, boyutu nedeniyle pakette yok, ayrıca gönderildi). Anlatım yapay sesle yapıldı; okunan metin ve zamanlar `docs/video_senaryosu.md` dosyasında. YouTube'a "Liste dışı" yükleyin, `docs/video/Konusmadan_Duygu_Tanima_Video.srt` dosyasını Türkçe altyazı olarak ekleyin, açıklamaya `docs/video/YouTube_aciklama.txt` metnini yapıştırın ve bağlantıyı hocaya gönderin.
4. **Zip.** Her öğrenci kendi zip'ini `ogrenciNo_ogrenciAdSoyad.zip` adıyla Oğuzhan hocaya teslim eder. Bu paketi açıp klasörün adını kendi numaranız ve adınızla değiştirerek yeniden sıkıştırın.
5. **Kodu kendiniz bir kez çalıştırın.** `notebooks/colab_run_all.ipynb` Colab'de GPU ile tüm hattı baştan çalıştırır (veri setlerini resmi kaynaklarından indirir). Sunumda soru gelirse her adımı anlatabilmeniz için iyi bir hazırlık olur.

## Bu pakette olmayanlar

- **Ham ses verisi:** lisans ve boyut nedeniyle pakette yok. RAVDESS'i Zenodo'dan, EmoDB'yi resmi sitesinden indirin (README).
- **WavLM-Large ağırlıkları (1,2 GB):** pakette yok. İnternet varken `microsoft/wavlm-large` adıyla Hugging Face'ten otomatik iner.
- **Ara özellik dosyaları (yaklaşık 4 GB):** `ssl_extract.py` bunları yeniden üretir.

## Bilinmesi gerekenler

- Projede wav2vec2 yerine WavLM-Large kullanıldı: aynı aileden ve duygu tanımada genelde daha güçlü bir model. Kod `--model facebook/wav2vec2-base` ile wav2vec2 için de çalışır.
- Deneyler GPU'suz bir makinede koşulduğu için fine-tune kısmi yapıldı: yalnız üst 2 katman eğitildi. Sonuç (%71,5) dondurulmuş modelin (%79,2) gerisinde kaldı. Raporda bunun nedenleri açıklanıyor; sunumda soru gelirse katman analizine (Şekil 4) gönderme yapın.
