# Video senaryosu (10:00)

Videonun son hali `Konusmadan_Duygu_Tanima_Video.mp4` (1920×1080, 30 fps, 10:00). YouTube'a "Liste dışı" yüklenip bağlantısı hocaya gönderilecek; yükleme ayarları ve açıklama metni `docs/video/YouTube_aciklama.txt` dosyasında, Türkçe altyazı `docs/video/Konusmadan_Duygu_Tanima_Video.srt` dosyasında.

Anlatım, Piper TTS ve tr_TR-dfki-medium sesiyle üretilmiş yapay sestir (ses modeli DFKI MaryTTS Türkçe verisiyle eğitilmiş, CC BY-NC-SA 4.0). Demodaki ses örnekleri RAVDESS'ten (CC BY-NC-SA 4.0). Aşağıdaki metin videoda okunan metnin aynısıdır; videoyu kendi sesinizle yeniden kaydetmek isterseniz aynı metni ve süreleri kullanabilirsiniz.

Hocanın istediği dört başlık: 1) problem tanımı 0:00, 2) yöntem ve model 1:25, 3) deneysel sonuçlar ve demo 4:02, 4) sonuçlar ve değerlendirme 8:59.

# 1. Problem tanımı

## 0:00 · Giriş

Merhaba, bu videoda Ses İşleme dersi dönem projemizi anlatıyoruz: konuşmadan duygu tanıma. Amacımız, kısa bir ses kaydından konuşanın duygusunu tahmin eden bir sistem geliştirmek ve önceden eğitilmiş büyük bir konuşma modelini klasik bir yöntemle adil koşullarda karşılaştırmaktı. Video dört bölümden oluşuyor: problemin tanımı, yöntem ve modeller, deneysel sonuçlar ile demo ve son olarak değerlendirme.

## 0:25 · Problem tanımı

Öncelikle problemi tanımlayalım. Konuşmadan duygu tanıma, bir ses kaydındaki perde, enerji, konuşma hızı ve ses rengi gibi ipuçlarından konuşanın duygusunu sınıflandırma problemidir. Çağrı merkezleri, sesli asistanlar, eğitim ve sağlık uygulamaları başlıca kullanım alanları. Ancak problem zor: aynı duyguyu herkes farklı seslendirir; dil, kültür ve kayıt koşulları da sesi değiştirir. Üstelik etiketli veri setleri küçüktür ve çoğu oyuncu kayıtlarından oluşur. Son yıllarda, etiketsiz on binlerce saatlik konuşmayla önceden eğitilen modeller bu alanda büyük ilerleme sağladı. Biz bu projede üç soruya cevap aradık. Birincisi: önceden eğitilmiş WavLM modeli, klasik log-Mel yöntemine göre ne kadar kazandırıyor? İkincisi: konuşmacıları eğitim ve test arasında karıştıran rastgele bir bölme, başarıyı ne kadar şişiriyor? Üçüncüsü: bir veri setinde eğitilen model, başka dildeki bir veri setinde ne kadar çalışıyor?

# 2. Yöntem ve model

## 1:25 · Veri setleri

İki herkese açık veri seti kullandık. RAVDESS'te on ikisi kadın, on ikisi erkek 24 oyuncu, iki İngilizce cümleyi 8 duyguyla seslendiriyor; toplam 1.440 kayıt var. EmoDB'de ise 10 Alman oyuncu, 7 duyguda 535 kayıt üretmiş. Tüm kayıtları 16 kilohertz tek kanala çevirdik ve baştaki ve sondaki sessizliği kırptık; kayıtların çoğu 1 ile 3 saniye arasında. Sınıflar dengesiz olabildiği için ana ölçüt olarak ağırlıksız ortalama duyarlılığı, kısaca UAR değerini kullandık.

## 1:58 · Değerlendirme protokolü

Değerlendirmede en çok dikkat ettiğimiz konu konuşmacı ayrımıydı. 24 oyuncuyu dörder kişilik 6 kata böldük; her katta iki kadın ve iki erkek var. Her turda bir kat test, bir kat doğrulama için ayrılıyor; kalan 16 oyuncuyla model eğitiliyor. Böylece model, sesini hiç duymadığı kişiler üzerinde ölçülüyor. Sızıntının etkisini ölçmek için konuşmacıları karıştıran rastgele bir bölme de yaptık. Çapraz veri seti deneyinde ise ortak altı sınıfta bir setle eğitip diğerinde test ettik; iki setle birlikte eğitmeyi de denedik. Sekiz sınıfta UAR için şans düzeyi %12,5.

## 2:37 · Modeller

Üç modeli aynı veri ve aynı katlar üzerinde karşılaştırdık. Birincisi, klasik baseline modeli: log-Mel spektrogram üzerinde iki evrişim katmanı, iki katmanlı çift yönlü LSTM ve dikkat havuzlaması. İkincisi, Microsoft'un WavLM-Large modeli: 94 bin saatlik etiketsiz konuşmayla önceden eğitilmiş, 24 katmanlı bir transformer. WavLM'i değiştirmeden her kayıt için 25 katmanın çıktısını alıyor, zaman ekseninde ortalama ve standart sapmayla özetliyoruz. Ardından katmanların önemini öğrenen bir ağırlıklı toplam ve küçük bir sınıflandırıcı eğitiyoruz. Üçüncüsü kısmi ince ayar, yani fine-tune: ilk 22 katman dondurulmuş kalıyor; son iki katman sınıflandırıcıyla birlikte eğitiliyor. Şemada mavi kutular dondurulmuş, turuncu kutular eğitilen kısımlar.

## 3:29 · GPU olmadan hesaplama

Projeyi GPU olmadan, yalnızca iki işlemci çekirdeğiyle yürüttük; bu yüzden hesabı akıllıca bölmemiz gerekti. WavLM'in dondurulmuş kısmının çıktısını her kayıt için bir kez hesaplayıp sakladık; bu, kayıt başına yaklaşık 0,7 saniye sürüyor. Böylece sınıflandırıcının eğitimi kat başına yarım dakikaya indi; baseline ise kat başına 5 ile 18 dakikada eğitildi. Kısmi fine-tune için de alt 22 katmanın çıktısını önbelleğe aldık ve bu kısayolun tam modelle aynı sonucu verdiğini sayısal olarak doğruladık.

# 3. Deneysel sonuçlar ve demo

## 4:02 · Ana sonuçlar

Şimdi ana sonuçlara geçelim. Konuşmacıdan bağımsız testte baseline %43,5 UAR elde ediyor. Dondurulmuş WavLM ise %79,2'ye ulaşıyor. Aradaki 35,7 puanlık fark, altı katın altısında da aynı yönde. Üst iki katmanı fine-tune ettiğimiz model %71,5'te kaldı: baseline'dan 28 puan iyi, ama dondurulmuş modelin 7,7 puan gerisinde. Nedenini bir sonraki grafikte göreceğiz: duygu bilgisi orta katmanlarda; biz ise yalnızca en üstteki iki katmanı, sınırlı bir bütçeyle 10 epoch eğitebildik. Veri artırmayla eğitilen baseline da %53'e çıkıyor, ama WavLM'in hâlâ 26 puan gerisinde.

## 4:47 · Katman analizi

Peki duygu bilgisi WavLM'in hangi katmanlarında? Bunu görmek için her katmanın çıktısı üzerinde ayrı bir doğrusal sınıflandırıcı eğittik. Başarı dokuzuncu katmanda %81 ile en yüksek değerine ulaşıyor, son katmanda ise %74'e iniyor. Ağırlıklı toplamın öğrendiği ağırlıklar da orta katmanları hafifçe öne çıkarıyor. Bu bulgu, yalnızca en üst katmanları eğiten kısmi fine-tune modelinin neden geride kaldığını açıklıyor.

## 5:16 · Konuşmacı sızıntısı

İkinci sorumuz konuşmacı sızıntısıydı. Konuşmacıları karıştıran rastgele bölmede başarı, baseline'da 26,4, WavLM'de 12,4 puan yükseliyor. Çünkü test edilen kişinin başka kayıtları eğitimde de bulunuyor ve model, duygunun yanında kişiyi de tanımaya başlıyor. Bu yüzden literatürde RAVDESS için bildirilen %90'ın üzerindeki sonuçlarda hangi bölmenin kullanıldığına mutlaka bakmak gerekiyor.

## 5:43 · Veri setleri arası genelleme

Üçüncü sorumuz ise genelleme yeteneği. Ortak altı sınıfta RAVDESS ile eğitilen WavLM, Almanca EmoDB'de %58,4'e düşüyor; bu, EmoDB'nin kendi verisiyle eğitilen modelin 33 puan altında. Ters yönde de 26 puanlık bir düşüş var. Baseline ise RAVDESS'ten EmoDB'ye geçerken tamamen çöküyor ve şans düzeyinde kalıyor. İki veri setiyle birlikte eğittiğimizde ise model, RAVDESS'te %82,7, EmoDB'de %95,1 ile ikisinde de en iyi sonucunu veriyor. Yani genelleme için eğitim verisinin çeşitliliği çok belirleyici.

## 6:22 · Gürültüye dayanıklılık

Gerçek hayatta kayıtlar temiz değildir; bu yüzden test kayıtlarına pembe gürültü ekledik. Sinyal gürültü oranı 10 dB olduğunda baseline %17,3'e, yani neredeyse şans düzeyine iniyor; WavLM ise %61,9'da kalıyor. Eğitimde gürültü, hız ve ses düzeyi değişimleriyle veri artırma yaptığımızda WavLM, 10 dB'de %74,5'e çıkıyor. Demo modelimizi de bu yüzden artırılmış veriyle eğittik.

## 6:51 · Hata analizi

Hatalara baktığımızda en zor sınıf üzüntü. Üzgün kayıtların %64,6'sı doğru tanınıyor; %10'u korkmuş, %9'u sakin, %8'i nötr sanılıyor. Sakin, nötr ve üzgün; düşük enerjili, birbirine yakın duygular. En kolay sınıf ise %88 ile öfke. Güçlü şiddetteki kayıtlar, normal şiddettekilerden 7,7 puan daha kolay tanınıyor. Kadın oyuncularda başarı %86,1, erkeklerde %72,4; her cinsiyette yalnızca 12 oyuncu olduğu için bu farkı temkinli yorumluyoruz.

## 7:28 · Demo

Şimdi demoya geçelim. Gradio ile hazırladığımız arayüz, mikrofondan kayıt alabiliyor ya da bir ses dosyası kabul ediyor. Demo modeli, iki veri setinin ortak altı sınıfında veri artırmayla eğitilmiş dondurulmuş WavLM.

*7:42 · Demo arayüzü açılır.*

Deneyeceğimiz kayıtlar, modelin eğitiminde kullanılmayan RAVDESS oyuncularına ait.

*7:48 · Kayıt 1: RAVDESS oyuncu 22 (kadın), mutlu*

İlk kayıtta bir kadın oyuncu, İngilizce bir cümleyi mutlu bir tonla söylüyor. [kayıt çalınır]

Model, kaydı %99 olasılıkla mutlu olarak sınıflandırıyor.

*8:03 · Kayıt 2: aynı oyuncu, öfkeli*

Aynı oyuncu, aynı cümleyi bu kez öfkeyle söylüyor. [kayıt çalınır]

Tahmin yine doğru: %99 ile öfkeli.

*8:16 · Kayıt 3: aynı oyuncu, üzgün*

Şimdi de üzgün bir tonla. [kayıt çalınır]

Model %82 ile üzgün diyor; ikinci sırada %15 ile korkmuş var. Bu, hata analizinde gördüğümüz üzüntü ve korku karışmasının küçük bir örneği.

*8:34 · Kayıt 4: RAVDESS oyuncu 21 (erkek), üzgün*

Son olarak bir erkek oyuncunun aynı cümleyi üzgün seslendirdiği kayıt. [kayıt çalınır]

Bu kez model yanılıyor: %80 ile mutlu diyor; doğru cevap olan üzgün %20'de kalıyor. Bu örnek, hata analizindeki iki zayıflığı birlikte gösteriyor: üzüntü en zor sınıf ve erkek oyuncularda başarı daha düşük.

# 4. Sonuçlar ve değerlendirme

## 8:59 · Sonuç ve değerlendirme

Sonuç olarak üç sorumuzun cevabı şöyle. Birincisi: önceden eğitilmiş WavLM klasik yöntemi 35,7 puan geçiyor ve dondurulmuş haliyle GPU olmadan bile kullanılabiliyor. İkincisi: konuşmacıları karıştıran bölmeler sonuçları 12 ile 26 puan arasında yapay olarak şişiriyor; bu yüzden konuşmacıdan bağımsız değerlendirme şart. Üçüncüsü: veri setleri arasında 26 ile 33 puanlık bir düşüş var; iki veri setiyle birlikte eğitmek bu açığı kapatıyor. Sınırlılıklarımız: veriler oyuncu kayıtlarından oluşuyor, Türkçe veri kullanmadık ve yalnızca kısmi fine-tune yapabildik. Gelecekte tam fine-tune denemek, Türkçe ve doğal konuşma verileriyle çalışmak ve emotion2vec gibi yeni modellerle karşılaştırma yapmak istiyoruz. Kodlar, sonuçlar ve rapor proje paketinde yer alıyor. Dinlediğiniz için teşekkür ederiz.

## 9:52 · Kaynaklar ve lisanslar

Videodaki seslendirme, açık kaynaklı Piper sistemiyle üretilmiş yapay bir sestir; kullanılan veri setleri ve lisanslar ekranda.
