# Yarışma Bilgilendirme

Bu doküman, DataLeague final datathon'u için yarışma kapsamını, veri seti bilgisini, beklenen çıktıları, teknik kısıtları ve pratik çalışma çerçevesini tek yerde toplar.

---

## 1. Yarışmanın Konusu

Görev adı:

```text
Sosyal Medya Manipülasyonu ve Anomali Tespit Sistemi
```

Temel problem:
- global sosyal medya verisi üzerinde çalışılacak
- içeriklerin hangilerinin organik, hangilerinin organize / manipülatif olabileceği tespit edilmeye çalışılacak
- sisteme yeni gelen veriler için hızlı tahmin üretilebilecek bir yapı kurulacak

Bu yarışma klasik denetimli sınıflandırma problemi değildir.

Sebep:
- veri setinde doğrudan `bot`, `human`, `manipulative`, `organic` gibi bir hedef sütun yok
- bu nedenle problem esas olarak `unsupervised anomaly detection` problemidir

---

## 2. Beklenen Ana Çıktılar

Yarışma tanımına göre ekipten beklenen 3 ana çıktı vardır.

### 2.1 Güvenilirlik / Organiklik Skoru

Her bir içerik veya `author_hash` için `0` ile `1` arasında bir skor üretilmesi beklenir.

Bu skor:
- içerik ne kadar organik görünüyor
- ya da ne kadar manipülasyon riski taşıyor

sorusuna cevap verir.

Pratik öneri:
- skoru `final_risk` veya `organicity_score` olarak tanımlamak daha savunulabilir olur
- çünkü doğrudan etiketli gerçek bot verisi yok

### 2.2 Manipülasyon Haritası / Dashboard

Manipülatif olduğu düşünülen içerik kümelerinin:
- hangi dillerde yoğunlaştığı
- hangi platformlarda yoğunlaştığı
- zaman içinde nasıl yayıldığı

gösterilmelidir.

Bu çıktı:
- grafik
- dashboard
- notebook içi görselleştirme

olarak hazırlanabilir.

### 2.3 Canlı Tahmin Modeli / Inference Pipeline

Sunum sırasında jüri, ekibe yeni ve gizli metinler verebilir.

Beklenti:
- bu metinleri anlık işleyen bir fonksiyon veya pipeline bulunması
- sistemin `Organik / Manipülatif` benzeri bir karar vermesi
- kararın nedenini açıklayabilmesi

Buradaki kritik nokta:
- sadece skor vermek değil
- açıklanabilir bir karar sunmak

---

## 3. Veri Seti Bilgisi

Veri dosyası:

```text
datathonFINAL.parquet
```

Format:

```text
Apache Parquet
```

Yaklaşık boyut:

```text
~1 GB
```

Yaklaşık satır sayısı:

```text
~5.000.000
```

Bu repo içindeki mevcut veri üzerinde yapılan doğrulamaya göre toplam satır sayısı:

```text
5,004,813
```

---

## 4. Veri Seti Kolonları

Yarışma dökümanında verilen sütunlar:

### 4.1 `original_text`
- sosyal medya gönderisinin orijinal ham metni

### 4.2 `english_keywords`
- metinden çıkarılmış İngilizce anahtar kelimeler

### 4.3 `sentiment`
- duygu skoru
- genel tanım:

```text
-1.0 = negatif
 0.0 = nötr
 1.0 = pozitif
```

### 4.4 `main_emotion`
- metne hakim ana duygu
- örnek: `joy`, `anger`, `neutral`

### 4.5 `primary_theme`
- gönderinin ana konusu / teması

### 4.6 `language`
- içeriğin dili
- ISO dil kodu formatında olabilir
- örnek: `tr`, `en`, `es`, `ja`

### 4.7 `url`
- verinin geldiği platform ya da kaynak alan adı
- örnek: `x.com`, `reddit.com`

### 4.8 `author_hash`
- anonimleştirilmiş kullanıcı kimliği
- bazı satırlarda boş olabilir

### 4.9 `date`
- gönderinin UTC tarih-saat bilgisi

---

## 5. Teknik Olarak Kritik Uyarılar

Yarışma açıklamasındaki en önemli teknik notlar şunlardır:

### 5.1 Hedef sütun yok

Bu veri setinde doğrudan eğitim etiketi yoktur.

Bu şu anlama gelir:
- klasik supervised classification doğrudan uygulanamaz
- modelin neyin bot neyin insan olduğunu "öğrenmesi" doğrudan mümkün değildir
- bunun yerine davranışsal ve semantik anomalilerden risk skoru üretilmelidir

### 5.2 RAM yönetimi kritik

`5M+` satır doğrudan belleğe alınırsa geliştirme aşamasında gereksiz yavaşlık ve OOM riski oluşur.

Pratik çıkarım:
- geliştirme aşamasında örneklem ile çalışmak mantıklıdır
- final pipeline tüm veri üzerinde çalıştırılmalıdır
- `Pandas` yerine `Polars` veya `DuckDB` kullanmak teknik olarak daha uygundur

### 5.3 Veri kirli olabilir

Veri sentetik değil, gerçek dünya verisidir.

Bu nedenle:
- eksik değerler olabilir
- argo ve bozuk yazımlar olabilir
- tekrar eden metinler olabilir
- anlamsız ya da eksik metadata olabilir

Veri temizliği ekibin sorumluluğundadır.

---

## 6. Teslim Edilmesi Beklenen Dosyalar

Yarışma ek dokümanına göre iki temel teslim vardır.

### 6.1 Çalışan Kaynak Kod

Format:
- `.ipynb`
- veya GitHub repo linki

Beklenti:
- veri temizliği
- feature engineering
- model / skor üretimi
- analiz adımları

tek bir çalışır akış içinde bulunmalıdır.

Kritik şart:
- sunum sırasında gizli test metinleri üzerinde canlı inference yapılabilmelidir

Bu nedenle kod:
- kırılmadan çalışmalı
- bağımlılıklar net olmalı
- inference fonksiyonu ayrı ve pratik kullanılabilir olmalı

### 6.2 Proje Sunumu

Format:
- `.pdf`
- veya `.pptx`

Beklenen içerik:
- problem tanımı
- veri seti özeti
- yöntem
- oluşturulan skor mantığı
- manipülasyon haritaları / analizler
- örnek çıktılar
- canlı demo akışı

Önerilen uzunluk:

```text
5-10 sayfa
```

---

## 7. Teslimat Süreci

Yarışma açıklamasına göre:
- yarışmanın bitimine 1 saat kala bir form linki paylaşılacak
- süre dolduğu anda form kapanacak
- bu süreden sonra ek yükleme kabul edilmeyecek

Bunun pratik sonucu:
- son saate kod yetiştirmeye çalışmak riskli
- sunum ve kod paketinin erken hazır olması gerekir

En doğru yaklaşım:
1. önce çalışan analiz ve skor pipeline'ını ayağa kaldırmak
2. sonra inference fonksiyonunu netleştirmek
3. ardından sunumu hazırlamak
4. son bölümde polish yapmak

---

## 8. Problemin Doğası

Bu problem doğrudan "bot tespiti" olarak ele alınsa da teknik olarak daha doğru çerçeve şudur:

```text
Davranışsal ve semantik manipülasyon riski tespiti
```

Sebep:
- ground truth yok
- her otomatik / koordineli davranış doğrudan bot olmayabilir
- bazı gerçek kullanıcılar da yüksek frekanslı olabilir
- bazı manipülatif kampanyalar tamamen insan kontrollü olabilir

Bu yüzden daha savunulabilir çıktı:
- `risk score`
- `anomaly score`
- `organicity score`

olur.

---

## 9. Önerilen Teknik Yaklaşım

Bu proje için savunulabilir ana yaklaşım şu iskelete oturur:

### 9.1 Davranışsal risk

Kaynaklar:
- paylaşım sıklığı
- zaman düzenliliği
- tekrar eden mesaj paterni
- tema / duygu çeşitliliği
- keyword tekrarları
- metadata anormallikleri

Bu katman, özellikle `author_hash` seviyesinde güçlüdür.

### 9.2 Semantik risk

Kaynaklar:
- embedding uzayı
- vektör benzerliği
- cluster yoğunluğu
- şüpheli semantik kümeler
- kısa sürede yayılan benzer metinler

Bu katman, koordineli içerik kampanyalarını bulmak için güçlüdür.

### 9.3 Nihai skor

Mesaj için önerilen üst form:

```text
final_risk = behavioral_risk + semantic_risk mantığında,
uygulamada ise normalize edilmis agirlikli toplama ile hesaplanir
```

Detaylı formüller için ayrıca:

```text
formulas.md
```

dosyası kullanılmalıdır.

---

## 10. Geliştirme Stratejisi

Tam veri üzerinde doğrudan deneme yapmak verimsiz olabilir.

Önerilen geliştirme akışı:

### 10.1 Örneklem ile geliştirme

Başlangıçta:
- `200K - 500K` satırlık örneklem

Örneklem dengesi için:
- `language`
- `url`
- `date`

kırılımlarını korumak mantıklıdır.

Ek olarak:
- yüksek frekanslı author'lar
- tekrar eden pattern'ler
- eksik metadata içeren satırlar

ayrı bir "risk-focused sample" ile de korunabilir.

### 10.2 Final aşamada tam veri

Önemli kural:
- feature tasarımı sample üzerinde yapılabilir
- ama final normalization ve final dağılım analizi tüm veri üzerinde tekrar hesaplanmalıdır

Sebep:
- percentile
- z-score
- risk threshold

gibi ölçüler sample dağılımında sapabilir.

---

## 11. Jüriye Karşı Savunulabilir Anlatı

Sunumda teknik yaklaşım şu çerçevede anlatılmalıdır:

1. Veri setinde etiket yok
2. Bu nedenle supervised classification yerine risk skorlama tabanlı bir anomali tespit sistemi kuruldu
3. Sistem iki ana katmandan oluşuyor:
   - davranışsal risk
   - semantik risk
4. Her mesaj ve yazar için normalize edilmiş risk oranı hesaplandı
5. Riskli kümeler dil, platform ve zaman ekseninde görselleştirildi
6. Yeni gelen metinler için aynı pipeline ile canlı inference yapılabiliyor

Bu anlatı, yarışmanın problem tanımıyla uyumludur.

---

## 12. Canlı Inference İçin Gerekenler

Sunum anında çalışacak modülün şu özellikleri taşıması gerekir:

### 12.1 Girdi
- yeni bir mesaj metni
- varsa ek metadata:
  - `language`
  - `url`
  - `date`
  - `author_hash`

### 12.2 Çıktı
- `final_risk`
- `behavioral_risk`
- `semantic_risk`
- kısa açıklama

Örnek açıklama yapısı:
- yüksek tekrar paterni
- şüpheli cluster'a yüksek benzerlik
- zaman penceresinde yoğun tekrar
- düşük içerik çeşitliliği

### 12.3 Teknik şart
- hızlı çalışmalı
- tek komut veya tek notebook hücresi ile üretilebilmeli
- demo sırasında dış servislere bağımlı olmaması tercih edilir

---

## 13. Repo İçin Pratik Çalışma Öncelikleri

Bu repo özelinde en mantıklı öncelik sırası:

1. Veri okuma ve keşif
2. Feature engineering
3. Davranışsal risk hesaplama
4. Semantik risk hesaplama
5. Final risk birleştirme
6. Görselleştirme / manipülasyon haritası
7. Canlı inference fonksiyonu
8. Sunum üretimi

---

## 14. Mevcut Repo İçeriği

Bu çalışma klasöründe şu temel dosyalar bulunmaktadır:

- `AGENTS.md`
- `README.txt`
- `datathonFINAL.parquet`
- `sonuçları yükleme.txt`
- `formulas.md`

Ayrıca notebook çalıştırmak için `.venv` ortamı ve veri okuma notebook'u hazırlanmıştır.

---

## 15. Özet

Bu yarışma için teknik olarak en doğru okuma şudur:

- bu bir etiketli bot sınıflandırma yarışması değildir
- bu bir sosyal medya manipülasyonu / anomali tespiti problemidir
- en savunulabilir çözüm, açıklanabilir bir risk skorlama sistemidir
- sistem hem davranışsal hem semantik sinyaller kullanmalıdır
- tüm çözümün canlı demo verebilen çalışır kod ile desteklenmesi gerekir

Bu doküman, genel çerçeveyi toplar.
Detaylı skor formülleri için:

- [formulas.md](/Users/yigit/codes/dataleaguefinal/formulas.md:1)

dosyası kullanılmalıdır.
