## WHAT IS THIS PROJECT

Bu proje, Dataleague final Datathon'udur.

# Yarışma kuralları

README.txt içinde bulunmaktadır.

# yarışma bilgilendirme

@yarisma_bilgilendirme.md'yi oku.

# DATALEAGUE DATATHON: VERI SETI REHBERI VE GOREV BILDIRIMI

1. GOREV (TASK) TANIMI

---

Gorev Ismi: "Sosyal Medya Manipulasyonu ve Anomali Tespit Sistemi"
Detay: Size verilen bu global sosyal medya veri setini kullanarak; iceriklerin hangilerinin organik (gercek kullanici), hangilerinin ise organize/manipulatif (bot aglari, algi kampanyalari) oldugunu tespit eden ve yeni gelen verilere anlik tepki verebilen bir sistem gelistirmeniz beklenmektedir.

Beklenen Ciktilar:

- Guvenilirlik/Organiklik Skoru Algoritmasi: Her bir icerige (veya author_hash'e) 0 ile 1 arasinda bir skor atayan model.
- Manipulasyon Haritasi (Dashboard): Tespit edilen manipulatif icerik kumelerinin hangi dillerde ve platformlarda yogunlastigini gosteren gorsel rapor.
- Canli Tahmin Modeli (Inference Pipeline): Sunum esnasinda juri tarafindan verilecek gizli (unseen) metinleri anlik isleyip "Organik/Manipulatif" ayrimi yapabilen ve kararin nedenini aciklayan fonksiyon.

2. VERI SETI OZETI

---

Dosya Adi: datathonFINAL.parquet
Format: Apache Parquet (Pandas veya PyArrow ile okunabilir)
Satir Sayisi: ~5.000.000
Boyut: ~1 GB

3. SUTUN ACIKLAMALARI

---

- original_text: Sosyal medya gonderisinin orijinal, ham metni.
- english_keywords: Metinden cikarilmis Ingilizce anahtar kelimeler.
- sentiment: Duygu skoru (-1.0 Negatif, 0.0 Notr, 1.0 Pozitif).
- main_emotion: Metne hakim olan ana duygu (Joy, Anger, vb.).
- primary_theme: Gonderinin ana konusu/temasi.
- language: Metnin orijinal dili (ISO Kodu: tr, en, es vb.).
- url: Verinin geldigi platform (x.com, reddit.com vb.).
- author_hash: Anonimlestirilmis kullanici kimligi (ID).
- date: Gonderinin UTC saat ve tarih bilgisi.

4. TEKNIK UYARILAR

---

- HEDEF SUTUN YOKTUR: Veri setinde kimin "Bot" veya "Insan" oldugunu gosteren bir etiket (label) bulunmamaktadir. Problem tamamen Unsupervised (Gozetimsiz) bir anomali tespit problemidir.
- RAM YONETIMI: 5 milyon satiri islerken bilgisayarinizin OOM (Out of Memory) hatasi verip donmamasi icin veriyi parca parca (batch) islemeniz veya ornekleme (sampling) yapmaniz siddetle onerilir.
- KIRLI VERI: Bu veri sentetik degildir, gercek dunya verisidir. Bosluklar, yazim hatalari ve argolar icerebilir. Veri temizligi sizin sorumlulugunuzdadir.

Basarilar dileriz!
DataLeague & YAZGIT Veri Muhendisligi Ekibi
