# Cleaning Theory

Bu dokuman, veri temizleme stratejisi uzerindeki ortak kararlarin kaydidir.
Her madde icin durum bilgisi tutulur:

- `Kabul edildi`
- `Reddedildi`
- `Acik / netlestirildi`

Amac:
- hangi cleaning kararlarinin uygulanacagini sabitlemek
- modelleme sirasinda manipülasyon sinyallerini yanlislikla silmemek
- kullanici tercihlerini sonradan kaybetmemek

---

## 1. Temel Ilke

Bu veri setinde asagidaki ayrim korunacak:

- gercekten bozuk veri temizlenir
- supheli davranis kaliplari silinmez, feature veya risk sinyali olarak korunur

Bu proje klasik data cleaning degil, ayni zamanda anomaly / manipulation detection problemi oldugu icin tekrar, burst, koordinasyon ve tekduzelik gibi yapilarin bir kismi "kir" degil, dogrudan sinyaldir.

---

## 2. Karar Kayitlari

### 2.1 Bos stringleri eksik deger sayma

Durum: `Kabul edildi`

Uygulanacak:
- `''`
- sadece whitespace iceren degerler
- uygun gorulen placeholder degerler (`nan`, `null`, `none`, `n/a` gibi)

bu alanlarda eksik deger olarak ele alinacak.

Oncelikli kolonlar:
- `author_hash`
- `original_text`
- `english_keywords`
- `primary_theme`

Not:
- Ham veride teknik olarak null olmayabilir.
- Ama bos stringler gercek eksik deger gibi davranacaktir.

---

### 2.2 `date` kolonunu parse etmek

Durum: `Kabul edildi`

Uygulanacak:
- `date` UTC datetime olarak parse edilecek
- parse edilmis kolon uzerinden zaman turevleri gerekirse sonradan uretilecek

Ornek turev kolonlar:
- `date_hour`
- `date_day`
- `date_week`
- `weekday`

Not:
- Mevcut incelemede tarih parse hatasi gorulmedi.

---

### 2.3 Metin normalizasyonu

Durum: `Kabul edildi`

Uygulanacak:
- bas ve son bosluk temizligi
- coklu bosluklari normalize etme
- satir sonlarini normalize etme
- Unicode normalization

Uygulanmayacak:
- lowercase / uppercase donusumu

Unicode normalization aciklamasi:
- Bazi karakterler ayni gorunse de bilgisayarda farkli byte dizileriyle tutulabilir.
- Ornek: aksanli harfler bazen tek karakter, bazen harf + ek isaret olarak saklanir.
- Unicode normalization, bu esit gibi gorunen yazimlari daha tutarli hale getirir.
- Bu islem metnin anlamini degistirmez; daha cok teknik tutarlilik saglar.

Not:
- Ham metin korunabilir.
- Istenirse ayri bir `normalized_text` kolonu olusturulabilir.

---

### 2.4 Bos veya asiri kisa metinler

Durum: `Kabul edildi`

Uygulanacak:
- `original_text` bos olan kayitlar dusuk kaliteli icerik olarak ele alinacak
- asiri kisa metinler ayrica tespit edilecek

Not:
- Burada agresif silme degil, dikkatli ele alma yaklasimi tercih edilecek.

---

### 2.5 Exact duplicate temizligi

Durum: `Kabul edildi`

Uygulanacak:
- Ayni `original_text + author_hash + date + url` kombinasyonu birebir tekrar ediyorsa gercek duplicate olarak degerlendirilebilir
- Bu tur teknik tekrarlar deduplicate edilebilir

Uygulanmayacak:
- Sadece `original_text` ayni diye kayit silinmeyecek

Gerekce:
- Bu projede tekrar paterni risk sinyalidir.
- Asiri agresif duplicate silme, manipülasyon sinyalini yok eder.

---

### 2.6 Near-duplicate temizligi

Durum: `Reddedildi`

Reddedilen onerme:
- Kucuk farklarla tekrar eden metinleri cleaning asamasinda gruplayip temizlemek

Kullanicinin pozisyonu:
- Bu asama kabul edilmedi

Kayda gecen teknik savunma:
- Near-duplicate yapilar koordineli kampanya, spam varyasyonu ve semantik tekrar icin cok guclu sinyal olabilir
- Cleaning asamasinda silinirse sonraki `repetition_risk` ve `semantic_risk` zayiflar
- Bu nedenle near-duplicate mantigi cleaning degil, daha cok feature engineering veya risk hesaplama katmaninda degerlendirilmelidir

Sonuc:
- Cleaning asamasinda near-duplicate silme veya collapse islemi uygulanmayacak
- Gerekirse daha sonra modelleme katmaninda feature olarak kullanilabilir

---

### 2.7 `author_hash` bos kayitlar

Durum: `Kabul edildi, ek formul ihtiyaci var`

Kullanicinin pozisyonu:
- Bu kayitlar atilmayacak
- Formulasyon asamasinda bu satirlar icin ozel bir sistem dusunulmeli

Uygulanacak temel ilke:
- `author_hash` bos olan satirlarda klasik author-level risk ayni sekilde kullanilmayacak

Onerilen yaklasim:
- author'a bagli olmayan message-level sinyaller daha agirlikli kullanilabilir
- grup-temelli risk kullanilabilir

Ek oneriler:
- `surrogate_group_risk` kurulabilir:
  - `(language, normalized_domain, time_window, primary_theme)` tabanli grup sinyali
- `contextual_risk` kurulabilir:
  - author yoksa mesaj, platform, zaman ve benzer icerik komsulugu ile risk hesabi
- `author_hash_missing_flag` risk modelinde acik bir sinyal olabilir

Onerilen prensip:
- `author_hash` yoksa bilgi kaybi var, ama karar imkansiz degil
- author-level yerine context-level risk kullanmak daha savunulabilir

---

### 2.8 `url` / domain normalizasyonu

Durum: `Kabul edildi`

Kullanicinin sorusu:
- `google.com` ve `news.google.com` gibi ic ice domain yapilarinda ne yapilacak

Uygulanacak prensip:
- Tek bir "domain" alanina koru-koru normalize etmek yetersiz olabilir
- En az iki seviye tutulmali

Onerilen yapi:
- `raw_url_domain`
- `registered_domain`
- `subdomain`
- gerekirse `platform_group`

Ornek:
- `news.google.com`
  - `registered_domain = google.com`
  - `subdomain = news`
- `www.reddit.com`
  - `registered_domain = reddit.com`
  - `subdomain = www`

Neden:
- Bazi durumlarda alt alan anlamsizdir (`www`)
- Bazi durumlarda alt alan anlamsaldir (`news.google.com`, `m.youtube.com`, `old.reddit.com`)

Pratik karar:
- Analizde hem kok domain hem alt domain bilgisi korunmali
- Kor normalize edip alt domain bilgisini yok etmek dogru degil

---

### 2.9 Rare language isaretleme

Durum: `Kabul edildi`

Uygulanacak:
- cok dusuk frekansli diller `rare` olarak isaretlenebilir

Uygulanmayacak:
- `unknown` gibi yapay fallback kategori uretmek

Not:
- Dil kodu korunacak
- Sadece dusuk frekansli olanlar ayrica raporlanabilir

---

### 2.10 `english_keywords` temizligi

Durum: `Kabul edildi, soru netlestirildi`

Kullanicinin sorusu:
- tekrar eden keyword derken ayni entry icindekiler mi?

Cevap:
- Evet, once ayni kaydin kendi icindeki tekrar eden keyword'lerden bahsediliyor

Uygulanacak:
- `english_keywords` parse edilecek
- ayiricilar temizlenecek
- bas/son bosluklar temizlenecek
- ayni satir icindeki duplicate keyword'ler kaldirilacak

Ornek:
- `apple, market, apple, stock`

su hale getirilebilir:
- `apple, market, stock`

Ek not:
- Bu, satirlar arasi keyword tekrarini silmek degil
- Sadece tek kaydin icindeki gereksiz tekrar temizligidir

---

### 2.11 `primary_theme` icin eksik etiketleme

Durum: `Kabul edildi`

Uygulanacak:
- `primary_theme` bos ise `unknown_theme` olarak etiketlenebilir

Uygulanmayacak:
- nadir temalari `other` altinda toplamak

---

### 2.12 Metin kalite flag seti

Durum: `Reddedildi`

Reddedilen onerme:
- `text_len_chars`
- `text_len_words`
- `unique_token_ratio`
- `non_alnum_ratio`
- `url_count`
- `hashtag_count`
- `mention_count`

gibi genis bir kalite flag setinin cleaning parcasi olarak kurulmasi

Kullanicinin pozisyonu:
- bu yaklasim begenilmedi
- uygulanmayacak

Sonuc:
- Bu asamada genis kalite etiketleme sistemi kurulmayacak

---

### 2.13 Dil-metin uyumsuzlugu label'lama

Durum: `Pratik yontemle netlestirildi`

Kullanicinin itirazi:
- 5 milyon kaydi LLM ile etiketlemek mantikli degil

Kesin karar:
- LLM tabanli satir-satir etiketleme yapilmayacak

Uygulanabilecek pratik alternatifler:
- karakter dagilimi tabanli heuristic
- script detection
- hafif dil tespit kutuphaneleriyle sample tabanli kontrol
- sadece supheli alt kume uzerinde denetim

Not:
- Bu konu cleaning'in zorunlu cekirdek parcasi degil
- Ilk versiyonda tamamen ertelenebilir

---

### 2.14 Zaman bazli burst / koordinasyon yapilari

Durum: `Not edildi`

Kullanicinin yorumu:
- bu madde tam anlasilmadi, not alinmasi istendi

Aciklama:
- Ayni dakika veya dar bir zaman penceresinde cok benzer iceriklerin toplu sekilde cikmasi manipulasyon sinyali olabilir
- Bu tur yapilar cleaning asamasinda silinmemeli
- Daha sonra `burst`, `coordination`, `semantic_burst` veya `sync_ratio` gibi feature'lara donusturulebilir

Bu asamadaki karar:
- silme yok
- not olarak korunacak
- feature engineering asamasinda tekrar ele alinacak

---

### 2.15 Supheli veriyi silmemek

Durum: `Kabul edildi`

Temel ilke:
- cleaning bahanesiyle manipülasyon sinyalleri yok edilmeyecek

---

### 2.16 Ayri quality table uretmek

Durum: `Reddedildi`

Reddedilen onerme:
- `clean_messages` ve `message_quality_flags` gibi iki ayri tablo tutmak

Sonuc:
- Ayrik bir quality-table tasarimi uygulanmayacak

---

### 2.17 Temizlik sonrasi yeniden kontrol

Durum: `Kabul edildi`

Temizlik sonrasinda tekrar bakilacak:
- bos alan oranlari
- duplicate oranlari
- domain / language dagilimlari
- satir sayisi farki

---

## 3. Uygulanacak Cekirdek Cleaning Seti

Su anki kararlara gore ilk cleaning cekirdegi:

1. Bos stringleri eksik deger kabul et
2. `date` kolonunu UTC datetime'a parse et
3. Metinlerde bosluk ve satir sonu normalizasyonu yap
4. Unicode normalization uygula
5. Lowercase / uppercase donusumu yapma
6. Bos veya asiri kisa metinleri tespit et
7. Sadece gercek exact duplicate kayitlari deduplicate et
8. `author_hash` bos kayitlari atma
9. `url` icin kok domain + subdomain bilgisini koru
10. `english_keywords` alanini parse edip satir ici tekrarlarini temizle
11. `primary_theme` bos ise `unknown_theme` etiketi kullan
12. Rare language mantigini gerekirse uygula
13. Temizlik sonrasi temel dagilim kontrollerini yeniden yap

---

## 4. Bilerek Ertelenen veya Cleaning Disinda Tutulan Basliklar

Su konular cleaning'in cekirdeginden bilerek disarida tutuldu:

- near-duplicate collapse
- genis quality-flag sistemi
- LLM tabanli language-content verification
- quality-table ayirma

Bu konularin bir kismi ileride su asamalarda geri gelebilir:
- feature engineering
- risk scoring
- semantic analysis

---

## 5. Acik Tasarim Notlari

Henuz tamamen kapanmayan ama yonu belli olan basliklar:

### 5.1 `author_hash` bos kayitlar icin alternatif risk modeli

Bu kisim sonraki asamada ayrica tasarlanacak.

Ilk adaylar:
- message-heavy scoring
- context-level scoring
- surrogate group scoring

### 5.2 Domain hiyerarsisi

Tek kolon yerine cok seviyeli domain temsili daha dogru gorunuyor:
- registered domain
- subdomain
- gerekirse platform group

### 5.3 Burst / coordination sinyali

Bu kisim cleaning degil, feature engineering tarafinda ele alinacak.

---

## 6. Sonuc

Bu dokumanin ana karari sudur:

- veri temizlenecek, ama risk sinyali olabilecek yapilar temizlenmeyecek
- cleaning ile anomaly sinyalini birbirine karistirmayacagiz
- gereksiz agresif sadeleştirme yerine kontrollu standardizasyon uygulanacak

Bu belge, mevcut kullanici kararlarina gore hazirlanmistir.
