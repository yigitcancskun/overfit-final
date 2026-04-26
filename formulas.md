# Risk Scoring Formulas

Bu dokuman, proje icin kullanilacak risk skorlama sisteminin tam formel tanimini verir.

Amac:
- her mesaj (`message-level`) icin bir `final_risk` skoru uretmek
- ayni zamanda her hesap (`author_hash`) icin bir `author_risk` skoru hesaplamak
- davranissal sinyaller ile semantik sinyalleri ayri hesaplayip sonra birlestirmek

Bu yapi etiketli veri olmadan, unsupervised / weakly-supervised bir risk motoru kurmak icin tasarlandi.

---

## 1. Genel Skor Mimarisi

Nihai mesaj riski:

```text
final_risk(m) = alpha * behavioral_risk(m) + beta * semantic_risk(m)
```

Kisit:

```text
alpha + beta = 1
```

Baslangic agirlik onerisi:

```text
alpha = 0.60
beta  = 0.40
```

Aciklama:
- `behavioral_risk(m)`: mesaji atan hesap ve mesajin davranissal paternlerinden gelir
- `semantic_risk(m)`: mesajin embedding uzayindaki konumu, benzer mesajlara yakinligi ve supheli cluster iliskilerinden gelir

Not:
- Tum skorlar `0` ile `1` arasina normalize edilir
- `0`: dusuk risk
- `1`: yuksek risk

---

## 2. Mesaj Seviyesinde Davranissal Risk

Mesaj seviyesindeki davranissal risk, hesap davranisi ile mesaj davranisini birlestirir:

```text
behavioral_risk(m) = gamma * author_behavioral_risk(a) + delta * message_behavioral_risk(m)
```

Burada:
- `m`: mesaj
- `a`: mesaji atan `author_hash`

Kisit:

```text
gamma + delta = 1
```

Baslangic agirlik onerisi:

```text
gamma = 0.70
delta = 0.30
```

Aciklama:
- Botluk ve manipulasyon cogunlukla hesap davranisinda gorulur
- Mesaj seviyesi sinyaller de gerekir, ama tek basina yeterli degildir

---

## 3. Author Seviyesinde Davranissal Risk

Hesap davranissal riski, alti ana bilesenden olusur:

```text
author_behavioral_risk(a) =
    w1 * activity_risk(a) +
    w2 * timing_risk(a) +
    w3 * repetition_risk(a) +
    w4 * diversity_risk(a) +
    w5 * coordination_risk(a) +
    w6 * metadata_anomaly_risk(a)
```

Kisit:

```text
w1 + w2 + w3 + w4 + w5 + w6 = 1
```

Baslangic agirlik onerisi:

```text
w1 = 0.20   # activity
w2 = 0.20   # timing
w3 = 0.25   # repetition
w4 = 0.20   # diversity
w5 = 0.10   # coordination
w6 = 0.05   # metadata anomaly
```

Aciklama:
- `repetition_risk` genelde en guclu bot/manipulasyon sinyalidir
- `activity_risk` ve `timing_risk` davranissal paterni yakalar
- `diversity_risk` makine gibi tekduzeligi yakalar
- `coordination_risk` ag benzeri hareketi yakalar
- `metadata_anomaly_risk` tek basina zayif ama destekleyicidir

---

## 4. Message Seviyesinde Davranissal Risk

Mesaj davranissal riski, mesajin kendi ozelliklerinden gelir:

```text
message_behavioral_risk(m) =
    v1 * text_repetition_risk(m) +
    v2 * text_diversity_risk(m) +
    v3 * keyword_pattern_risk(m) +
    v4 * theme_emotion_pattern_risk(m) +
    v5 * metadata_message_risk(m)
```

Kisit:

```text
v1 + v2 + v3 + v4 + v5 = 1
```

Baslangic agirlik onerisi:

```text
v1 = 0.30
v2 = 0.20
v3 = 0.25
v4 = 0.15
v5 = 0.10
```

Aciklama:
- Bu katman author bazli olmayan, dogrudan mesaja ait riskleri olcer
- Hesabi bilinmeyen ya da bos `author_hash` olan satirlar icin de calisabilir

---

## 5. Activity Risk Formulleri

Ama:
- bir hesabin paylasim yogunlugu normal dagilimin ne kadar disinda
- patlama (`burst`) yapip yapmadigi

### 5.1 Temel tanimlar

Bir author icin:

```text
total_posts(a) = author tarafindan atilan toplam mesaj sayisi
active_days(a) = author'in en az 1 mesaj attigi gun sayisi
active_hours(a) = author'in en az 1 mesaj attigi saat sayisi
```

Gunluk ortalama paylasim:

```text
posts_per_day(a) = total_posts(a) / max(active_days(a), 1)
```

Saatlik ortalama paylasim:

```text
posts_per_active_hour(a) = total_posts(a) / max(active_hours(a), 1)
```

Bir saatteki maksimum paylasim:

```text
max_posts_one_hour(a) = max(hourly_post_count(a, h))
```

Burst orani:

```text
burst_ratio(a) = max_posts_one_hour(a) / max(posts_per_active_hour(a), eps)
```

Burada:

```text
eps = cok kucuk sabit, ornegin 1e-6
```

### 5.2 Activity risk bilesenleri

```text
activity_risk(a) =
    0.50 * N(posts_per_day(a)) +
    0.20 * N(posts_per_active_hour(a)) +
    0.30 * N(burst_ratio(a))
```

Buradaki `N(x)` bir risk-normalizasyon fonksiyonudur. Asagida tanimlanmistir.

Aciklama:
- Asiri uretken hesaplar risklidir
- Saat bazli yogunlasma risklidir
- Kisa sureli patlamalar koordinasyon sinyali olabilir

---

## 6. Timing Risk Formulleri

Ama:
- paylasim araliklari ne kadar mekanik
- paylasim saatleri ne kadar tekduze

### 6.1 Inter-post interval tanimi

Bir author'in zaman sirasina gore siralanmis mesajlari icin:

```text
interval_i(a) = timestamp_i - timestamp_(i-1)
```

Ortalama interval:

```text
mean_interval(a) = mean(interval_i(a))
```

Interval standart sapmasi:

```text
std_interval(a) = std(interval_i(a))
```

Varyasyon katsayisi:

```text
interval_cv(a) = std_interval(a) / max(mean_interval(a), eps)
```

### 6.2 Saat dagilimi entropisi

Her author icin 24 saatlik dagilim:

```text
p_h(a) = author'in h saatindeki mesaj oranı
```

Saat entropisi:

```text
hour_entropy(a) = - sum_h p_h(a) * log(p_h(a) + eps)
```

Normalize edilmis saat entropisi:

```text
hour_entropy_norm(a) = hour_entropy(a) / log(24)
```

### 6.3 Timing risk

Burada dusuk cesitlilik veya asiri duzenlilik risklidir. O yuzden ters normalizasyon kullanilir:

```text
timing_risk(a) =
    0.60 * I(interval_cv(a)) +
    0.40 * I(hour_entropy_norm(a))
```

Burada:
- `I(x)` = inverse-risk-normalization
- dusuk deger yuksek risk demektir

Aciklama:
- `interval_cv` dusukse, mesaj aralari cok duzenlidir
- `hour_entropy_norm` dusukse, hesap surekli ayni saatlerde paylasim yapiyordur

---

## 7. Repetition Risk Formulleri

Ama:
- hesap ayni veya cok benzer icerigi tekrar ediyor mu

### 7.1 Exact duplicate ratio

Bir author icin ayni metinlerin oranı:

```text
exact_duplicate_ratio(a) =
    duplicated_text_count(a) / max(total_posts(a), 1)
```

Burada:
- `duplicated_text_count(a)`: ayni `original_text` ile birden fazla kez gecen author mesajlari

### 7.2 Near-duplicate ratio

Metin benzerligi tabanli:

```text
near_duplicate_ratio(a) =
    near_duplicate_post_count(a) / max(total_posts(a), 1)
```

Burada `near_duplicate_post_count` su sekilde tanimlanabilir:
- ayni author icinde cosine similarity veya Jaccard similarity esigi uzerindeki mesajlar

### 7.3 Keyword repeat ratio

```text
keyword_repeat_ratio(a) =
    repeated_keyword_pattern_count(a) / max(total_posts(a), 1)
```

Burada:
- `english_keywords` normalize edilir, siralanir, canonical forma cevrilir
- ayni keyword setinin tekrar etme orani olculur

### 7.4 Repetition risk

```text
repetition_risk(a) =
    0.40 * N(exact_duplicate_ratio(a)) +
    0.35 * N(near_duplicate_ratio(a)) +
    0.25 * N(keyword_repeat_ratio(a))
```

Aciklama:
- Ayni veya cok benzer icerik tekrarinin yuksek olmasi bot/manipulasyon icin cok guclu sinyaldir

---

## 8. Diversity Risk Formulleri

Ama:
- author cok dar bir icerik uzayinda mi hareket ediyor

### 8.1 Theme entropy

Bir author icin tema dagilimi:

```text
p_t(a) = theme t altindaki mesaj orani
```

Tema entropisi:

```text
theme_entropy(a) = - sum_t p_t(a) * log(p_t(a) + eps)
```

Normalize edilmis:

```text
theme_entropy_norm(a) = theme_entropy(a) / log(num_themes)
```

### 8.2 Emotion entropy

```text
p_e(a) = emotion e altindaki mesaj orani

emotion_entropy(a) = - sum_e p_e(a) * log(p_e(a) + eps)

emotion_entropy_norm(a) = emotion_entropy(a) / log(num_emotions)
```

### 8.3 Text length variability

Mesaj uzunlugu:

```text
text_len(m) = karakter veya token sayisi
```

Bir author icin:

```text
text_len_std(a) = std(text_len(m))
text_len_cv(a)  = text_len_std(a) / max(mean(text_len(m)), eps)
```

### 8.4 Lexical diversity

Bir author'in tum mesajlari birlestirilerek:

```text
lexical_diversity(a) = unique_token_count(a) / max(total_token_count(a), 1)
```

### 8.5 Diversity risk

Dusuk cesitlilik yuksek risk oldugu icin ters normalizasyon kullanilir:

```text
diversity_risk(a) =
    0.30 * I(theme_entropy_norm(a)) +
    0.25 * I(emotion_entropy_norm(a)) +
    0.20 * I(text_len_cv(a)) +
    0.25 * I(lexical_diversity(a))
```

Aciklama:
- temalar cok tekduzeyse risk artar
- emotion kullanim sekli tekduzeyse risk artar
- mesaj uzunluklari cok benzerse risk artar
- kelime cesitliligi dusukse risk artar

---

## 9. Coordination Risk Formulleri

Ama:
- author baska hesaplarla koordineli gorunuyor mu

Bu kisim davranissal ve semantik arasinda bir kopru gorevi gorur.

### 9.1 Time-window synchronization

Bir zaman penceresi tanimlanir:

```text
window = 10 dakika veya 30 dakika
```

Her mesaj icin:
- ayni pencere icinde
- ayni platformda
- ayni dilde
- benzer tema veya benzer keyword yapisinda
paylasim yapan hesap sayisi bulunur

Bir author icin:

```text
sync_count(a) = author mesajlarinin dahil oldugu senkron event sayisi
sync_ratio(a) = sync_count(a) / max(total_posts(a), 1)
```

### 9.2 Coordination risk

```text
coordination_risk(a) = N(sync_ratio(a))
```

Aciklama:
- Eger author mesajlari surekli toplu hareket eden paternler icinde yer aliyorsa risk artar

Not:
- Bu bilesen ileride vector-store / embedding benzerligi ile guclendirilebilir

---

## 10. Metadata Anomaly Risk Formulleri

Ama:
- metadata tarafinda supheli aykiriliklar var mi

### 10.1 Missing author rate

Bir author'a ozel degil, mesaj seviyesi veya grup seviyesi ele alinabilir.

Eger `author_hash` bos ise mesaj bazli risk eklenebilir:

```text
missing_author_flag(m) = 1 if author_hash bos else 0
```

### 10.2 Platform concentration

Bir author icin:

```text
platform_concentration(a) = max_platform_share(a)
```

Burada:
- author'in mesajlarinin en buyuk paya sahip platform oranı

### 10.3 Language-platform anomaly

Bu veri dagilimindan cikar:

```text
lp_prob(language, platform) = P(language, platform)
```

Mesaj icin:

```text
language_platform_anomaly(m) = 1 - lp_prob(language_m, platform_m)
```

Author icin ortalaması:

```text
avg_language_platform_anomaly(a) = mean(language_platform_anomaly(m))
```

### 10.4 Metadata anomaly risk

```text
metadata_anomaly_risk(a) =
    0.50 * N(platform_concentration(a)) +
    0.50 * N(avg_language_platform_anomaly(a))
```

Mesaj bazli ilave ceza:

```text
metadata_message_penalty(m) = 0.10 * missing_author_flag(m)
```

Bu ceza `message_behavioral_risk(m)` icine eklenebilir.

---

## 11. Message-Level Text Repetition Risk

Mesaj bazli tam tekrar riski:

```text
text_repetition_risk(m) = N(local_duplicate_density(m))
```

Burada:
- `local_duplicate_density(m)`: ayni ya da cok benzer mesajin veri setinde kac kez tekrar ettigi

Ornek tanim:

```text
local_duplicate_density(m) =
    duplicate_cluster_size(m) / total_messages
```

Pratikte daha yararli versiyon:

```text
local_duplicate_density(m) =
    duplicate_cluster_size(m) / max(cluster_average_size, 1)
```

---

## 12. Message-Level Text Diversity Risk

Mesaj tekduzeligi:

```text
text_diversity_risk(m) =
    0.50 * I(unique_token_ratio(m)) +
    0.50 * I(text_length_relative_variation(m))
```

Burada:

```text
unique_token_ratio(m) = unique_token_count(m) / max(total_token_count(m), 1)
```

ve

```text
text_length_relative_variation(m)
```

mesaj uzunlugunun ilgili tema / dil / platform grubuna gore ne kadar dogal oldugunu olcer.

Ornek:

```text
text_length_relative_variation(m) =
    abs(text_len(m) - median_group_text_len(group)) / IQR_group_text_len(group)
```

Burada group = `(language, url, primary_theme)`

Not:
- Bu feature aslinda yuksek sapmada da risk verebilir
- Dilersen burada `I()` yerine `A()` isimli aykirilik fonksiyonu kullanabilirsin

---

## 13. Message-Level Keyword Pattern Risk

```text
keyword_pattern_risk(m) =
    0.70 * N(keyword_pattern_frequency(m)) +
    0.30 * N(keyword_pattern_burstiness(m))
```

Burada:

```text
keyword_pattern_frequency(m)
```

aynı keyword yapisinin veri setinde ne kadar sık tekrar ettigini,

```text
keyword_pattern_burstiness(m)
```

aynı pattern'in kisa zaman penceresinde ne kadar yogunlastigini olcer.

---

## 14. Message-Level Theme-Emotion Pattern Risk

```text
theme_emotion_pattern_risk(m) =
    0.50 * N(theme_emotion_frequency(m)) +
    0.50 * N(theme_emotion_burstiness(m))
```

Aciklama:
- Belirli tema + duygu kombinasyonlarinin anormal frekansta ve yogunlukta tekrar etmesi manipulasyon isareti olabilir

---

## 15. Message-Level Metadata Risk

```text
metadata_message_risk(m) =
    0.60 * missing_author_flag(m) +
    0.40 * language_platform_anomaly(m)
```

Not:
- Eger bu terim `0-1` disina cikarsa tekrar normalize edilir

---

## 16. Semantic Risk Formulleri

Semantik risk, mesajin embedding uzayindaki konumundan gelir.

```text
semantic_risk(m) =
    s1 * cluster_density_risk(m) +
    s2 * suspicious_cluster_membership_risk(m) +
    s3 * nearest_neighbor_similarity_risk(m) +
    s4 * semantic_burst_risk(m)
```

Kisit:

```text
s1 + s2 + s3 + s4 = 1
```

Baslangic agirlik onerisi:

```text
s1 = 0.20
s2 = 0.35
s3 = 0.30
s4 = 0.15
```

### 16.1 Cluster density risk

```text
cluster_density_risk(m) = N(cluster_size(m))
```

veya daha iyi:

```text
cluster_density_risk(m) = N(local_embedding_density(m))
```

### 16.2 Suspicious cluster membership risk

Bir cluster'in suphelilik puani:

```text
cluster_suspicion_score(c) =
    0.50 * mean_behavioral_risk(c) +
    0.30 * time_burstiness(c) +
    0.20 * cross_author_homogeneity(c)
```

Mesaj icin:

```text
suspicious_cluster_membership_risk(m) = cluster_suspicion_score(cluster(m))
```

### 16.3 Nearest neighbor similarity risk

```text
nearest_neighbor_similarity_risk(m) =
    mean_top_k_cosine_similarity(m, suspicious_neighbors)
```

veya:

```text
nearest_neighbor_similarity_risk(m) =
    max_cosine_similarity_to_suspicious_cluster(m)
```

### 16.4 Semantic burst risk

```text
semantic_burst_risk(m) = N(similar_message_count_in_time_window(m))
```

Aciklama:
- cok benzer mesajlar kisa surede yigin halinde yayinlaniyorsa risk artar

---

## 17. Normalizasyon Fonksiyonlari

Tum ham feature'lari direkt toplamak yanlistir. Once ortak skala olan `0-1` araligina getirilmeleri gerekir.

### 17.1 Standard risk normalization

Yuksek deger yuksek risk demekse:

```text
N(x) = percentile_rank(x)
```

Burada:

```text
N(x) in [0, 1]
```

### 17.2 Inverse risk normalization

Dusuk deger yuksek risk demekse:

```text
I(x) = 1 - percentile_rank(x)
```

### 17.3 Outlier normalization

Eger hem cok dusuk hem cok yuksek degerler riskliyse:

```text
A(x) = min(1, abs(robust_z(x)) / tau)
```

Robust z-score:

```text
robust_z(x) = (x - median(x)) / max(IQR(x), eps)
```

Burada:

```text
tau = aykirilik esigi, ornegin 3
```

### 17.4 Winsorization

Cok asiri degerlerin sistemi bozmasini engellemek icin:

```text
x_winsorized = clip(x, p01, p99)
```

Sonra `N(x_winsorized)` uygulanabilir.

---

## 18. Risk Bandlari

Final skorun yorumlanabilir olmasi icin band tanimlanir:

```text
0.00 - 0.20 : low risk
0.20 - 0.40 : guarded
0.40 - 0.60 : elevated
0.60 - 0.80 : high
0.80 - 1.00 : critical
```

Alternatif:
- bandlari sabit araliklarla degil
- percentiles ile de tanimlayabilirsin

Ornek:

```text
P0-P80  : low
P80-P95 : medium
P95-P99 : high
P99+    : critical
```

Eger veri dagilimi cok carpik cikarsa percentile band daha dogru olur.

---

## 19. Tam Nihai Formul

Tum sistemi tek zincirde yazarsak:

```text
final_risk(m) =
    alpha * behavioral_risk(m) +
    beta  * semantic_risk(m)
```

```text
behavioral_risk(m) =
    gamma * author_behavioral_risk(a) +
    delta * message_behavioral_risk(m)
```

```text
author_behavioral_risk(a) =
    w1 * activity_risk(a) +
    w2 * timing_risk(a) +
    w3 * repetition_risk(a) +
    w4 * diversity_risk(a) +
    w5 * coordination_risk(a) +
    w6 * metadata_anomaly_risk(a)
```

```text
message_behavioral_risk(m) =
    v1 * text_repetition_risk(m) +
    v2 * text_diversity_risk(m) +
    v3 * keyword_pattern_risk(m) +
    v4 * theme_emotion_pattern_risk(m) +
    v5 * metadata_message_risk(m)
```

```text
semantic_risk(m) =
    s1 * cluster_density_risk(m) +
    s2 * suspicious_cluster_membership_risk(m) +
    s3 * nearest_neighbor_similarity_risk(m) +
    s4 * semantic_burst_risk(m)
```

---

## 20. Onerilen Baslangic Agirliklari

```text
alpha = 0.60
beta  = 0.40

gamma = 0.70
delta = 0.30

w1 = 0.20
w2 = 0.20
w3 = 0.25
w4 = 0.20
w5 = 0.10
w6 = 0.05

v1 = 0.30
v2 = 0.20
v3 = 0.25
v4 = 0.15
v5 = 0.10

s1 = 0.20
s2 = 0.35
s3 = 0.30
s4 = 0.15
```

Bu agirliklar baslangic icindir.

Kalibrasyon su sekilde yapilir:
1. sample veri uzerinde feature dagilimlarini incele
2. supheli cluster ve case'ler uzerinden hata analizi yap
3. agirliklari eldeki bulgulara gore guncelle
4. final pipeline'da tum veri uzerinde yeniden normalize et

---

## 21. Uygulama Kurallari

Bu dokumandaki sistemi kodlarken su kurallari uygula:

1. Ham feature'lari once author-level ve message-level olarak ayri tablolarda uret
2. Her feature icin `N`, `I` veya `A` secimi acikca tanimla
3. Tum normalization'lari sample'da degil final dagilimda yeniden hesapla
4. `author_hash` bos olan mesajlar icin author-level risk yerine sadece message-level risk kullan
5. Tum ara skorlar `0-1` araliginda tutulmali
6. Nihai skor tek basina karar degil, risk oranidir

---

## 22. Kisa Uygulama Ozet Formu

En kisa haliyle sistem:

```text
final_risk = 0.60 * behavioral_risk + 0.40 * semantic_risk
```

```text
behavioral_risk = 0.70 * author_behavioral_risk + 0.30 * message_behavioral_risk
```

```text
author_behavioral_risk =
    0.20 * activity_risk +
    0.20 * timing_risk +
    0.25 * repetition_risk +
    0.20 * diversity_risk +
    0.10 * coordination_risk +
    0.05 * metadata_anomaly_risk
```

Bu yapi, bu proje icin ilk calisan ve savunulabilir iskelet olarak kullanilabilir.
