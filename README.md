# Fusion Risk Scoring Model

Fusion, sosyal medya içeriklerinde manipülatife yakın davranış paternlerini bulmak için yazılmış açıklanabilir bir risk skorlama modelidir.

Bu proje klasik supervised bot classifier değildir. Veri setinde `bot`, `human`, `manipulative`, `organic` gibi hedef etiket yoktur. Bu nedenle model, etiket öğrenmek yerine davranışsal ve semantik sinyallerden `0-1` arası risk skoru üretir.

## Modelin Amacı

Fusion aşağıdaki sorulara cevap verir:

- Bir mesaj organik mi, manipülatife yakın mi?
- Bir `author_hash` normal kullanıcı davranışina mi, koordineli/yüksek frekanslı davranışa mi benziyor?
- Aynı veya benzer metinler kısa zaman penceresinde tekrar ediyor mu?
- RoBERTa tabanlı semantik model mesajı bot/manipulasyon sinyaline yakın goruyor mu?
- Son skor hangi davranışsal ve semantik bileşenlerden geliyor?

Ana çıktı:

```text
final_score: 0.0 = düşük risk, 1.0 = yüksek risk
```

## Mimari

Fusion iki ana katmandan oluşur:

```text
final_score = behavioral_weight * behavioral_score
            + roberta_weight    * roberta_score
```

Önce baz ağırlıklar tanımlanır:

```text
behavioral = 0.45
roberta    = 0.55
```

Sonra her skorun 0.5'ten uzaklığına göre güven katsayısı hesaplanır:

```text
distance = abs(score - 0.5)
normalized_distance = distance / 0.5
confidence(score) = min_weight + (1 - min_weight) * normalized_distance^power
```

Varsayılan:

```text
min_weight = 0.20
power      = 2.00
sigmoid_steepness = 8.00
```

Bu yapı paraboliktir:

```text
score = 0.50 -> confidence = 0.20
score = 0.00 -> confidence = 1.00
score = 1.00 -> confidence = 1.00
score = 0.10 -> confidence = score 0.90 ile aynıdır
score = 0.45 -> confidence = score 0.55 ile aynıdır
```

`min_weight = 0.20` ve `power = 2.00` için eşdeğer açık form:

```text
confidence(score) = 0.20 + 3.20 * (score - 0.5)^2
```

Nihai efektif ağırlık sigmoid gate ile hesaplanır:

```text
raw_behavioral_weight = behavioral_base_weight * confidence(behavioral_score)
raw_roberta_weight    = roberta_base_weight    * confidence(roberta_score)

roberta_effective_weight =
    sigmoid(sigmoid_steepness * (raw_roberta_weight - raw_behavioral_weight))

behavioral_effective_weight =
    1 - roberta_effective_weight

final_score_before_rules =
    behavioral_effective_weight * behavioral_score +
    roberta_effective_weight    * roberta_score
```

Hard rule tetiklenirse `final_score` yine doğrudan `1.0` olur.

### 1. Davranışsal Katman

Davranışsal skor iki parçadan gelir:

```text
behavioral_score = author_weight * author_score
                 + message_weight * message_score
```

Varsayılan:

```text
author  = 0.90
message = 0.10
```

`author_score` şunları kullanır:

- paylaşım frekansi
- bir saatteki maksimum mesaj sayısı
- mesajlar arası zaman araligi
- ayni metni tekrar etme oranı
- farklı author'lar tarafından ayni metnin kullanılması
- dil, tema ve sentiment çeşitliliği

`message_score` şunları kullanır. Bu skor artık düşük ağırlıklı yardımcı davranış skoru olarak tutulur:

- ayni metnin tekrar sayısı
- ayni metni atan benzersiz author sayısı
- ayni metnin kısa zaman penceresindeki yoğunluğu
- hashtag yoğunluğu
- token tekrar paterni
- uzun metin cezasi
- keyword sinyali

Hard rule:

- Author için aşırı saatlik patlama varsa skor doğrudan `1.0` olabilir.
- Aynı metin hem çok tekrar ediyor, hem birden fazla author tarafından kullanılıyor, hem de kısa zaman penceresinde yoğunlaşıyorsa skor doğrudan `1.0` olabilir.

### 2. Semantik Katman

Semantik katman preset tabanlı sequence classification modeli kullanır:

```python
"semantic_adapter": {
    "selected_model_key": "distil_bot_en",
    "models": {
        "distil_bot_en": {
            "model_name": "junaid1993/distilroberta-bot-detection",
            "supported_languages": ["en"],
        },
        "xlmr_base_multilingual": {
            "model_name": "FacebookAI/xlm-roberta-base",
            "supported_languages": "all",
        },
    },
}
```

Varsayılan preset:

```text
distil_bot_en
```

`distil_bot_en` sadece İngilizce çalışır. `xlmr_base_multilingual` ise config içinde test amaçlı preset olarak durur, fakat ham `FacebookAI/xlm-roberta-base` checkpoint'i bot classifier değildir. Bu preset seçilecekse `model_name` alanı fine-tune edilmiş bir XLM-R sequence-classification checkpoint ile değiştirilmelidir.

Desteklenmeyen diller için semantik skor varsayılan olarak nötr kabul edilir:

```python
"unsupported_language_score": 0.50
```

## Neden Fusion

Tek başına RoBERTa yeterli değildir.

Sebep:

- Veri çok dilli.
- Model sadece desteklenen dillerde güvenilir çalışır.
- Manipülasyon sadece metin anlamından degil, davranış paterninden de anlaşılır.
- Aynı mesajın tekrar edilmesi, kısa sürede yayilmasi ve author davranışi kritik sinyaldir.

Tek başına davranışsal skor da yeterli değildir.

Sebep:

- Bazı organik hesaplar yüksek frekanslı olabilir.
- Bazı manipülatife yakın metinler tekil gorunebilir.
- Semantik model metnin dilsel sinyalini yakalar.

Fusion bu iki katmanı birleştirir.

## Kod Yapısı

Public repo görünümü için çalışma kodu artık katmanlara ayrılmıştır:

```text
main.py
fusion_pipeline/
  config.py
  constants.py
  data_processing.py
  scoring.py
  artifacts.py
  inference.py
  pipeline.py
  legacy_impl.py
formula_scoring_pipeline.py
```

Rol dağılımı:

- `main.py`: tek giriş noktası
- `fusion_pipeline/config.py`: varsayılan config ve JSON override yükleme
- `fusion_pipeline/constants.py`: shared schema, version, and store constants
- `fusion_pipeline/data_processing.py`: veri temizleme, feature extraction, semantic preprocess
- `fusion_pipeline/scoring.py`: formula, weighting, final score mantığı
- `fusion_pipeline/artifacts.py`: SQLite store, manifest, QA tabloları, artefact validation
- `fusion_pipeline/inference.py`: tek mesaj inference
- `fusion_pipeline/pipeline.py`: build ve rescore orchestration
- `fusion_pipeline/legacy_impl.py`: compatibility layer for older imports
- `formula_scoring_pipeline.py`: eski notebook importları için compatibility shim

Aktif public path:

- `main.py`
- `fusion_pipeline/`
- `config.sample.json`
- `fusion.ipynb`

Arşivlenmiş eski deneyler:

- `archive/`

## Ana Dosyalar

```text
fusion.ipynb
```

Ana notebook. Config, model test, full build, rescore, QA tablolar, grafikler ve tek mesaj inference akışi buradadır.

```text
main.py
```

Notebook dışı çalıştırma için CLI entrypoint.

```text
formula_scoring_pipeline.py
```

Geriye dönük uyumluluk katmanı. Asıl kod `fusion_pipeline/` altına taşınmıştır.

```text
formulas.md
```

Skor mantığının formel açıklaması.

```text
data/datathonFINAL.parquet
```

Ana veri dosyası.

## Package-First Kullanım

Config template üret:

```bash
python main.py --mode write-config --output-config config.sample.json
```

Var olan artefact'ları doğrula:

```bash
python main.py --mode validate --config config.sample.json
```

Tek mesaj skorla:

```bash
python main.py --mode score-single --config config.sample.json --message "sample message"
```

Notebook isteyen akış için `fusion.ipynb` korunur. Public repo içinde desteklenen CLI yolu `main.py` üzerindendir.

## Çalışma Modları

`fusion.ipynb` içindeki ana seçim flag'i:

```python
"use_model_test_sample": True
```

### Model Test 200K

Bug temizlemek ve RoBERTa akışinin çalıştığını görmek için kullanılır.

```python
"use_model_test_sample": True
"model_test_rows": 200_000
```

Bu mod:

- ilk 200K satırdan ayrı bir parquet oluşturur
- in-memory pipeline çalıştırır
- SQLite batch store olusturmaz
- full build çıktılarini ezmez
- RoBERTa'yı açık tutar
- hızlı hata yakalamak içindir

Üretilen test input:

```text
data/fusion_model_test_input_200k.parquet
```

### Full Build

Final skor dosyalarını uretmek için kullanılır.

```python
"use_model_test_sample": False
```

Bu mod:

- tüm parquet dosyasını işler
- SQLite store oluşturur
- text cluster tablolarını oluşturur
- author skorlarını üretir
- message skorlarını üretir
- final score parquet dosyasını yazar
- manifest dosyası yazar

Ana çıktılar:

```text
data/fusion_batch_store.sqlite
data/fusion_author_scores.parquet
data/fusion_scored_messages.parquet
data/fusion_manifest.json
```

### Rescore From Existing Store

Sadece weight veya scoring threshold değiştiğinde kullanılır.

Bu mod:

- SQLite store'u yeniden kurmaz
- RoBERTa'yı yeniden çalıştırmaz
- mevcut store üzerinden skor dosyalarını tekrar yazar

Kullanma koşulu:

```python
"use_model_test_sample": False
```

## Config Ayarları

Ana config `fusion.ipynb` içindedir.

### Runtime

```python
"runtime": {
    "mode": "full",
    "use_model_test_sample": True,
    "model_test_rows": 200_000,
    "build_mode": "lean",
    "batch_size": 150_000,
    "max_batches": None,
    "sample_n_rows": None,
    "overwrite_outputs": True,
    "enable_progress_logs": True,
    "progress_every_batches": 10,
    "top_n_domain_context": 128,
    "author_batch_size": 10_000,
    "message_batch_size": 250_000,
}
```

Önemli alanlar:

- `use_model_test_sample`: `True` ise 200K test, `False` ise full build.
- `model_test_rows`: sample test satır sayısı.
- `batch_size`: full build batch boyutu.
- `author_batch_size`: author scoring batch boyutu.
- `message_batch_size`: message scoring batch boyutu.
- `overwrite_outputs`: mevcut çıktılari ezme davranışi.

### Semantic Adapter

```python
"semantic_adapter": {
    "enabled": True,
    "selected_model_key": "distil_bot_en",
    "models": {
        "distil_bot_en": {
            "model_name": "junaid1993/distilroberta-bot-detection",
            "supported_languages": ["en"],
        },
        "xlmr_base_multilingual": {
            "model_name": "FacebookAI/xlm-roberta-base",
            "supported_languages": "all",
        },
    },
    "unsupported_language_score": 0.50,
    "max_length": 128,
    "batch_size": 64,
    "device": "mps",
}
```

Önemli alanlar:

- `enabled`: RoBERTa açık/kapalı.
- `max_length`: tokenizer truncation limiti. `128` hızlı, `256` daha güvenli, `512` yavaş.
- `batch_size`: RoBERTa inference batch boyutu.
- `device`: M4 Pro için `mps`.

### Thresholds

Thresholdlar hard rule ve spam paternlerini belirler.

```python
"hard_bot_time_window_sec": 300
"hard_bot_repeat_threshold": 5
"hard_bot_multi_author_threshold": 2
"hard_bot_time_cluster_threshold": 3
"spam_repeat_threshold": 3
"spam_multi_author_threshold": 2
"spam_time_cluster_threshold": 3
```

Örnek:

- Aynı metin 5+ kez tekrar ederse,
- 2+ farklı author tarafından kullanılırsa,
- 300 saniyelik pencerede 3+ kez görülürse,
- hard bot cluster flag tetiklenir.

### Weights

```python
"dynamic_final_weighting": {
    "enabled": True,
    "min_confidence_weight": 0.20,
    "power": 2.0,
    "sigmoid_steepness": 8.0,
}

"neutral_score_policy": {
    "neutral_score": 0.50,
    "epsilon": 1e-6,
}

"behavioral_vs_semantic": {
    "behavioral": 0.45,
    "semantic": 0.55,
}
```

RoBERTa skoru final kararda daha yüksek ağırlık alır. Davranışsal katman korunur, fakat message-level davranış skoru düşük ağırlıklı yardımcı sinyal olarak kullanılır.

Dinamik final ağırlıklandırma açıksa `behavioral_score` ve `roberta_score` 0.5'e yakın kaldığında düşük güven alır. 0 veya 1 uçlarına yaklaştığında ilgili skor daha fazla söz hakkı kazanır.

Ek kural:

- `neutral_score_policy.neutral_score = 0.50` ise, herhangi bir birleşim noktasında `0.50` alan taraf anlamsız kabul edilir.
- `author_score = 0.50` ise `behavioral_score` hesabında author tarafı düşer, `message_score` kalır.
- `message_score = 0.50` ise `behavioral_score` hesabında message tarafı düşer, `author_score` kalır.
- `roberta_score = 0.50` ise final fusion içinde semantic taraf düşer, `behavioral_score` kalır.
- Her iki taraf da `0.50` ise birleşim sonucu da nötr `0.50` kalır.

## Çalıştırma Sırası

Notebook kernel:

```text
Python (.venv overfit-final)
```

Önce test:

1. `use_model_test_sample = True`
2. Config hücrelerini çalıştır
3. `Selected Pipeline Run` hücrelerini çalıştır
4. QA tablolarını kontrol et

Sonra full build:

1. `use_model_test_sample = False`
2. Config hücrelerini tekrar çalıştır
3. `Selected Pipeline Run` hücrelerini çalıştır
4. `Artefact Readiness Check` çalıştır
5. QA tablolarını ve grafiklerini kontrol et

Weight değişikliği sonrası:

1. `use_model_test_sample = False`
2. `Rescore From Existing Store` çalıştır
3. `Artefact Readiness Check` çalıştır

CLI karşılıkları:

1. Full build: `python main.py --mode build --config config.sample.json`
2. Rescore: `python main.py --mode rescore --config config.sample.json`
3. Validate: `python main.py --mode validate --config config.sample.json`

## Çıktılar

### Author Scores

```text
data/fusion_author_scores.parquet
```

Temel kolonlar:

- `author_hash`
- `author_score`
- `author_hard_hourly_flag`
- `max_posts_one_hour`
- `language_nunique`
- `theme_nunique`
- `sentiment_std`
- `median_interpost_sec`

### Scored Messages

```text
data/fusion_scored_messages.parquet
```

Temel kolonlar:

- `message_id`
- `author_hash`
- `author_type`
- `same_text_repeat_count`
- `same_text_unique_author_count`
- `same_text_time_window_count`
- `spam_pattern_flag`
- `hard_bot_cluster_flag`
- `author_score`
- `message_score`
- `behavioral_score`
- `roberta_score`
- `final_score`

### Manifest

```text
data/fusion_manifest.json
```

Manifest config hash, schema versiyonları ve çıktı uyumluluğu için kullanılır.

## Tek Mesaj Inference

Notebook sonunda `score_single_message` akışi vardır.

Girdi:

- mesaj metni
- dil
- url
- tarih
- author_hash
- keyword
- tema

Çıktı:

- `final_score`
- `behavioral_score`
- `roberta_score`
- flagler
- tekrar ve cluster sinyalleri

Bu kısım demo sırasında yeni mesaj skorlama için kullanılır.

## Yorumlama

Skor bandları:

```text
0.00 - 0.40: düşük risk
0.40 - 0.60: izlenebilir
0.60 - 0.70: orta risk
0.70 - 0.85: yüksek risk
0.85 - 1.00: çok yüksek risk
1.00       : hard rule tetiklenmiş risk
```

Bu skorlar kesin bot etiketi değildir. Skor, manipulasyon riski ve organiklikten sapma sinyali olarak okunmalıdır.

## Sınırlar

- Ground truth label yoktur.
- RoBERTa sadece desteklenen dillerde çalışır.
- Desteklenmeyen dillerde semantik skor nötr kabul edilir.
- Yüksek frekans her zaman bot demek değildir.
- Aynı metin tekrari kampanya, haber yayılımı veya spam olabilir.
- Final yorumda tablo ve örnekler birlikte incelenmelidir.

