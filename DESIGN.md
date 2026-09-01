# Tasarım

Fusion, etiketsiz sosyal medya akışında manipülasyon riskini `0–1` aralığında sıralar. Denetimli sınıf tahmini değildir.

## Varsayım

Metin tek başına yetmez. Organik hesap yüksek frekanslı olabilir; manipülatif metin tekil görünebilir. Asıl kanıt davranışsal bağlamdır: tempo, tekrar, çok yazarlı kopya, kısa pencerede küme. Semantik model bunu destekler, ezmez.

## Sinyaller

### Yazar

`author_score`, bileşen ağırlıklarıyla (`activity`, `timing`, `repetition`, `diversity`) birleşir. Her bileşen, ilgili ham özelliklerin yüzde birlik ölçeklemesi veya log-cezasıdır.

Saatlik patlama eşiği aşılırsa `author_hard_hourly_flag = 1`.

### Mesaj

`message_score` aynı metin tekrarı, spam bayrağı, hashtag yoğunluğu, token tekrarı, uzun metin ve anahtar kelime sinyalini birleştirir. Hashtag sayısı `> 5` veya ünlem `> 10` ise mesaj skoru `1.0` olur.

### Semantik

Sequence-classification checkpoint açıksa `roberta_score` üretir. Desteklenmeyen dilde `unsupported_language_score` (varsayılan `0.50`).

## Dinamik birleşim

Nötr nokta \(s = 0.5\). Güven:

\[
c(s) = w_{\min} + (1 - w_{\min})\,\bigl(2|s-0.5|\bigr)^{p}
\]

Varsayılan \(w_{\min}=0.20\), \(p=2\):

\[
c(s) = 0.20 + 3.20\,(s-0.5)^{2}
\]

Ham ağırlıklar:

\[
\tilde{w}_b = \pi_b\, c(s_b),\qquad \tilde{w}_r = \pi_r\, c(s_r)
\]

Sigmoid kapı (\(k=8\)):

\[
w_r = \sigma\bigl(k(\tilde{w}_r - \tilde{w}_b)\bigr),\qquad w_b = 1 - w_r
\]

Bir taraf nötrse (`|s - 0.5| \le \varepsilon`) o tarafın ağırlığı sıfırlanır, diğeri 1'e çekilir. İkisi de nötrse eşit paylaşım.

Anonim yazarda davranış skoru yalnızca `message_score` olur.

## Hard kural

Aşağıdakilerden biri varsa `final_score = 1.0`:

- `author_hard_hourly_flag`
- `hard_bot_cluster_flag` (tekrar + çok yazar + kısa pencere)
- `hard_same_text_repeat_flag`

## Artefaktlar

| Dosya | Rol |
| :--- | :--- |
| `fusion_batch_store.sqlite` | temizlenmiş mesaj, metin kümeleri, yazar skorları |
| `fusion_author_scores.parquet` | yazar düzeyi skor |
| `fusion_scored_messages.parquet` | mesaj düzeyi nihai skor ve açıklama kolonları |
| `fusion_manifest.json` | şema sürümü ve yol kaydı |

`rescore` store'u yeniden kurmaz; ağırlık/eşik değişince semantik çıkarımı tekrarlamaz.

## Sınır

Fusion kimlik, niyet veya otomasyon kanıtlamaz. Yüksek skor, örüntünün manipülasyon riskiyle daha uyumlu göründüğü anlamına gelir.
