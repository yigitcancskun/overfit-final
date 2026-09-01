# Testler

Fusion skorlama hattı için pytest.

| Dosya | Kapsam |
| :--- | :--- |
| `conftest.py` | config ve SQLite fixture |
| `test_config.py` | yükleme, deep merge, şablon yazma |
| `test_data_processing.py` | normalizasyon, anahtar kelime, domain, batch temizlik |
| `test_scoring.py` | ölçekleme, nötr ağırlık, hard kural |
| `test_inference.py` | tek mesaj girdi şekli ve çıktı sözleşmesi |

```bash
pytest -q tests
```

Semantik adaptör testlerde kapalıdır. Üretim parquet'ine ihtiyaç yoktur.
