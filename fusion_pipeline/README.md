# fusion_pipeline

Fusion'ın çekirdek paketi. Denetimli bot sınıflandırıcı değil; etiketsiz akışta açıklanabilir manipülasyon-risk skoru üretir.

Kamu yüzeyi kök [README.md](../README.md) ve [DESIGN.md](../DESIGN.md).

## Modüller

| Dosya | İş |
| :--- | :--- |
| `config.py` | varsayılan config, JSON birleştirme |
| `constants.py` | şema sürümü, zorunlu kolonlar, SQLite tanımları |
| `data_processing.py` | temizlik, özellik çıkarımı, semantik ön işlem |
| `scoring.py` | ölçekleme, güven, nihai ağırlık, hard kural |
| `artifacts.py` | SQLite store, manifest, QA |
| `inference.py` | tek mesaj skorlama |
| `pipeline.py` | `build` / `rescore` orkestrasyon |
| `legacy_impl.py` | eski import uyumu |

Yeni kod doğrudan bu modülleri kullanır. `formula_scoring_pipeline.py` yalnızca geriye dönük shim'dir.
