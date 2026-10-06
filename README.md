# Fibonacci Ampirik Çalışma

Hisselerde Fibonacci düzeltme (0.236 / 0.382 / 0.5 / 0.618 / 0.65 / 0.786) ve uzantı
(1 / 1.272 / 1.618 / 2 / 2.618) seviyelerinin gerçekten "özel" olup olmadığını test eder.

Zaman dilimleri: **M, 2W, W, 3D, D** — parametreler aylık ve haftalık için ayarlı.

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```bash
python fib_study.py --tf M,W              # aylık + haftalık
python fib_study.py --sweep --tf M,W      # parametre sağlamlık taraması
python fib_study.py                       # tüm zaman dilimleri
python fib_study.py --synthetic           # internetsiz test (rastgele veri)
```

Kendi hisse listen için aynı klasöre `tickers.txt` koy (her satıra bir sembol).
Yoksa S&P 500 kullanılır (bugünkü üyeler → survivorship bias).

## Çıktılar (`fib_out/`)

| Dosya | İçerik |
|---|---|
| `summary.md` | Tüm tablolar: kümelenme, sonuç tabloları, hacim filtresi, yön ayrımı |
| `events.csv` | Her itki-düzeltme olayı tek satır |
| `{TF}_depth_hazard.png` | Derinlik histogramı + seviyede dönme olasılığı |
| `{TF}_hazard.csv` | Hazard verisi |
| `sweep.csv` | `--sweep` sonuçları |

## Tablolar nasıl okunur

- **Kümelenme:** `excess_ratio > 1` ve `pctile_vs_random` yüksekse, düzeltme dipleri o seviyede komşu bölgelere göre fazla birikiyor demektir. Tek bir zaman diliminde çıkan sonuç gürültü olabilir; M ve W'de tutarlılık aranmalı.
- **İşlenebilir (trd):** Giriş, dip ZigZag ile onaylandıktan sonra. Gerçek trade'e yakın.
- **Tanımlayıcı (desc):** Dipten itibaren ölçer (hindsight) → iyimser.
