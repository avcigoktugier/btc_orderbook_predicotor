# BTC/USDT Orderbook ML Tahmin Sistemi + Polymarket Bot

## 🎯 Ne Yapar?

Binance borsasından **Level 2 (Seviye 2) Orderbook** verilerini alarak, Bitcoin'in (BTC/USDT) fiyatının **5 dakika sonra** yükselip yükselmeyeceğini **makine öğrenmesi** ile tahmin eder ve **Polymarket** üzerindeki BTC 5-dakikalık Up/Down marketlerinde otomatik işlem yapar.

**Polymarket'in resolution kaynağıyla senkronize çalışır:**
- Resolution: [Chainlink BTC/USD Data Stream](https://data.chain.link/streams/btc-usd)
- On-chain okuma: [Polygon Mainnet Price Feed](https://data.chain.link/feeds/polygon/mainnet/btc-usd)
- Contract: `0xc907E116054Ad103354f2D350FD2514433D57F6f`

## 📐 Mimari

```
┌─────────────────────────────────────────────────────────────────┐
│         BTC/USDT Orderbook ML + Polymarket Bot v3               │
│                *** PENCERE-SENKRONİZE ***                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │ Binance  │───▶│ Feature  │───▶│ XGBoost/ │───▶│ Poly-    │ │
│  │ Orderbook│    │ Engine   │    │ RF Model │    │ market   │ │
│  │ (L2 20)  │    │ (28 feat)│    │ (tahmin) │    │ (işlem)  │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│       │                               │               │        │
│  REST + WebSocket              Tahmin: ▲/▼       BTC 5-dk      │
│                                Güven: %65+      Up/Down Bet    │
│                                                                 │
│  ┌──────────┐              ┌────────────────────────────┐      │
│  │ Chainlink│──────────────│ Pencere Senkronizasyonu     │      │
│  │ BTC/USD  │  Resolution  │ 12:00-12:05, 12:05-12:10.. │      │
│  │ (Polygon)│  Kaynağı     │ Açılış fiyatı Chainlink'ten │      │
│  └──────────┘              └────────────────────────────┘      │
│                                                                 │
│  Modlar: DRY-RUN (simülasyon) | LIVE (gerçek işlem)           │
└─────────────────────────────────────────────────────────────────┘
```

## 🗂️ Dosya Yapısı

```
btc_orderbook_predictor/
├── main.py              # Ana giriş noktası (veri toplama, eğitim, tahmin)
├── bot.py               # Polymarket otomatik işlem botu (pencere-senkronize)
├── config.py            # Tüm ayarlar (Binance + Polymarket + Chainlink)
├── chainlink_feed.py    # Chainlink BTC/USD on-chain fiyat okuyucu
├── data_collector.py    # Binance API veri toplama (REST + WebSocket)
├── features.py          # Özellik mühendisliği (28 feature)
├── model.py             # ML modelleri (XGBoost, Random Forest, Ensemble)
├── train.py             # Eğitim pipeline
├── live_predictor.py    # Canlı tahmin motoru (sadece tahmin, işlem yok)
├── polymarket_client.py # Polymarket CLOB API + pencere zamanlama
├── requirements.txt     # Python bağımlılıkları
├── .env.example         # Ortam değişkenleri şablonu
├── data/                # Toplanan veriler (CSV)
├── models/              # Eğitilmiş modeller (.joblib)
└── logs/                # Log dosyaları + işlem geçmişi
```

## ⚡ Hızlı Başlangıç

### 1. Gereksinimleri Kur

```bash
cd btc_orderbook_predictor
pip install -r requirements.txt
```

### 2. Ortam Değişkenleri

```bash
copy .env.example .env       # Windows
```

`.env` dosyasını düzenleyin (Polymarket Kurulumu bölümüne bakın).

### 3. Bağlantı Testi

```bash
python main.py test
```

### 4. Veri Topla (Minimum 30 dakika)

```bash
python main.py collect --duration 60
```

Durdurmak için: `Ctrl + C`

### 5. Model Eğit

```bash
python main.py train --model xgboost
```

### 6. Bot Çalıştır

```bash
# DRY-RUN (simülasyon)
python bot.py

# Gerçek işlem
python bot.py --live
```

---

## 🔗 Chainlink Senkronizasyonu

### Neden Chainlink?

Polymarket BTC 5-dk marketleri **Chainlink BTC/USD Data Stream** ile resolve edilir:
- Kaynak: https://data.chain.link/streams/btc-usd
- On-chain feed: [Polygon Mainnet](https://data.chain.link/feeds/polygon/mainnet/btc-usd)
- Contract: `0xc907E116054Ad103354f2D350FD2514433D57F6f`
- Deviation threshold: **%0.1** (çok hassas güncelleme)
- Heartbeat: ~27 saniye

### Bot Nasıl Senkronize Kalır?

1. **Pencere başlangıcını bekler** (örn. 12:05:00 ET)
2. **Chainlink'ten açılış fiyatını okur** (Polygon RPC üzerinden on-chain)
3. **ML tahminini yapar** (Binance orderbook verileriyle)
4. **İşlem kararını verir**: Tahmin UP → Up token al, DOWN → Down token al
5. **Sonraki pencereye geçer** (12:10:00 ET)

### Chainlink vs Binance Fiyat Farkı

Bot her iki kaynağı da izler ve loglar:
- **Chainlink**: Polymarket'in gerçek resolution kaynağı
- **Binance**: ML modelinin feature kaynağı (orderbook)
- **Spread**: İkisi arasındaki fark genellikle **<%0.05** (normal koşullarda)

```bash
# Fiyat kaynakları durumunu kontrol et
python bot.py --status
```

Çıktı:
```
── Fiyat Kaynakları ──
🔗 Chainlink:  $84,229.32  ← Polymarket resolution
📊 Binance:    $84,181.21  ← ML veri kaynağı
✅ Spread:     $48.11 (0.0571%)
```

---

## 🤖 Polymarket Bot Kullanımı

### Polymarket Kurulumu

1. **Private Key**: Polymarket > Cüzdan > Cash > ⋮ > Export Private Key
2. **Funder Address**: Polymarket cüzdan deposit adresi
3. `.env` dosyasını düzenleyin:

```bash
POLY_PRIVATE_KEY=0xSIZIN_PRIVATE_KEY
POLY_FUNDER_ADDRESS=0xSIZIN_ADRES
POLY_SIGNATURE_TYPE=1
POLY_BET_AMOUNT=5.0
```

### Bot Komutları

```bash
# Durum kontrolü (Polymarket + Chainlink + Fiyatlar)
python bot.py --status

# DRY-RUN modu (varsayılan, gerçek para harcanmaz)
python bot.py

# Farklı model ile
python bot.py --model ensemble

# İşlem tutarını değiştir
python bot.py --bet 10

# Süre belirle (30 dakika çalış)
python bot.py --duration 30

# GERÇEK İŞLEM MODU (dikkatli!)
python bot.py --live
```

### Bot İş Akışı (Pencere-Senkronize)

```
1. Bot başlar → ML modeli yüklenir
2. Chainlink BTC/USD bağlantısı kontrol edilir
3. Polymarket bağlantısı kontrol edilir
4. Orderbook stream başlar → veri birikiyor (60+ snapshot)
5. ═══════════════════════════════════════════════════
   PENCERE DÖNGÜSÜ (5 dakikada bir tekrarlanır):
   ─────────────────────────────────────────────────
   a. Sonraki 5-dk pencerenin başlamasını BEKLE
   b. Pencere açılınca Chainlink'ten AÇILIŞ FİYATI kaydet
   c. ML modeli ile yön tahmini yap (UP/DOWN)
   d. Güven ≥ %65 → Polymarket'te işlem AÇ
   e. Güven < %65 → İşlem yok, sonraki pencereyi bekle
   ═══════════════════════════════════════════════════
6. İşlem geçmişi logs/bot_trades.csv'ye kaydedilir
7. Durdurmak için: Ctrl + C
```

### Güvenlik Notları

- **DRY-RUN varsayılandır** — `--live` olmadan gerçek para harcanmaz
- `--live` modunda "EVET" onayı gerekir
- Pencere sonuna 60 saniye kala işlem açılmaz
- Aynı pencerede tekrar işlem açılmaz
- `POLY_MIN_CONFIDENCE = 0.65` — model %65'ten düşükse atlar
- `POLY_MAX_OPEN_POSITIONS = 2` — max 2 eş zamanlı pozisyon

### Opsiyonel: Özel Polygon RPC

Varsayılan ücretsiz public RPC yeterlidir. Daha hızlı Chainlink okuması için:

```bash
# .env dosyasına ekleyin
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY
```

---

## 📊 Hesaplanan Özellikler (28 Feature)

| Kategori | Özellikler |
|----------|-----------|
| **Temel** | Mid-Price, Spread, Spread (%) |
| **OBI** | depth=1, depth=5, depth=20 |
| **WAP** | VWAP Bid/Ask, LWAP Bid/Ask, VWAP vs Mid |
| **Hacim** | Total Volume, Volume Ratio, Top Concentration, Depth % |
| **Rolling** | Return, SMA, OBI Trend, Spread Trend, Volume Trend (5/10/20/50 pencere) |

## ⚙️ Konfigürasyon

`config.py` dosyasındaki önemli ayarlar:

```python
# Tahmin ufku
PREDICTION_HORIZON_MIN = 5          # 5 dakika sonrasını tahmin et

# Chainlink ayarları
CHAINLINK_BTC_USD_CONTRACT = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
WINDOW_ENTRY_OFFSET_SEC = 10        # Pencere başladıktan 10s sonra işlem aç
WINDOW_EXIT_BUFFER_SEC = 60         # Pencere bitişine 60s kala işlem yapma

# Polymarket işlem ayarları
POLY_BET_AMOUNT_USDC = 5.0          # Her işlem $5 USDC
POLY_MIN_CONFIDENCE = 0.65          # Bot için min %65 güven
POLY_DRY_RUN = True                 # True = simülasyon
```

## 🔧 Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| `Binance erişim yok` | `EXCHANGE_ID=binanceus` veya `bybit` ayarlayın |
| `Model dosyası bulunamadı` | `python main.py train` çalıştırın |
| `Chainlink erişilemedi` | `.env`'de `POLYGON_RPC_URL` ayarlayın |
| `Aktif market bulunamadı` | Polymarket'te BTC 5-dk market olmayabilir |
| `py-clob-client hatası` | `pip install py-clob-client --upgrade` |
| `Kimlik doğrulama başarısız` | `POLY_SIGNATURE_TYPE` (0 veya 1) kontrol edin |
| `Spread çok yüksek` | Chainlink-Binance farkı normalden büyükse dikkatli olun |

## ⚠️ Uyarılar

1. **Bu sistem yatırım tavsiyesi değildir.** Eğitim ve araştırma amaçlıdır.
2. **Geçmiş performans gelecek sonuçları garanti etmez.**
3. **DRY-RUN modunda kapsamlı test yapmadan gerçek para kullanmayın.**
4. Polymarket resolution kaynağı Chainlink'tir, Binance değil — küçük fiyat farkları sonucu etkileyebilir.
5. Model yeterli veri ile eğitilmelidir (minimum 30-60 dakika, ideal 2-4 saat).
