# BTC/USDT Orderbook ML + Polymarket Otomatik İşlem Botu
!!! Bu bot geliştirme aşamasındadır, yatırım tavsiyesi içermez!!!
## Kapsamlı Kurulum ve Kullanım Rehberi

---

## Sistem Özeti

Bu bot, Binance Level 2 orderbook verileri üzerinden XGBoost/Random Forest ML modeli ile BTC fiyat yönü tahmini yapar ve Polymarket'teki 5 dakikalık BTC Up/Down marketlerinde otomatik işlem açar.

**Veri Akışı:**
```
Binance Orderbook L2 → 28 Feature Mühendisliği → XGBoost ML Tahmin
                                                        ↓
Chainlink BTC/USD (Polygon On-Chain) → Pencere Açılış Fiyatı
                                                        ↓
Polymarket CLOB API → Otomatik Up/Down İşlem
```

**Pencere-Senkronize Çalışma:**
- Polymarket 5-dk marketleri sabit pencerelerle çalışır (12:00-12:05, 12:05-12:10, ...)
- Bot, her pencere başında Chainlink'ten açılış fiyatını kaydeder
- ML modeli tahmin yapar → güven ≥%65 ise işlem açar
- Polymarket, Chainlink BTC/USD ile resolve eder → bot aynı kaynağı kullanır

---

## Dosya Yapısı

```
D:\btc_orderbook_predictor\
│
├── config.py              # Tüm ayarlar (Binance, Polymarket, Chainlink, ML)
├── chainlink_feed.py      # Chainlink BTC/USD Polygon on-chain fiyat okuyucu
├── data_collector.py      # REST + WebSocket orderbook veri toplama
├── features.py            # 28 özellik: OBI, Spread, Mid-Price, VWAP, rolling stats
├── model.py               # XGBoost, Random Forest, Ensemble modeller
├── train.py               # Eğitim pipeline'ı
├── live_predictor.py      # Bağımsız canlı tahmin motoru (işlem yapmaz)
├── main.py                # CLI giriş noktası (test/collect/train/predict/demo)
├── polymarket_client.py   # Polymarket CLOB API + pencere zamanlaması
├── bot.py                 # Ana bot: ML + Polymarket pencere-senkronize işlem
├── requirements.txt       # Python bağımlılıkları
├── .env.example           # Ortam değişkenleri şablonu
├── .env                   # Sizin gerçek ayarlarınız (oluşturmanız gerekiyor)
│
├── data/                  # Toplanan veriler (otomatik oluşur)
├── models/                # Eğitilmiş ML modelleri (otomatik oluşur)
└── logs/                  # Log dosyaları (otomatik oluşur)
```

---

## ADIM 1: Python Kurulumu (Windows)

### 1.1 Python Kontrol
Komut İstemi (CMD) açın:
```cmd
python --version
```

Eğer "python bulunamadı" hatası alıyorsanız, tam yolu kullanın:
```cmd
C:\Users\avcig\AppData\Local\Python\pythoncore-3.14-64\python.exe --version
```

### 1.2 PATH Sorunu Çözümü
Python PATH'e eklenmemişse her komutta tam yol kullanmanız gerekir.
Kalıcı çözüm:
1. Windows Arama → "Ortam Değişkenleri" / "Environment Variables"
2. "Kullanıcı değişkenleri" → Path → Düzenle
3. Bu iki satırı ekleyin:
   ```
   C:\Users\avcig\AppData\Local\Python\pythoncore-3.14-64\
   C:\Users\avcig\AppData\Local\Python\pythoncore-3.14-64\Scripts\
   ```
4. CMD'yi kapatıp yeniden açın

---

## ADIM 2: Proje Dosyalarını Kur

### 2.1 Proje Klasörü
ZIP dosyasını `D:\btc_orderbook_predictor\` konumuna çıkarın.

### 2.2 CMD ile Klasöre Git
```cmd
cd /d D:\btc_orderbook_predictor
```

### 2.3 Bağımlılıkları Kur
```cmd
pip install -r requirements.txt
```

Eğer `pip` bulunamıyorsa:
```cmd
C:\Users\avcig\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pip install -r requirements.txt
```

> **Not:** `WARNING: The script ... is installed in '...\Scripts' which is not on PATH.` uyarısı görürseniz sorun yoktur, kurulum başarılıdır.

---

## ADIM 3: .env Dosyasını Ayarla

### 3.1 Şablonu Kopyala
```cmd
copy .env.example .env
```

### 3.2 .env Dosyasını Düzenle
Notepad veya VS Code ile `.env` dosyasını açın:
```cmd
notepad .env
```

### 3.3 .env İçeriği

```env
# --- Binance (opsiyonel - orderbook okuma için key gerekmez) ---
BINANCE_API_KEY=
BINANCE_API_SECRET=

# --- Polymarket Ayarları ---
POLY_PRIVATE_KEY=0xBURAYA_PRIVATE_KEY_YAZIN
POLY_FUNDER_ADDRESS=0xBURAYA_FUNDER_ADRES_YAZIN
POLY_SIGNATURE_TYPE=1
POLY_BET_AMOUNT=5.0

# --- Borsa Ayarı ---
# EXCHANGE_ID=binance
```

### 3.4 Polymarket Private Key Nasıl Alınır
1. [polymarket.com](https://polymarket.com) → Giriş yapın
2. Sağ üst → Cüzdan (Wallet) simgesi
3. "Cash" sekmesi → ⋮ (üç nokta menü)
4. "Export Private Key" → Kopyalayın
5. `.env` dosyasında `POLY_PRIVATE_KEY=` satırına yapıştırın

### 3.5 Funder Address Nedir
Polymarket'e USDC yatırdığınız Polygon adresi. Cüzdan sayfanızda görünen `0x...` adresiniz.

### 3.6 Signature Type
- **1** = Email/Magic wallet ile giriş yaptıysanız (Polymarket'e email ile)
- **0** = MetaMask veya harici EOA cüzdan kullandıysanız

---

## ADIM 4: Bağlantı Testi

```cmd
python main.py test
```

**Beklenen Çıktı:**
```
  ✅ Bağlantı BAŞARILI!
  Best Bid:    $74,200.00
  Best Ask:    $74,201.50
  Mid-Price:   $74,200.75
  Spread:      $1.50 (0.0020%)

  📊 Feature Engineering Testi:
  OBI (tüm):    0.1234
  OBI (top 5):  0.0987
  VWAP Mid:     $74,200.50
  Toplam 28 özellik hesaplandı.
  Tüm sistemler çalışıyor! ✅
```

> **Binance Erişim Sorunu:** Türkiye'den Binance global erişilemiyorsa, bot otomatik olarak `binanceus` API'sine düşer. Eğer o da çalışmazsa `.env` dosyasında `EXCHANGE_ID=bybit` yazın.

---

## ADIM 5: Veri Topla (Minimum 30 Dakika)

### 5.1 Veri Toplama Başlat
```cmd
python main.py collect --duration 60
```

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--duration` | Toplama süresi (dakika) | 60 |
| `--interval` | Snapshot aralığı (saniye) | 1.0 |

**Önerilen:** En az 60 dakika toplayın. Ne kadar uzun → model o kadar iyi.

### 5.2 Veri Toplamayı Durdurmak
`Ctrl + C` tuşuna basın. O ana kadar toplanan veriler kaydedilir.

### 5.3 Toplanan Veri Nereye Kaydedilir
```
D:\btc_orderbook_predictor\data\orderbook_raw.csv
```

---

## ADIM 6: ML Modelini Eğit

### 6.1 XGBoost ile Eğit (Önerilen)
```cmd
python main.py train --model xgboost --cv
```

### 6.2 Diğer Model Seçenekleri
```cmd
# Random Forest
python main.py train --model random_forest --cv

# Ensemble (XGBoost + Random Forest birleşik)
python main.py train --model ensemble --cv
```

### 6.3 Yeni Veri Toplayarak Doğrudan Eğit
```cmd
python main.py train --model xgboost --cv --collect 60
```

| Parametre | Açıklama |
|-----------|----------|
| `--model` | xgboost, random_forest, ensemble |
| `--cv` | Cross-validation uygula (önerilir) |
| `--collect N` | Eğitim öncesi N dakika veri topla |
| `--csv dosya.csv` | Belirli CSV dosyasından eğit |

### 6.4 Minimum Veri Gereksinimleri
- Feature hesaplama: en az 300 snapshot (~5 dakika)
- Eğitim: en az 300 feature satırı
- İdeal: 3.600+ snapshot (60+ dakika)

---

## ADIM 7: Seçenek A — Sadece Tahmin (İşlem Yapmadan)

ML modelinin tahminlerini görmek istiyorsanız (Polymarket'e bağlamadan):

```cmd
python main.py predict
```

```cmd
# Farklı model ile
python main.py predict --model ensemble

# REST modu (WebSocket sorun yaparsa)
python main.py predict --rest

# 30 dakika çalıştır
python main.py predict --duration 30
```

Bu mod sadece tahmin gösterir, hiçbir işlem yapmaz.

---

## ADIM 8: Seçenek B — Polymarket Bot (Otomatik İşlem)

### 8.1 Önce Durum Kontrolü
```cmd
python bot.py --status
```

Bu komut şunları gösterir:
- Gamma API bağlantısı ✅/❌
- Kimlik doğrulama durumu
- Chainlink BTC/USD fiyatı (Polymarket resolution kaynağı)
- Binance fiyatı (ML veri kaynağı)
- Chainlink-Binance spread'i
- Aktif BTC 5-dk market sayısı
- Mevcut pencere bilgisi

### 8.2 Simülasyon Modu (Varsayılan — Güvenli)
```cmd
python bot.py
```

- Gerçek veri toplar, gerçek tahmin yapar
- Ama işlem **simüle** eder (gerçek para harcanmaz)
- İlk kez kullanırken **kesinlikle bu modda** başlayın

### 8.3 Gerçek İşlem Modu
```cmd
python bot.py --live
```

- Gerçek USDC harcar
- "EVET" yazarak onaylamanız gerekir
- Her işlem varsayılan $5 USDC

### 8.4 Bot Parametreleri
```cmd
# Farklı bahis tutarı
python bot.py --bet 10

# Ensemble model ile
python bot.py --model ensemble

# 2 saat çalıştır
python bot.py --duration 120

# Hepsi birden (gerçek işlem, $10, ensemble, 2 saat)
python bot.py --live --bet 10 --model ensemble --duration 120
```

### 8.5 Bot Nasıl Çalışır (Adım Adım)
1. ML modeli yüklenir
2. Chainlink BTC/USD bağlantısı test edilir
3. Polymarket API kontrol edilir
4. Orderbook WebSocket stream başlatılır
5. Minimum 60 snapshot biriktirilir (~1 dakika)
6. **Sonraki 5-dk pencere beklenir**
7. Pencere açılınca → Chainlink'ten açılış fiyatı kaydedilir
8. ML modeli tahmin yapar (UP / DOWN)
9. Güven ≥ %65 ise → Polymarket'te işlem açılır
10. Sonraki pencereye geçilir → tekrarla

### 8.6 Botu Durdurmak
`Ctrl + C` — Bot düzgün kapanır ve özet gösterir.

---

## ADIM 9: Hızlı Demo (Tek Komutla Deneme)

Her şeyi sıfırdan denemek için:
```cmd
python main.py demo --duration 10
```

Bu komut sırasıyla: 10 dk veri toplar → model eğitir → 5 dk canlı tahmin yapar.

---

## Önemli Ayarlar (config.py)

Gerekirse `config.py` dosyasından değiştirebilirsiniz:

| Ayar | Varsayılan | Açıklama |
|------|-----------|----------|
| `PREDICTION_HORIZON_MIN` | 5 | Kaç dakika sonrasını tahmin et |
| `ORDERBOOK_DEPTH` | 20 | Orderbook derinliği (5, 10, 20) |
| `POLY_MIN_CONFIDENCE` | 0.65 | Minimum ML güven eşiği (%65) |
| `POLY_BET_AMOUNT_USDC` | 5.0 | İşlem başına USDC |
| `POLY_MAX_OPEN_POSITIONS` | 2 | Maksimum eşzamanlı açık pozisyon |
| `POLY_DRY_RUN` | True | True = simülasyon, False = gerçek |
| `WINDOW_ENTRY_OFFSET_SEC` | 10 | Pencere başladıktan kaç sn sonra işlem aç |
| `WINDOW_EXIT_BUFFER_SEC` | 60 | Pencere bitişine kaç sn kala işlem yapma |
| `CHAINLINK_MAX_AGE_SEC` | 120 | Chainlink verisi max eski olabilir (sn) |

---

## Teknik Detaylar

### Chainlink BTC/USD (Polygon Mainnet)
- Contract: `0xc907E116054Ad103354f2D350FD2514433D57F6f`
- 8 ondalık basamak
- Deviation Threshold: %0.1
- Heartbeat: ~27 saniye
- RPC: `https://polygon-bor-rpc.publicnode.com` (ücretsiz, ek hesap gereksiz)

### ML Özellikleri (28 Feature)
- OBI (Order Book Imbalance) — farklı derinliklerde
- Spread (mutlak + yüzdesel)
- Mid-Price
- VWAP (Volume Weighted Average Price)
- Bid/Ask hacim oranları
- Depth metrikleri
- Rolling istatistikler (5, 10, 20, 50 pencere)

### Polymarket Entegrasyonu
- API: Gamma API (market keşfi) + CLOB API (işlem)
- py-clob-client SDK ile FOK (Fill or Kill) market order
- 5-dk pencere senkronizasyonu (Polymarket'in kendi zamanlaması)

---

## Optimal Kullanım Akışı (Özet)

```
CMD aç → cd /d D:\btc_orderbook_predictor

1. pip install -r requirements.txt          ← İlk kez
2. copy .env.example .env                   ← İlk kez, sonra düzenle
3. python main.py test                      ← Bağlantı kontrolü
4. python main.py collect --duration 60     ← 60 dk veri topla
5. python main.py train --model xgboost --cv ← Model eğit
6. python bot.py --status                   ← Polymarket + Chainlink kontrol
7. python bot.py                            ← Simülasyon modu başlat
8. python bot.py --live                     ← Gerçek işlem (hazır olunca)
```

---

## Sık Karşılaşılan Sorunlar

| Sorun | Çözüm |
|-------|-------|
| `python bulunamadı` | Tam yol kullanın veya PATH'e ekleyin |
| `pip WARNING ... not on PATH` | Sorun değil, kurulum başarılıdır |
| `451 Unavailable For Legal Reasons` | Binance kısıtlaması → otomatik binanceus'a düşer |
| `ZeroDivisionError` eğitimde | Yeterli veri yok → en az 30 dk toplayın |
| `Model bulunamadı` | Önce eğitin: `python main.py train` |
| `Chainlink erişilemedi` | Bot otomatik Binance fiyatına düşer |
| `py-clob-client hata` | `pip install py-clob-client` |
| Polymarket auth hatası | `.env` dosyasında private key doğru mu? |
