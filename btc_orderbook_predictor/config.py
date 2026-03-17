"""
============================================================
BTC/USDT Orderbook ML Prediction System - Konfigürasyon
============================================================
Tüm sistem parametreleri bu dosyada merkezi olarak yönetilir.
.env dosyası veya ortam değişkenleri ile override edilebilir.
============================================================
"""

import os
from dotenv import load_dotenv

# .env dosyasını yükle (varsa)
load_dotenv()

# ============================================================
# BORSA AYARLARI (Exchange Settings)
# ============================================================
# Desteklenen borsalar: "binance", "binanceus", "bybit", "okx"
# Binance kısıtlı bölgelerde "binanceus" veya "bybit" kullanın.
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")  # Borsa adı (ccxt formatı)
SYMBOL = "BTC/USDT"                              # İşlem çifti
BINANCE_WS_SYMBOL = "btcusdt"                    # WebSocket format (küçük harf, / yok)

# Binance API Anahtarları (opsiyonel - public endpoint'ler için gerekli değil)
# Canlı ticaret yapmak isterseniz .env dosyasına ekleyin
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# ============================================================
# ORDERBOOK AYARLARI (Orderbook Settings)
# ============================================================
ORDERBOOK_DEPTH = 20                             # Kaç seviye bid/ask alınacak (5, 10, 20)
WS_UPDATE_SPEED = "100ms"                        # WebSocket güncelleme hızı (100ms veya 1000ms)

# Binance WebSocket URL'leri
WS_BASE_URL = "wss://stream.binance.com:9443"
WS_STREAM_URL = f"{WS_BASE_URL}/ws/{BINANCE_WS_SYMBOL}@depth{ORDERBOOK_DEPTH}@{WS_UPDATE_SPEED}"

# Binance REST API
REST_BASE_URL = "https://api.binance.com"
REST_DEPTH_URL = f"{REST_BASE_URL}/api/v3/depth"

# ============================================================
# VERİ TOPLAMA AYARLARI (Data Collection Settings)
# ============================================================
SNAPSHOT_INTERVAL_SEC = 1.0                      # REST API ile kaç saniyede bir snapshot al
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")   # Veri dizini
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models") # Model dizini
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")     # Log dizini

# Veri dosyası adlandırması
RAW_DATA_FILE = os.path.join(DATA_DIR, "orderbook_raw.csv")
FEATURES_FILE = os.path.join(DATA_DIR, "orderbook_features.csv")
TRAINING_DATA_FILE = os.path.join(DATA_DIR, "training_data.csv")

# ============================================================
# ÖZELLİK MÜHENDİSLİĞİ (Feature Engineering Settings)
# ============================================================
# Tahmin penceresi: Kaç dakika sonrasını tahmin ediyoruz?
PREDICTION_HORIZON_MIN = 5                       # 5 dakika sonrası

# Hedef fiyat belirleme yöntemi
# "mid_price": Mevcut mid-price'ı hedef olarak kullan
# "custom": Sabit bir hedef fiyat belirle
TARGET_PRICE_MODE = "mid_price"
CUSTOM_TARGET_PRICE = None                       # TARGET_PRICE_MODE="custom" ise buraya yaz

# Rolling window boyutları (feature hesaplamaları için)
ROLLING_WINDOWS = [5, 10, 20, 50]                # Kaç snapshot geriye bakılacak

# ============================================================
# MODEL AYARLARI (Model Settings)
# ============================================================
# Ana model: XGBoost
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,                            # L1 regularization
    "reg_lambda": 1.0,                           # L2 regularization
    "scale_pos_weight": 1,                       # Dengesiz sınıflar için ayarla
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}

# Yedek model: Random Forest
RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

# Eğitim / Test bölme oranı
TEST_SIZE = 0.2                                  # %20 test
VALIDATION_SIZE = 0.15                           # %15 validation (early stopping için)

# Model kaydetme
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.joblib")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")

# ============================================================
# POLYMARKET AYARLARI (Polymarket Trading Settings)
# ============================================================
# Private key'inizi Polymarket'ten export edin:
#   Polymarket > Cüzdan > Cash > ⋮ menü > Export Private Key
# Veya MetaMask'tan alın.
POLY_PRIVATE_KEY = os.getenv("POLY_PRIVATE_KEY", "")
POLY_FUNDER_ADDRESS = os.getenv("POLY_FUNDER_ADDRESS", "")  # Deposit adresiniz

# Giriş yönteminize göre ayarlayın:
#   0 = MetaMask / EOA cüzdan
#   1 = Email / Magic wallet (Polymarket doğrudan giriş)
POLY_SIGNATURE_TYPE = int(os.getenv("POLY_SIGNATURE_TYPE", "1"))

POLY_CLOB_API = "https://clob.polymarket.com"
POLY_GAMMA_API = "https://gamma-api.polymarket.com"
POLY_CHAIN_ID = 137  # Polygon

# İşlem Parametreleri
POLY_BET_AMOUNT_USDC = float(os.getenv("POLY_BET_AMOUNT", "5.0"))  # Her işlemde kaç USDC
POLY_MAX_OPEN_POSITIONS = 2          # Aynı anda max açık pozisyon
POLY_MIN_CONFIDENCE = 0.65           # Minimum model güven eşiği (%65)
POLY_DRY_RUN = True                  # True = Gerçek işlem yapmaz, sadece simüle eder

# ============================================================
# CHAINLINK AYARLARI (Polymarket Resolution Kaynağı)
# ============================================================
# Polymarket BTC 5-dk marketleri Chainlink BTC/USD Data Stream ile
# resolve edilir. Bot aynı kaynaktan fiyat okuyarak senkronize kalır.
# Kaynak: https://data.chain.link/feeds/polygon/mainnet/btc-usd
CHAINLINK_BTC_USD_CONTRACT = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
CHAINLINK_DECIMALS = 8

# Polygon RPC endpoint'leri (öncelik sırasına göre)
POLYGON_RPC_ENDPOINTS = [
    "https://polygon-bor-rpc.publicnode.com",
    "https://polygon-rpc.com",
    "https://rpc-mainnet.matic.quiknode.pro",
    "https://polygon.drpc.org",
]

# Özel Polygon RPC kullanmak isterseniz .env'de tanımlayın
# (Alchemy, Infura, QuickNode gibi servislerden alabilirsiniz)
CUSTOM_POLYGON_RPC = os.getenv("POLYGON_RPC_URL", "")
if CUSTOM_POLYGON_RPC:
    POLYGON_RPC_ENDPOINTS.insert(0, CUSTOM_POLYGON_RPC)

# Pencere zamanlama ayarları
WINDOW_ENTRY_OFFSET_SEC = 10     # Pencere başladıktan kaç sn sonra işlem aç
WINDOW_EXIT_BUFFER_SEC = 60      # Pencere bitişine kaç sn kala işlem yapma
CHAINLINK_MAX_AGE_SEC = 120      # Chainlink verisi max kaç sn eski olabilir

# ============================================================
# CANLI TAHMİN AYARLARI (Live Prediction Settings)
# ============================================================
LIVE_PREDICTION_INTERVAL_SEC = 10                # Her kaç saniyede bir tahmin yap
MIN_SAMPLES_FOR_FEATURES = 60                    # Feature hesaplamak için minimum snapshot
CONFIDENCE_THRESHOLD = 0.6                       # Tahmin güven eşiği (%60+)

# ============================================================
# LOGLAMA (Logging Settings)
# ============================================================
LOG_LEVEL = "INFO"                               # DEBUG, INFO, WARNING, ERROR
LOG_FILE = os.path.join(LOG_DIR, "predictor.log")

# ============================================================
# Dizinlerin varlığını garanti et
# ============================================================
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)
