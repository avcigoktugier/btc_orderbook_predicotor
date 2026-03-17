"""
============================================================
BTC/USDT Orderbook ML Prediction System - Özellik Mühendisliği
============================================================
Ham orderbook verilerinden modelin eğitilmesi için anlamlı
özellikler (features) çıkarır.

Hesaplanan Özellikler:
  1. Mid-Price (Orta Fiyat)
  2. Spread (Alış-Satış Farkı) - mutlak ve yüzdesel
  3. Order Book Imbalance (OBI - Emir Defteri Dengesizliği)
  4. Weighted Average Price (WAP - Ağırlıklı Ortalama Fiyat)
  5. Hacim Metrikleri (toplam bid/ask hacim, hacim oranları)
  6. Derinlik Metrikleri (fiyat derinliği, hacim yoğunluğu)
  7. Rolling (Hareketli) İstatistikler
  8. Hedef Değişken (Target Variable)
============================================================
"""

import logging
from typing import List, Optional

import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)


# ============================================================
# 1. TEK SNAPSHOT İÇİN ÖZELLİK HESAPLAYICILAR
# ============================================================

def compute_mid_price(row: pd.Series) -> float:
    """
    Mid-Price (Orta Fiyat) hesapla.
    
    Formula: mid_price = (best_bid + best_ask) / 2
    
    Piyasanın "gerçek" fiyatına en yakın tahmindir.
    """
    return (row["bid_price_0"] + row["ask_price_0"]) / 2


def compute_spread(row: pd.Series) -> dict:
    """
    Spread (Alış-Satış Farkı) metrikleri hesapla.
    
    Spread dar ise: Yüksek likidite, düşük volatilite
    Spread geniş ise: Düşük likidite, yüksek volatilite
    """
    best_bid = row["bid_price_0"]
    best_ask = row["ask_price_0"]
    mid = (best_bid + best_ask) / 2
    
    spread_abs = best_ask - best_bid           # Mutlak spread ($)
    spread_pct = (spread_abs / mid) * 100      # Yüzdesel spread (%)
    spread_bps = spread_pct * 100              # Basis points (bps)
    
    return {
        "spread_abs": spread_abs,
        "spread_pct": spread_pct,
        "spread_bps": spread_bps,
    }


def compute_order_book_imbalance(row: pd.Series, levels: int = None) -> dict:
    """
    Order Book Imbalance (OBI - Emir Defteri Dengesizliği) hesapla.
    
    Formula: OBI = (Σ bid_volume - Σ ask_volume) / (Σ bid_volume + Σ ask_volume)
    
    OBI > 0: Alıcılar baskın (fiyat yükselme eğilimi)
    OBI < 0: Satıcılar baskın (fiyat düşme eğilimi)
    OBI = 0: Dengeli piyasa
    
    Ayrıca farklı derinlik seviyelerinde OBI hesaplar.
    """
    if levels is None:
        levels = config.ORDERBOOK_DEPTH
    
    result = {}
    
    # Farklı derinlik seviyelerinde OBI
    for depth in [1, 5, 10, levels]:
        if depth > levels:
            continue
            
        bid_vol = sum(row.get(f"bid_volume_{i}", 0) for i in range(depth))
        ask_vol = sum(row.get(f"ask_volume_{i}", 0) for i in range(depth))
        
        total_vol = bid_vol + ask_vol
        if total_vol > 0:
            obi = (bid_vol - ask_vol) / total_vol
        else:
            obi = 0.0
        
        result[f"obi_depth_{depth}"] = obi
    
    # Ana OBI (tüm seviyeler)
    result["obi"] = result.get(f"obi_depth_{levels}", 0.0)
    
    return result


def compute_weighted_average_price(row: pd.Series, levels: int = None) -> dict:
    """
    Ağırlıklı Ortalama Fiyat (WAP) hesapla.
    
    İki farklı ağırlıklandırma yöntemi:
    
    1) Volume-Weighted: Hacme göre ağırlıklı
       WAP = Σ(price_i × volume_i) / Σ(volume_i)
    
    2) Level-Weighted: Seviyeye göre azalan ağırlık (1/i)
       En iyi fiyata daha fazla ağırlık verir
    """
    if levels is None:
        levels = config.ORDERBOOK_DEPTH
    
    # --- Volume-Weighted Average Price ---
    bid_price_vol_sum = 0.0
    bid_vol_sum = 0.0
    ask_price_vol_sum = 0.0
    ask_vol_sum = 0.0
    
    for i in range(levels):
        bp = row.get(f"bid_price_{i}", 0)
        bv = row.get(f"bid_volume_{i}", 0)
        ap = row.get(f"ask_price_{i}", 0)
        av = row.get(f"ask_volume_{i}", 0)
        
        if pd.notna(bp) and pd.notna(bv):
            bid_price_vol_sum += bp * bv
            bid_vol_sum += bv
        if pd.notna(ap) and pd.notna(av):
            ask_price_vol_sum += ap * av
            ask_vol_sum += av
    
    vwap_bid = bid_price_vol_sum / bid_vol_sum if bid_vol_sum > 0 else 0
    vwap_ask = ask_price_vol_sum / ask_vol_sum if ask_vol_sum > 0 else 0
    vwap_mid = (vwap_bid + vwap_ask) / 2 if (vwap_bid > 0 and vwap_ask > 0) else 0
    
    # --- Level-Weighted Average Price ---
    weights = [1.0 / (i + 1) for i in range(levels)]
    w_sum = sum(weights)
    weights_norm = [w / w_sum for w in weights]
    
    lwap_bid = sum(
        weights_norm[i] * row.get(f"bid_price_{i}", 0)
        for i in range(levels)
        if pd.notna(row.get(f"bid_price_{i}", 0))
    )
    lwap_ask = sum(
        weights_norm[i] * row.get(f"ask_price_{i}", 0)
        for i in range(levels)
        if pd.notna(row.get(f"ask_price_{i}", 0))
    )
    
    mid = (row["bid_price_0"] + row["ask_price_0"]) / 2
    
    return {
        "vwap_bid": vwap_bid,
        "vwap_ask": vwap_ask,
        "vwap_mid": vwap_mid,
        "lwap_bid": lwap_bid,
        "lwap_ask": lwap_ask,
        "vwap_spread": vwap_ask - vwap_bid,                 # VWAP bazlı spread
        "vwap_mid_diff": (vwap_mid - mid) / mid * 100       # VWAP vs Mid-Price farkı (%)
            if mid > 0 else 0,
    }


def compute_volume_metrics(row: pd.Series, levels: int = None) -> dict:
    """
    Hacim (Volume) metrikleri hesapla.
    
    - Toplam bid/ask hacim
    - Hacim oranları
    - En iyi seviye hacim yoğunluğu
    """
    if levels is None:
        levels = config.ORDERBOOK_DEPTH
    
    bid_volumes = [row.get(f"bid_volume_{i}", 0) for i in range(levels)]
    ask_volumes = [row.get(f"ask_volume_{i}", 0) for i in range(levels)]
    
    total_bid_vol = sum(v for v in bid_volumes if pd.notna(v))
    total_ask_vol = sum(v for v in ask_volumes if pd.notna(v))
    total_vol = total_bid_vol + total_ask_vol
    
    # Top-of-book hacim yoğunluğu (ilk seviye / toplam)
    top_bid_concentration = bid_volumes[0] / total_bid_vol if total_bid_vol > 0 else 0
    top_ask_concentration = ask_volumes[0] / total_ask_vol if total_ask_vol > 0 else 0
    
    return {
        "total_bid_volume": total_bid_vol,
        "total_ask_volume": total_ask_vol,
        "total_volume": total_vol,
        "bid_ask_volume_ratio": total_bid_vol / total_ask_vol if total_ask_vol > 0 else 1.0,
        "top_bid_concentration": top_bid_concentration,
        "top_ask_concentration": top_ask_concentration,
        "volume_log_ratio": np.log(total_bid_vol / total_ask_vol)
            if (total_bid_vol > 0 and total_ask_vol > 0) else 0,
    }


def compute_depth_metrics(row: pd.Series, levels: int = None) -> dict:
    """
    Derinlik (Depth) metrikleri hesapla.
    
    - Fiyat derinliği: En iyi fiyat ile en derin seviye arası fark
    - Kümülatif hacim eğrisi özellikleri
    """
    if levels is None:
        levels = config.ORDERBOOK_DEPTH
    
    best_bid = row["bid_price_0"]
    best_ask = row["ask_price_0"]
    
    # Bid tarafı fiyat derinliği
    deepest_bid = row.get(f"bid_price_{levels-1}", best_bid)
    if pd.isna(deepest_bid):
        deepest_bid = best_bid
    bid_depth_pct = ((best_bid - deepest_bid) / best_bid * 100) if best_bid > 0 else 0
    
    # Ask tarafı fiyat derinliği
    deepest_ask = row.get(f"ask_price_{levels-1}", best_ask)
    if pd.isna(deepest_ask):
        deepest_ask = best_ask
    ask_depth_pct = ((deepest_ask - best_ask) / best_ask * 100) if best_ask > 0 else 0
    
    # Kümülatif hacim: İlk 5 seviye vs son 15 seviye
    bid_vol_top5 = sum(row.get(f"bid_volume_{i}", 0) for i in range(min(5, levels)))
    bid_vol_bottom = sum(row.get(f"bid_volume_{i}", 0) for i in range(5, levels))
    ask_vol_top5 = sum(row.get(f"ask_volume_{i}", 0) for i in range(min(5, levels)))
    ask_vol_bottom = sum(row.get(f"ask_volume_{i}", 0) for i in range(5, levels))
    
    return {
        "bid_depth_pct": bid_depth_pct,
        "ask_depth_pct": ask_depth_pct,
        "depth_asymmetry": bid_depth_pct - ask_depth_pct,
        "bid_vol_top5_ratio": bid_vol_top5 / (bid_vol_top5 + bid_vol_bottom)
            if (bid_vol_top5 + bid_vol_bottom) > 0 else 0.5,
        "ask_vol_top5_ratio": ask_vol_top5 / (ask_vol_top5 + ask_vol_bottom)
            if (ask_vol_top5 + ask_vol_bottom) > 0 else 0.5,
    }


# ============================================================
# 2. SNAPSHOT-BAZLI TÜM ÖZELLİKLERİ HESAPLA
# ============================================================

def compute_snapshot_features(row: pd.Series) -> dict:
    """
    Tek bir snapshot için tüm özellikleri hesapla.
    
    Bu fonksiyon hem eğitim hem canlı tahmin sırasında kullanılır.
    """
    features = {}
    
    # 1. Mid-Price
    features["mid_price"] = compute_mid_price(row)
    
    # 2. Spread
    features.update(compute_spread(row))
    
    # 3. Order Book Imbalance
    features.update(compute_order_book_imbalance(row))
    
    # 4. Weighted Average Price
    features.update(compute_weighted_average_price(row))
    
    # 5. Volume Metrics
    features.update(compute_volume_metrics(row))
    
    # 6. Depth Metrics
    features.update(compute_depth_metrics(row))
    
    return features


# ============================================================
# 3. ROLLING (HAREKETLİ) İSTATİSTİKLER
# ============================================================

def add_rolling_features(df: pd.DataFrame, 
                         windows: List[int] = None) -> pd.DataFrame:
    """
    Zaman serisi özelliklerini ekle (rolling statistics).
    
    Her pencere boyutu için:
      - Mid-Price değişim oranı (return)
      - Mid-Price hareketli ortalama
      - Mid-Price standart sapma (volatilite)
      - OBI hareketli ortalama (trend)
      - Spread hareketli ortalama
      - Hacim hareketli ortalama
    """
    if windows is None:
        windows = config.ROLLING_WINDOWS
    
    df = df.copy()
    
    # Mid-price log return
    df["mid_price_return"] = np.log(df["mid_price"] / df["mid_price"].shift(1))
    
    for w in windows:
        # Mid-price rolling istatistikler
        df[f"mid_return_mean_{w}"] = df["mid_price_return"].rolling(w).mean()
        df[f"mid_return_std_{w}"] = df["mid_price_return"].rolling(w).std()
        df[f"mid_price_sma_{w}"] = df["mid_price"].rolling(w).mean()
        
        # Mid-price momentum: Mevcut fiyat vs SMA oranı
        df[f"mid_price_sma_ratio_{w}"] = (
            df["mid_price"] / df[f"mid_price_sma_{w}"]
        )
        
        # OBI trend
        df[f"obi_mean_{w}"] = df["obi"].rolling(w).mean()
        df[f"obi_std_{w}"] = df["obi"].rolling(w).std()
        
        # Spread trend
        df[f"spread_pct_mean_{w}"] = df["spread_pct"].rolling(w).mean()
        
        # Volume trend
        df[f"volume_ratio_mean_{w}"] = df["bid_ask_volume_ratio"].rolling(w).mean()
        
        # VWAP trend
        df[f"vwap_mid_diff_mean_{w}"] = df["vwap_mid_diff"].rolling(w).mean()
    
    return df


# ============================================================
# 4. HEDEF DEĞİŞKEN (TARGET VARIABLE) OLUŞTUR
# ============================================================

def create_target_variable(df: pd.DataFrame, 
                           horizon_minutes: int = None,
                           snapshot_interval_sec: float = None) -> pd.DataFrame:
    """
    İkili sınıflandırma için hedef değişken oluştur.
    
    Target:
      1 = Fiyat {horizon_minutes} dakika sonra mevcut mid-price'ın ÜSTÜNDE
      0 = Fiyat {horizon_minutes} dakika sonra mevcut mid-price'ın ALTINDA
    
    Args:
        df: Feature'ları içeren DataFrame
        horizon_minutes: Kaç dakika sonrasını tahmin ediyoruz
        snapshot_interval_sec: Snapshot'lar arası kaç saniye
    """
    if horizon_minutes is None:
        horizon_minutes = config.PREDICTION_HORIZON_MIN
    if snapshot_interval_sec is None:
        snapshot_interval_sec = config.SNAPSHOT_INTERVAL_SEC
    
    # Kaç satır ileri bakılacak
    horizon_rows = int((horizon_minutes * 60) / snapshot_interval_sec)
    
    df = df.copy()
    
    if config.TARGET_PRICE_MODE == "mid_price":
        # Mevcut mid-price'a göre: gelecek > mevcut ise 1
        future_mid = df["mid_price"].shift(-horizon_rows)
        df["target"] = (future_mid > df["mid_price"]).astype(int)
        df["future_mid_price"] = future_mid
        df["price_change_pct"] = (
            (future_mid - df["mid_price"]) / df["mid_price"] * 100
        )
    elif config.TARGET_PRICE_MODE == "custom":
        # Sabit hedef fiyata göre
        target_price = config.CUSTOM_TARGET_PRICE
        future_mid = df["mid_price"].shift(-horizon_rows)
        df["target"] = (future_mid > target_price).astype(int)
        df["future_mid_price"] = future_mid
    
    logger.info(
        f"Hedef değişken oluşturuldu: horizon={horizon_minutes}dk, "
        f"shift={horizon_rows} satır"
    )
    
    return df


# ============================================================
# 5. ANA PIPELINE: HAM VERİDEN EĞİTİM VERİSİNE
# ============================================================

def build_feature_pipeline(raw_df: pd.DataFrame,
                           horizon_minutes: int = None,
                           snapshot_interval_sec: float = None) -> pd.DataFrame:
    """
    Ham orderbook verisinden eğitime hazır veri seti oluştur.
    
    Pipeline:
      1. Ham veriden snapshot özellikleri hesapla
      2. Rolling istatistikler ekle
      3. Hedef değişken oluştur
      4. NaN satırları temizle
    
    Args:
        raw_df: data_collector'den gelen ham veri
        horizon_minutes: Tahmin ufku (dakika)
        snapshot_interval_sec: Snapshot aralığı (saniye)
        
    Returns:
        pd.DataFrame: Eğitime hazır, temizlenmiş veri seti
    """
    logger.info(f"Feature pipeline başlıyor. Gelen veri: {raw_df.shape}")
    
    # Adım 1: Snapshot bazlı özellikler
    print("  [1/4] Snapshot özellikleri hesaplanıyor...")
    feature_rows = []
    for idx, row in raw_df.iterrows():
        features = compute_snapshot_features(row)
        features["timestamp"] = idx
        feature_rows.append(features)
    
    feature_df = pd.DataFrame(feature_rows)
    feature_df = feature_df.set_index("timestamp")
    
    logger.info(f"Snapshot özellikleri: {feature_df.shape[1]} özellik")
    
    # Adım 2: Rolling istatistikler
    print("  [2/4] Rolling istatistikler ekleniyor...")
    feature_df = add_rolling_features(feature_df)
    
    # Adım 3: Hedef değişken
    print("  [3/4] Hedef değişken oluşturuluyor...")
    feature_df = create_target_variable(
        feature_df, horizon_minutes, snapshot_interval_sec
    )
    
    # Adım 4: NaN temizliği
    print("  [4/4] NaN satırları temizleniyor...")
    rows_before = len(feature_df)
    feature_df = feature_df.dropna()
    rows_after = len(feature_df)
    
    logger.info(
        f"NaN temizliği: {rows_before} -> {rows_after} satır "
        f"({rows_before - rows_after} satır silindi)"
    )
    
    # Sınıf dağılımı
    if "target" in feature_df.columns and len(feature_df) > 0:
        class_dist = feature_df["target"].value_counts()
        total = len(feature_df)
        print(f"\n  Sınıf Dağılımı:")
        print(f"    1 (Yükseliş): {class_dist.get(1, 0)} ({class_dist.get(1, 0)/total*100:.1f}%)")
        print(f"    0 (Düşüş):    {class_dist.get(0, 0)} ({class_dist.get(0, 0)/total*100:.1f}%)")
    elif len(feature_df) == 0:
        print(f"\n  ⚠ UYARI: Tüm satırlar temizlendi, veri kalmadı!")
        print(f"  Bu genellikle yetersiz veri toplandığı anlamına gelir.")
        print(f"  Tahmin ufku {config.PREDICTION_HORIZON_MIN} dk = ")
        horizon_rows = int((config.PREDICTION_HORIZON_MIN * 60) / (snapshot_interval_sec or config.SNAPSHOT_INTERVAL_SEC))
        print(f"  {horizon_rows} satır ileri bakılması gerekiyor.")
        print(f"  En az {horizon_rows + max(config.ROLLING_WINDOWS) + 10} snapshot toplamalısınız.")
    
    return feature_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Model eğitimi için kullanılacak özellik sütunlarını döndür.
    
    Hariç tutulanlar: target, future_mid_price, price_change_pct, mid_price
    (mid_price bilgi sızıntısına neden olabilir)
    """
    exclude = {
        "target", "future_mid_price", "price_change_pct", 
        "mid_price", "timestamp_ms", "last_update_id"
    }
    
    feature_cols = [
        col for col in df.columns 
        if col not in exclude and df[col].dtype in ["float64", "int64", "float32"]
    ]
    
    return sorted(feature_cols)


# ============================================================
# CANLI TAHMİN İÇİN TEK SATIRLIK FEATURE HESAPLAMA
# ============================================================

def compute_live_features(snapshot_buffer: list, 
                          rolling_windows: List[int] = None) -> Optional[pd.Series]:
    """
    Canlı tahmin sırasında, son N snapshot'dan tek satırlık
    feature vektörü oluştur.
    
    Args:
        snapshot_buffer: Son snapshot'ların listesi (dict)
        rolling_windows: Rolling pencere boyutları
        
    Returns:
        pd.Series: Tek satırlık feature vektörü
        None: Yeterli veri yoksa
    """
    if rolling_windows is None:
        rolling_windows = config.ROLLING_WINDOWS
    
    min_required = max(rolling_windows) + 5  # Güvenlik marjı
    
    if len(snapshot_buffer) < min_required:
        logger.debug(
            f"Yeterli veri yok: {len(snapshot_buffer)}/{min_required}"
        )
        return None
    
    # Buffer'ı DataFrame'e çevir
    df = pd.DataFrame(snapshot_buffer)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    
    # Snapshot özellikleri hesapla
    feature_rows = []
    for idx, row in df.iterrows():
        features = compute_snapshot_features(row)
        features["timestamp"] = idx
        feature_rows.append(features)
    
    feature_df = pd.DataFrame(feature_rows).set_index("timestamp")
    
    # Rolling istatistikler
    feature_df = add_rolling_features(feature_df, windows=rolling_windows)
    
    # Son satırı döndür (en güncel)
    last_row = feature_df.iloc[-1].dropna()
    
    return last_row


# ============================================================
# TEST / Demo
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("  FEATURE ENGINEERING - DEMO")
    print("=" * 60)
    
    # Test: Sahte bir snapshot oluştur
    import numpy as np
    
    base_price = 85000.0
    test_row = pd.Series()
    
    for i in range(20):
        test_row[f"bid_price_{i}"] = base_price - (i * 1.5)
        test_row[f"bid_volume_{i}"] = np.random.uniform(0.1, 5.0)
        test_row[f"ask_price_{i}"] = base_price + 2 + (i * 1.5)
        test_row[f"ask_volume_{i}"] = np.random.uniform(0.1, 5.0)
    
    features = compute_snapshot_features(test_row)
    
    print(f"\n  Test Mid-Price: ${features['mid_price']:,.2f}")
    print(f"  Spread: ${features['spread_abs']:.2f} ({features['spread_pct']:.4f}%)")
    print(f"  OBI (tüm seviyeler): {features['obi']:.4f}")
    print(f"  OBI (ilk 5 seviye):  {features['obi_depth_5']:.4f}")
    print(f"  VWAP Mid: ${features['vwap_mid']:,.2f}")
    print(f"  VWAP vs Mid farkı: {features['vwap_mid_diff']:.4f}%")
    print(f"  Bid/Ask Hacim Oranı: {features['bid_ask_volume_ratio']:.4f}")
    print(f"\n  Toplam {len(features)} özellik hesaplandı.")
