"""
============================================================
BTC/USDT Orderbook ML Prediction System - Eğitim Pipeline
============================================================
Tam eğitim döngüsü:
  1. Veri topla (veya mevcut CSV'den yükle)
  2. Feature engineering uygula
  3. Modeli eğit
  4. Değerlendir ve raporla
  5. Modeli kaydet

Kullanım:
  # Yeni veri toplayarak eğit:
  python train.py --collect 120

  # Mevcut CSV dosyasından eğit:
  python train.py --csv data/orderbook_raw.csv

  # Cross-validation ile eğit:
  python train.py --csv data/orderbook_raw.csv --cv
============================================================
"""

import argparse
import logging
import sys
import os
import time
from datetime import datetime

import pandas as pd
import numpy as np

import config
from data_collector import HybridOrderbookCollector
from features import build_feature_pipeline, get_feature_columns
from model import (
    OrderbookPredictor,
    split_time_series_data,
    time_series_cross_validate,
)

# Logging kurulumu
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. VERİ TOPLAMA AŞAMASI
# ============================================================

def collect_data(duration_minutes: int = 60, 
                 interval_sec: float = 1.0) -> pd.DataFrame:
    """
    Binance API'den canlı orderbook verisi topla.
    
    Args:
        duration_minutes: Toplama süresi (dakika)
        interval_sec: Snapshot aralığı (saniye)
        
    Returns:
        pd.DataFrame: Ham orderbook verisi
    """
    print(f"\n{'#'*60}")
    print(f"  ADIM 1: VERİ TOPLAMA")
    print(f"  Süre: {duration_minutes} dakika")
    print(f"  Tahmini snapshot sayısı: ~{int(duration_minutes * 60 / interval_sec)}")
    print(f"{'#'*60}")
    
    collector = HybridOrderbookCollector()
    raw_df = collector.collect_training_data(
        duration_minutes=duration_minutes,
        interval_sec=interval_sec,
    )
    
    print(f"\n  Toplanan veri: {raw_df.shape[0]} satır, {raw_df.shape[1]} sütun")
    return raw_df


def load_existing_data(csv_path: str) -> pd.DataFrame:
    """
    Mevcut CSV dosyasından veri yükle.
    
    Args:
        csv_path: CSV dosya yolu
        
    Returns:
        pd.DataFrame: Ham orderbook verisi
    """
    print(f"\n  Mevcut veri yükleniyor: {csv_path}")
    
    df = pd.read_csv(csv_path, index_col="timestamp", parse_dates=True)
    
    print(f"  Yüklenen veri: {df.shape[0]} satır, {df.shape[1]} sütun")
    print(f"  Zaman aralığı: {df.index.min()} → {df.index.max()}")
    
    return df


# ============================================================
# 2. TAM EĞİTİM PİPELİNE
# ============================================================

def full_training_pipeline(
    raw_df: pd.DataFrame,
    model_type: str = "xgboost",
    do_cross_validation: bool = False,
    snapshot_interval_sec: float = 1.0,
) -> OrderbookPredictor:
    """
    Tam eğitim pipeline: Feature engineering → Train → Evaluate → Save.
    
    Args:
        raw_df: Ham orderbook verisi
        model_type: "xgboost", "random_forest", "ensemble"
        do_cross_validation: CV uygulansın mı?
        snapshot_interval_sec: Veri toplama aralığı
        
    Returns:
        OrderbookPredictor: Eğitilmiş model
    """
    start_time = time.time()
    
    # ---- ADIM 2: Feature Engineering ----
    print(f"\n{'#'*60}")
    print(f"  ADIM 2: ÖZELLİK MÜHENDİSLİĞİ (Feature Engineering)")
    print(f"{'#'*60}")
    
    feature_df = build_feature_pipeline(
        raw_df,
        horizon_minutes=config.PREDICTION_HORIZON_MIN,
        snapshot_interval_sec=snapshot_interval_sec,
    )
    
    if feature_df.empty or len(feature_df) < 100:
        print(f"\n  {'='*55}")
        print(f"  HATA: Yeterli veri yok!")
        print(f"  {'='*55}")
        print(f"  Mevcut kullanılabilir satır: {len(feature_df)}")
        print(f"  Minimum gerekli: 100 satır")
        print(f"")
        print(f"  Tahmin ufku {config.PREDICTION_HORIZON_MIN} dakika olduğu için,")
        min_minutes = (config.PREDICTION_HORIZON_MIN * 2) + 5
        print(f"  en az {min_minutes} dakika veri toplamalısınız.")
        print(f"  Önerilen: 30-60 dakika")
        print(f"")
        print(f"  Komutu şöyle çalıştırın:")
        print(f"  python main.py collect --duration 30")
        print(f"  {'='*55}")
        sys.exit(1)
    
    # Feature sütunlarını belirle
    feature_columns = get_feature_columns(feature_df)
    print(f"\n  Toplam {len(feature_columns)} özellik kullanılacak")
    
    # Feature'ları kaydet
    feature_df.to_csv(config.FEATURES_FILE)
    print(f"  Feature'lar kaydedildi: {config.FEATURES_FILE}")
    
    # ---- ADIM 3: VERİ BÖLME ----
    print(f"\n{'#'*60}")
    print(f"  ADIM 3: VERİ BÖLME (Time Series Split)")
    print(f"{'#'*60}")
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_time_series_data(
        feature_df, feature_columns
    )
    
    # ---- ADIM 4: MODEL EĞİTİMİ ----
    print(f"\n{'#'*60}")
    print(f"  ADIM 4: MODEL EĞİTİMİ ({model_type.upper()})")
    print(f"{'#'*60}")
    
    predictor = OrderbookPredictor(model_type=model_type)
    train_results = predictor.train(X_train, y_train, X_val, y_val)
    
    print(f"\n  Eğitim Sonuçları:")
    for key, value in train_results.items():
        print(f"    {key}: {value:.4f}")
    
    # ---- ADIM 5: DEĞERLENDİRME ----
    print(f"\n{'#'*60}")
    print(f"  ADIM 5: MODEL DEĞERLENDİRME")
    print(f"{'#'*60}")
    
    metrics = predictor.print_evaluation(X_test, y_test)
    
    # Feature importance
    imp_df = predictor.get_feature_importance(top_n=15)
    if not imp_df.empty:
        print(f"\n  En Önemli 15 Özellik:")
        for _, row in imp_df.iterrows():
            bar = "█" * int(row["importance"] * 100)
            print(f"    {row['feature']:<35} {row['importance']:.4f} {bar}")
    
    # Feature importance grafiği
    predictor.plot_feature_importance(top_n=20)
    
    # ---- Cross Validation (Opsiyonel) ----
    if do_cross_validation:
        print(f"\n{'#'*60}")
        print(f"  EK: TIME SERIES CROSS-VALIDATION")
        print(f"{'#'*60}")
        
        X_all = feature_df[feature_columns]
        y_all = feature_df["target"]
        cv_results = time_series_cross_validate(X_all, y_all, model_type=model_type)
    
    # ---- ADIM 6: MODEL KAYDETME ----
    print(f"\n{'#'*60}")
    print(f"  ADIM 6: MODEL KAYDETME")
    print(f"{'#'*60}")
    
    predictor.save()
    
    elapsed = time.time() - start_time
    print(f"\n  Toplam süre: {elapsed:.1f} saniye ({elapsed/60:.1f} dakika)")
    print(f"  Model tipi: {model_type}")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test F1-Score: {metrics['f1_score']:.4f}")
    print(f"  Test ROC-AUC:  {metrics['roc_auc']:.4f}")
    
    return predictor


# ============================================================
# 3. ANA GİRİŞ NOKTASI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="BTC/USDT Orderbook ML Tahmin Sistemi - Eğitim"
    )
    
    parser.add_argument(
        "--collect", type=int, default=0,
        help="Kaç dakika veri toplanacak (0 = toplama, mevcut veriyi kullan)"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Mevcut CSV dosya yolu"
    )
    parser.add_argument(
        "--model", type=str, default="xgboost",
        choices=["xgboost", "random_forest", "ensemble"],
        help="Model tipi (varsayılan: xgboost)"
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Cross-validation uygula"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Snapshot aralığı - saniye (varsayılan: 1.0)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  BTC/USDT ORDERBOOK ML TAHMİN SİSTEMİ")
    print(f"  Eğitim Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Veri kaynağı belirleme
    if args.collect > 0:
        raw_df = collect_data(
            duration_minutes=args.collect,
            interval_sec=args.interval,
        )
    elif args.csv:
        raw_df = load_existing_data(args.csv)
    elif os.path.exists(config.RAW_DATA_FILE):
        print(f"\n  Mevcut veri dosyası bulundu: {config.RAW_DATA_FILE}")
        raw_df = load_existing_data(config.RAW_DATA_FILE)
    else:
        print("\n  HATA: Veri kaynağı belirtilmedi!")
        print("  Kullanım örnekleri:")
        print("    python train.py --collect 60      # 60 dk veri topla ve eğit")
        print("    python train.py --csv veriler.csv  # CSV'den eğit")
        sys.exit(1)
    
    # Eğitim başlat
    predictor = full_training_pipeline(
        raw_df=raw_df,
        model_type=args.model,
        do_cross_validation=args.cv,
        snapshot_interval_sec=args.interval,
    )
    
    print(f"\n{'='*60}")
    print(f"  EĞİTİM TAMAMLANDI!")
    print(f"  Model '{args.model}' kaydedildi: {config.MODEL_DIR}")
    print(f"  Canlı tahmine başlamak için: python live_predictor.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
