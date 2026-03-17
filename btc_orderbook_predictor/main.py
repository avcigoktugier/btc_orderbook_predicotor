#!/usr/bin/env python3
"""
============================================================
BTC/USDT ORDERBOOK ML TAHMİN SİSTEMİ - Ana Giriş Noktası
============================================================
Kripto para (BTC/USDT) fiyat yönü tahmin sistemi.
Level 2 orderbook verileri + XGBoost/Random Forest.

Modlar:
  1. collect  - Binance'den orderbook verisi topla
  2. train    - Toplanan veriyle model eğit
  3. predict  - Canlı tahmin motorunu çalıştır
  4. demo     - Hızlı demo (topla → eğit → tahmin)
  5. test     - API bağlantı testi

Kullanım:
  python main.py collect --duration 120
  python main.py train --model xgboost --cv
  python main.py predict --model xgboost
  python main.py demo --duration 10
  python main.py test
============================================================

Yazar: Quant Trading System
Tarih: 2026
Lisans: MIT
"""

import argparse
import logging
import sys
import os
import time
from datetime import datetime

# Proje dizinini Python path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def setup_logging():
    """Merkezi loglama yapılandırması."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_FILE, mode="a"),
        ],
    )


def print_banner():
    """Başlık banner'ı."""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║    BTC/USDT ORDERBOOK ML TAHMİN SİSTEMİ                 ║
║    ──────────────────────────────────────                 ║
║    Level 2 Orderbook + XGBoost/Random Forest             ║
║    5 Dakika Fiyat Yönü Tahmini                           ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)


# ============================================================
# KOMUT: TEST
# ============================================================

def cmd_test(args):
    """API bağlantı testi."""
    print("\n  🔌 API Bağlantı Testi...")
    print(f"  Borsa: {config.EXCHANGE_ID}")
    print(f"  Sembol: {config.SYMBOL}")
    print(f"  Orderbook Derinliği: {config.ORDERBOOK_DEPTH} seviye\n")
    
    from data_collector import OrderbookSnapshotCollector
    
    collector = OrderbookSnapshotCollector()
    snapshot = collector.fetch_single_snapshot()
    
    if snapshot:
        mid = (snapshot["bid_price_0"] + snapshot["ask_price_0"]) / 2
        spread = snapshot["ask_price_0"] - snapshot["bid_price_0"]
        
        print(f"  ✅ Bağlantı BAŞARILI!")
        print(f"  ──────────────────────")
        print(f"  Best Bid:    ${snapshot['bid_price_0']:,.2f}")
        print(f"  Best Ask:    ${snapshot['ask_price_0']:,.2f}")
        print(f"  Mid-Price:   ${mid:,.2f}")
        print(f"  Spread:      ${spread:,.2f} ({spread/mid*100:.4f}%)")
        print(f"  Zaman (UTC): {snapshot['timestamp'][:19]}")
        print()
        
        # Feature engineering testi
        from features import compute_snapshot_features
        import pandas as pd
        
        row = pd.Series(snapshot)
        features = compute_snapshot_features(row)
        
        print(f"  📊 Feature Engineering Testi:")
        print(f"  OBI (tüm):    {features['obi']:.4f}")
        print(f"  OBI (top 5):  {features['obi_depth_5']:.4f}")
        print(f"  VWAP Mid:     ${features['vwap_mid']:,.2f}")
        print(f"  Bid/Ask Hacim Oranı: {features['bid_ask_volume_ratio']:.4f}")
        print(f"  Toplam {len(features)} özellik hesaplandı.")
        print(f"\n  Tüm sistemler çalışıyor! ✅")
    else:
        print("  ❌ Bağlantı BAŞARISIZ!")
        print("  İnternet bağlantınızı kontrol edin.")
        print("  Binance erişim engeli olabilir (VPN gerekebilir).")
        sys.exit(1)


# ============================================================
# KOMUT: COLLECT
# ============================================================

def cmd_collect(args):
    """Orderbook verisi topla."""
    from data_collector import HybridOrderbookCollector
    
    duration = args.duration
    interval = args.interval
    
    print(f"\n  📡 Veri Toplama Modu")
    print(f"  Süre: {duration} dakika")
    print(f"  Aralık: {interval} saniye")
    print(f"  Tahmini snapshot: ~{int(duration * 60 / interval)}")
    print(f"  Kayıt yeri: {config.RAW_DATA_FILE}")
    print()
    
    collector = HybridOrderbookCollector()
    df = collector.collect_training_data(
        duration_minutes=duration,
        interval_sec=interval,
    )
    
    print(f"\n  ✅ Veri toplama tamamlandı!")
    print(f"  Toplam: {len(df)} snapshot")
    print(f"  Dosya: {config.RAW_DATA_FILE}")


# ============================================================
# KOMUT: TRAIN
# ============================================================

def cmd_train(args):
    """Model eğit."""
    from train import full_training_pipeline, load_existing_data, collect_data
    
    # Veri kaynağı
    if args.collect > 0:
        raw_df = collect_data(args.collect, args.interval)
    elif args.csv:
        raw_df = load_existing_data(args.csv)
    elif os.path.exists(config.RAW_DATA_FILE):
        raw_df = load_existing_data(config.RAW_DATA_FILE)
    else:
        print("\n  ❌ Eğitim verisi bulunamadı!")
        print("  Önce veri toplayın: python main.py collect --duration 60")
        sys.exit(1)
    
    predictor = full_training_pipeline(
        raw_df=raw_df,
        model_type=args.model,
        do_cross_validation=args.cv,
        snapshot_interval_sec=args.interval,
    )
    
    print(f"\n  ✅ Eğitim tamamlandı! Model: {config.MODEL_DIR}")


# ============================================================
# KOMUT: PREDICT
# ============================================================

def cmd_predict(args):
    """Canlı tahmin motorunu çalıştır."""
    from live_predictor import LivePredictor
    
    # Model dosyası var mı kontrol et
    model_path = config.XGBOOST_MODEL_PATH
    if args.model == "random_forest":
        model_path = config.RF_MODEL_PATH
    
    if not os.path.exists(model_path) and args.model != "ensemble":
        print(f"\n  ❌ Eğitilmiş model bulunamadı: {model_path}")
        print("  Önce model eğitin: python main.py train")
        sys.exit(1)
    
    engine = LivePredictor(model_type=args.model)
    engine.run(
        prediction_interval=args.interval,
        use_websocket=not args.rest,
        duration_minutes=args.duration,
    )


# ============================================================
# KOMUT: DEMO
# ============================================================

def cmd_demo(args):
    """
    Hızlı demo: Kısa süre veri topla → Eğit → Tahmin yap.
    Sistemin çalıştığını doğrulamak için.
    """
    duration = args.duration  # Varsayılan: 10 dakika
    
    print(f"\n  🚀 HIZLI DEMO MODU")
    print(f"  ────────────────────")
    print(f"  1. {duration} dakika veri toplanacak")
    print(f"  2. Model eğitilecek")
    print(f"  3. 5 dakika canlı tahmin yapılacak")
    print(f"  ────────────────────\n")
    
    # 1. Veri topla
    from data_collector import HybridOrderbookCollector
    collector = HybridOrderbookCollector()
    raw_df = collector.collect_training_data(
        duration_minutes=duration,
        interval_sec=1.0,
    )
    
    if len(raw_df) < 100:
        print("\n  ❌ Yeterli veri toplanamadı. Daha uzun süre deneyin.")
        sys.exit(1)
    
    # 2. Eğit
    from train import full_training_pipeline
    predictor = full_training_pipeline(
        raw_df=raw_df,
        model_type="xgboost",
        snapshot_interval_sec=1.0,
    )
    
    # 3. Canlı tahmin (5 dakika)
    from live_predictor import LivePredictor
    engine = LivePredictor(model_type="xgboost")
    engine.run(
        prediction_interval=10,
        use_websocket=True,
        duration_minutes=5,
    )


# ============================================================
# ANA GİRİŞ
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="BTC/USDT Orderbook ML Tahmin Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py test                          # API bağlantı testi
  python main.py collect --duration 120        # 2 saat veri topla
  python main.py train --model xgboost --cv    # XGBoost + CV ile eğit
  python main.py train --model ensemble        # Ensemble model eğit
  python main.py predict                       # Canlı tahmin başlat
  python main.py predict --model ensemble      # Ensemble ile tahmin
  python main.py demo --duration 10            # 10 dk hızlı demo
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Komut")
    
    # --- test ---
    test_parser = subparsers.add_parser("test", help="API bağlantı testi")
    
    # --- collect ---
    collect_parser = subparsers.add_parser("collect", help="Orderbook verisi topla")
    collect_parser.add_argument("--duration", type=int, default=60,
                                help="Toplama süresi - dakika (varsayılan: 60)")
    collect_parser.add_argument("--interval", type=float, default=1.0,
                                help="Snapshot aralığı - saniye (varsayılan: 1.0)")
    
    # --- train ---
    train_parser = subparsers.add_parser("train", help="Model eğit")
    train_parser.add_argument("--model", type=str, default="xgboost",
                               choices=["xgboost", "random_forest", "ensemble"])
    train_parser.add_argument("--csv", type=str, default=None,
                               help="Mevcut CSV dosya yolu")
    train_parser.add_argument("--collect", type=int, default=0,
                               help="Eğitim öncesi veri toplama süresi (dakika)")
    train_parser.add_argument("--cv", action="store_true",
                               help="Cross-validation uygula")
    train_parser.add_argument("--interval", type=float, default=1.0,
                               help="Snapshot aralığı - saniye")
    
    # --- predict ---
    predict_parser = subparsers.add_parser("predict", help="Canlı tahmin")
    predict_parser.add_argument("--model", type=str, default="xgboost",
                                 choices=["xgboost", "random_forest", "ensemble"])
    predict_parser.add_argument("--interval", type=int, default=10,
                                 help="Tahmin aralığı - saniye (varsayılan: 10)")
    predict_parser.add_argument("--rest", action="store_true",
                                 help="WebSocket yerine REST kullan")
    predict_parser.add_argument("--duration", type=int, default=0,
                                 help="Çalışma süresi - dakika (0 = sonsuz)")
    
    # --- demo ---
    demo_parser = subparsers.add_parser("demo", help="Hızlı demo")
    demo_parser.add_argument("--duration", type=int, default=10,
                              help="Demo veri toplama süresi - dakika (varsayılan: 10)")
    
    args = parser.parse_args()
    
    # Logging başlat
    setup_logging()
    
    # Banner
    print_banner()
    
    # Komut yönlendirme
    if args.command == "test":
        cmd_test(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()
        print("\n  Başlamak için: python main.py test")


if __name__ == "__main__":
    main()
