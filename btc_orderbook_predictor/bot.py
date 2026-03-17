#!/usr/bin/env python3
"""
============================================================
BTC/USDT ORDERBOOK ML + POLYMARKET OTOMATİK İŞLEM BOTU
============================================================
ML modeli orderbook verilerinden fiyat yönü tahmin eder,
Polymarket'teki BTC 5-dk Up/Down marketlerinde otomatik
pozisyon açar.

*** POLYMARKET PENCERE-SENKRONİZE ÇALIŞIR ***

Polymarket 5-dk marketleri:
  - Sabit pencereler: 12:00-12:05, 12:05-12:10, ... (ET)
  - Resolution kaynağı: Chainlink BTC/USD Data Stream
  - Contract: 0xc907E116054Ad103354f2D350FD2514433D57F6f

Bot Döngüsü:
  1. Sonraki 5-dk pencerenin başlamasını BEKLE
  2. Pencere açılınca Chainlink'ten başlangıç fiyatını kaydet
  3. Orderbook ML modeli ile yön tahmini yap (UP/DOWN)
  4. Güven eşiği geçilirse Polymarket'te işlem aç
  5. Pencere sonuna kadar izle, sonraki pencereye geç
  6. Tekrarla

Kullanım:
  # Simülasyon modu (varsayılan, güvenli)
  python bot.py

  # Gerçek işlem (dikkatli kullanın!)
  python bot.py --live

  # Farklı model ile
  python bot.py --model ensemble

  # Polymarket + Chainlink durumunu kontrol et
  python bot.py --status
============================================================
"""

import argparse
import logging
import sys
import os
import time
import signal
from datetime import datetime, timezone, timedelta
from collections import deque

import pandas as pd

# Proje modülleri
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_collector import OrderbookWebSocketStream, OrderbookSnapshotCollector
from features import compute_live_features
from model import OrderbookPredictor
from chainlink_feed import (
    get_chainlink_btc_price,
    get_price_comparison,
    print_price_status,
)
from polymarket_client import (
    find_next_market,
    find_market_for_window,
    place_bet,
    get_open_positions,
    get_current_5min_window,
    get_next_5min_window,
    print_status as poly_print_status,
    check_polymarket_status,
)

# Logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# ANA BOT SINIFI - PENCERE SENKRONİZE
# ============================================================

class TradingBot:
    """
    ML Tahmin + Polymarket İşlem Botu.
    
    *** PENCERE-SENKRONİZE ÇALIŞIR ***
    
    Her Polymarket 5-dk penceresi için:
      1. Pencere başlangıcını bekle
      2. Chainlink'ten açılış fiyatını kaydet  
      3. ML tahmini yap
      4. İşlem aç (güven yeterliyse)
      5. Sonraki pencereye geç
    
    Binance orderbook → Feature Engineering → ML Tahmin
                                                  ↓
    Chainlink BTC/USD (pencere açılış fiyatı) → Referans
                                                  ↓
    Polymarket CLOB API → İşlem Aç (Up/Down)
    """
    
    def __init__(self, model_type: str = "xgboost", dry_run: bool = True):
        self.model_type = model_type
        self.dry_run = dry_run
        
        # ML Model
        self.predictor = None
        
        # Veri toplama
        self.ws_stream = None
        self.rest_collector = None
        self.snapshot_buffer = deque(maxlen=5000)
        
        # İşlem geçmişi
        self.trade_history = []
        self.total_trades = 0
        self.total_dry_trades = 0
        
        # Pencere takibi
        self.current_window_start = None
        self.window_open_price = None  # Chainlink pencere açılış fiyatı
        self.windows_processed = 0
        
        # Durum
        self.is_running = False
        self.last_trade_time = None
        
        # Graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print("\n\n  Bot kapatılıyor...")
        self.stop()
    
    def initialize(self):
        """Bot bileşenlerini başlat."""
        
        print(f"\n{'═'*62}")
        print(f"  BTC ORDERBOOK ML + POLYMARKET BOTU")
        print(f"  *** PENCERE-SENKRONİZE MOD ***")
        print(f"  {'─'*54}")
        print(f"  Model:     {self.model_type}")
        print(f"  Mod:       {'🔸 DRY-RUN (Simülasyon)' if self.dry_run else '🔴 GERÇEK İŞLEM'}")
        print(f"  Bahis:     ${config.POLY_BET_AMOUNT_USDC:.2f} USDC / işlem")
        print(f"  Güven:     >{config.POLY_MIN_CONFIDENCE:.0%}")
        print(f"  Resolution: Chainlink BTC/USD (Polygon)")
        print(f"  Veri:      Binance Orderbook L2 → ML Tahmin")
        print(f"{'═'*62}")
        
        # 1. ML Modeli yükle
        print("\n  [1/4] ML modeli yükleniyor...")
        self.predictor = OrderbookPredictor(model_type=self.model_type)
        try:
            self.predictor.load()
            print(f"         ✅ Model yüklendi ({len(self.predictor.feature_names)} özellik)")
        except FileNotFoundError:
            print("         ❌ Eğitilmiş model bulunamadı!")
            print("         Önce model eğitin: python main.py train")
            sys.exit(1)
        
        # 2. Chainlink bağlantısı test et
        print("\n  [2/4] Chainlink BTC/USD kontrol ediliyor...")
        cl_data = get_chainlink_btc_price()
        if cl_data:
            print(f"         ✅ Chainlink: ${cl_data['price']:,.2f} ({cl_data['age_seconds']:.0f}s önce)")
        else:
            print("         ⚠️  Chainlink erişilemedi - Binance fiyatı yedek olarak kullanılacak")
        
        # Spread kontrolü
        comp = get_price_comparison()
        if comp["spread_pct"] is not None:
            icon = "✅" if comp["aligned"] else "⚠️"
            print(f"         {icon} Chainlink-Binance Spread: {comp['spread_pct']:.4f}%")
        
        # 3. Polymarket durumunu kontrol et
        print("\n  [3/4] Polymarket kontrol ediliyor...")
        poly_status = check_polymarket_status()
        
        if poly_status["api_ok"]:
            print(f"         ✅ Gamma API erişimi OK")
        else:
            print(f"         ❌ Gamma API erişilemedi")
        
        print(f"         📊 {poly_status['active_markets']} aktif BTC 5-dk market")
        
        if not self.dry_run:
            if poly_status["auth_ok"]:
                print(f"         ✅ Kimlik doğrulama OK")
                if poly_status["balance"] is not None:
                    print(f"         💰 Bakiye: ${poly_status['balance']:.2f} USDC")
            else:
                print(f"         ❌ Kimlik doğrulama başarısız!")
                print(f"         .env dosyasında POLY_PRIVATE_KEY ayarlayın.")
                print(f"         DRY-RUN moduna geçiliyor...")
                self.dry_run = True
        
        # 4. Veri akışını başlat
        print("\n  [4/4] Orderbook stream başlatılıyor...")
        self._start_data_stream()
        print(f"         ✅ Stream aktif")
        
        # Mevcut pencere bilgisi
        window = get_current_5min_window()
        print(f"\n  ⏱  Mevcut pencere: {window['window_label']}")
        print(f"     Kalan: {window['seconds_remaining']:.0f} saniye")
    
    def _start_data_stream(self):
        """WebSocket veya REST ile veri akışı başlat."""
        
        def on_snapshot(snapshot):
            self.snapshot_buffer.append(snapshot)
        
        try:
            self.ws_stream = OrderbookWebSocketStream()
            self.ws_stream.start(on_snapshot=on_snapshot)
            
            # Bağlantı bekle
            for _ in range(15):
                if self.ws_stream.buffer_size > 0:
                    return
                time.sleep(1)
            
            # WebSocket başarısızsa REST'e geç
            logger.warning("WebSocket bağlanamadı, REST moduna geçiliyor...")
            self.ws_stream.stop()
            self.ws_stream = None
        except Exception:
            pass
        
        self.rest_collector = OrderbookSnapshotCollector()
        logger.info("REST API polling modu aktif.")
    
    def _collect_rest_snapshot(self):
        """REST ile tek snapshot al."""
        if self.rest_collector:
            snapshot = self.rest_collector.fetch_single_snapshot()
            if snapshot:
                self.snapshot_buffer.append(snapshot)
    
    def _make_prediction(self) -> dict:
        """
        ML modeli ile tahmin yap.
        
        Returns:
            dict: {prediction, direction, confidence, prob_up, prob_down, 
                   mid_price, chainlink_price}
            None: Yeterli veri yoksa
        """
        if len(self.snapshot_buffer) < config.MIN_SAMPLES_FOR_FEATURES:
            return None
        
        buffer_list = list(self.snapshot_buffer)
        features = compute_live_features(buffer_list)
        
        if features is None:
            return None
        
        # Feature vektörünü hazırla
        expected = self.predictor.feature_names
        feature_vector = pd.DataFrame([features])
        
        for col in expected:
            if col not in feature_vector.columns:
                feature_vector[col] = 0.0
        
        feature_vector = feature_vector[expected]
        
        # Tahmin
        prediction = self.predictor.predict(feature_vector)[0]
        probability = self.predictor.predict_proba(feature_vector)[0]
        
        latest = buffer_list[-1]
        mid_price = (latest["bid_price_0"] + latest["ask_price_0"]) / 2
        
        # Chainlink fiyatını da al (referans)
        cl_price = None
        cl_data = get_chainlink_btc_price()
        if cl_data:
            cl_price = cl_data["price"]
        
        return {
            "prediction": int(prediction),
            "direction": "UP" if prediction == 1 else "DOWN",
            "confidence": float(max(probability)),
            "prob_up": float(probability[1]),
            "prob_down": float(probability[0]),
            "mid_price": mid_price,
            "chainlink_price": cl_price,
            "window_open_price": self.window_open_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def _should_trade(self, prediction: dict, window: dict) -> bool:
        """
        İşlem yapılmalı mı karar ver.
        
        Pencere-senkronize kontroller:
          1. Güven eşiği geçildi mi?
          2. Pencerede yeterli süre var mı?
          3. Pencere çok geç mi? (son 60 saniye riskli)
          4. Bu pencerede zaten işlem açıldı mı?
          5. Max açık pozisyon aşıldı mı?
        """
        # 1. Güven kontrolü
        if prediction["confidence"] < config.POLY_MIN_CONFIDENCE:
            return False
        
        # 2. Pencerede yeterli süre kontrolü
        remaining = window["seconds_remaining"]
        if remaining < config.WINDOW_EXIT_BUFFER_SEC:
            logger.info(f"Pencere sonuna çok yakın ({remaining:.0f}s), işlem atlanıyor.")
            return False
        
        # 3. Bu pencerede zaten işlem açıldı mı?
        if self.last_trade_time and self.current_window_start:
            if self.last_trade_time >= self.current_window_start:
                logger.info("Bu pencerede zaten işlem açıldı.")
                return False
        
        # 4. Pozisyon limiti (gerçek modda)
        if not self.dry_run:
            try:
                positions = get_open_positions()
                if len(positions) >= config.POLY_MAX_OPEN_POSITIONS:
                    logger.info("Max pozisyon limitine ulaşıldı.")
                    return False
            except Exception:
                pass
        
        return True
    
    def _execute_trade(self, prediction: dict, window: dict):
        """
        Polymarket'te işlem aç.
        
        Pencere-senkronize: Mevcut pencereye ait marketi bulur.
        """
        # Pencereye ait marketi bul
        market = find_market_for_window(window["window_start"])
        
        if not market:
            # Fallback: sıradaki en yakın marketi al
            market = find_next_market()
        
        if not market:
            print(f"  ⚠ Aktif BTC 5-dk market bulunamadı, işlem atlanıyor.")
            return
        
        # Markette yeterli süre var mı?
        minutes_left = market.get("_minutes_left", 0)
        if minutes_left < 1:
            print(f"  ⚠ Market kapanmak üzere ({minutes_left:.1f} dk), atlanıyor.")
            return
        
        # İşlem yap
        direction = "up" if prediction["direction"] == "UP" else "down"
        
        result = place_bet(
            market=market,
            direction=direction,
            amount_usdc=config.POLY_BET_AMOUNT_USDC,
            dry_run=self.dry_run,
        )
        
        # ML + Chainlink bilgilerini ekle
        result["ml_prediction"] = prediction
        result["window_label"] = window["window_label"]
        result["chainlink_open_price"] = self.window_open_price
        result["chainlink_current_price"] = prediction.get("chainlink_price")
        self.trade_history.append(result)
        
        if result["success"]:
            if self.dry_run:
                self.total_dry_trades += 1
            else:
                self.total_trades += 1
            self.last_trade_time = datetime.now(timezone.utc)
        
        # Log
        self._save_trade_log()
    
    def _print_prediction(self, prediction: dict, window: dict, will_trade: bool):
        """Tahmin sonucunu pencere bilgisiyle birlikte göster."""
        conf = prediction["confidence"]
        direction_icon = "📈" if prediction["direction"] == "UP" else "📉"
        
        print(f"\n  {'─'*58}")
        print(f"  🪟 Pencere: {window['window_label']}  ({window['seconds_remaining']:.0f}s kaldı)")
        print(f"  ⏱  {prediction['timestamp'][:19]} UTC")
        
        # Fiyat kaynakları
        print(f"  💰 Binance Mid:  ${prediction['mid_price']:,.2f}")
        if prediction.get("chainlink_price"):
            print(f"  🔗 Chainlink:    ${prediction['chainlink_price']:,.2f}  ← Polymarket resolution")
        if self.window_open_price:
            current = prediction.get("chainlink_price") or prediction["mid_price"]
            diff = current - self.window_open_price
            diff_icon = "📈" if diff >= 0 else "📉"
            print(f"  📌 Pencere Açılış: ${self.window_open_price:,.2f}  (Δ {diff_icon} ${abs(diff):,.2f})")
        
        # ML Tahmin
        print(f"  {direction_icon} Tahmin: {prediction['direction']} (Güven: {conf:.1%})")
        print(f"     Up: {prediction['prob_up']:.1%} | Down: {prediction['prob_down']:.1%}")
        
        if will_trade:
            mode = "DRY-RUN" if self.dry_run else "GERÇEK"
            print(f"  🎯 İŞLEM AÇILIYOR ({mode})...")
        else:
            if conf < config.POLY_MIN_CONFIDENCE:
                print(f"  ⏸  Güven düşük ({conf:.1%} < {config.POLY_MIN_CONFIDENCE:.0%}), bekliyor...")
            elif window["seconds_remaining"] < config.WINDOW_EXIT_BUFFER_SEC:
                print(f"  ⏸  Pencere sonu yakın, atlanıyor...")
            else:
                print(f"  ⏸  Bu pencerede zaten işlem açıldı.")
        
        print(f"  Buffer: {len(self.snapshot_buffer)} | "
              f"Pencere: #{self.windows_processed} | "
              f"İşlemler: {self.total_trades + self.total_dry_trades}")
        print(f"  {'─'*58}")
    
    def _wait_for_next_window(self) -> dict:
        """
        Sonraki 5-dk pencereyi bekle ve açılış fiyatını kaydet.
        
        Returns:
            dict: Yeni pencere bilgisi
        """
        next_win = get_next_5min_window()
        wait_time = next_win["seconds_until_start"] + config.WINDOW_ENTRY_OFFSET_SEC
        
        if wait_time > 0:
            print(f"\n  ⏳ Sonraki pencere: {next_win['window_label']}")
            print(f"     Bekleniyor: {wait_time:.0f} saniye ", end="", flush=True)
            
            # REST modunda beklerken veri toplamaya devam et
            while wait_time > 0 and self.is_running:
                if self.rest_collector:
                    self._collect_rest_snapshot()
                
                sleep_chunk = min(wait_time, 2.0)
                time.sleep(sleep_chunk)
                wait_time -= sleep_chunk
                
                # Her 10 saniyede geri sayım göster
                mins = int(wait_time // 60)
                secs = int(wait_time % 60)
                print(f"\r     ⏱  {mins:02d}:{secs:02d} kaldı  "
                      f"(buffer: {len(self.snapshot_buffer)})   ", end="", flush=True)
            
            print(f"\r     ✅ Pencere açıldı!                              ")
        
        # Yeni pencere bilgisini al
        window = get_current_5min_window()
        self.current_window_start = window["window_start"]
        self.windows_processed += 1
        
        # *** KRİTİK: Chainlink'ten pencere açılış fiyatını kaydet ***
        cl_data = get_chainlink_btc_price()
        if cl_data:
            self.window_open_price = cl_data["price"]
            logger.info(
                f"Pencere açılış fiyatı (Chainlink): ${cl_data['price']:,.2f} "
                f"(age: {cl_data['age_seconds']:.0f}s)"
            )
        else:
            # Fallback: Binance mid-price
            if self.snapshot_buffer:
                latest = self.snapshot_buffer[-1]
                self.window_open_price = (latest["bid_price_0"] + latest["ask_price_0"]) / 2
                logger.warning(
                    f"Chainlink erişilemedi, Binance mid-price kullanılıyor: "
                    f"${self.window_open_price:,.2f}"
                )
        
        return window
    
    def run(self, duration_minutes: int = 0):
        """
        Ana bot döngüsü - PENCERE SENKRONİZE.
        
        Her 5-dk pencerede:
          1. Pencere başlangıcını bekle
          2. Chainlink açılış fiyatını kaydet
          3. ML tahmini yap
          4. İşlem kararı ver ve uygula
          5. Sonraki pencereye geç
        
        Args:
            duration_minutes: Çalışma süresi (0 = sonsuz)
        """
        self.initialize()
        
        # Veri birikimi bekle
        min_req = config.MIN_SAMPLES_FOR_FEATURES
        print(f"\n  Veri birikiyor... (minimum {min_req} snapshot)")
        
        self.is_running = True
        start_time = time.time()
        
        # İlk birikim
        while self.is_running and len(self.snapshot_buffer) < min_req:
            if self.rest_collector:
                self._collect_rest_snapshot()
            
            remaining = min_req - len(self.snapshot_buffer)
            print(f"\r  Birikim: {len(self.snapshot_buffer)}/{min_req} "
                  f"(~{remaining} kaldı)   ", end="", flush=True)
            time.sleep(1)
        
        if not self.is_running:
            return
        
        print(f"\r  ✅ Veri hazır! ({len(self.snapshot_buffer)} snapshot)           ")
        
        # ============================================================
        # ANA PENCERE DÖNGÜSÜ
        # ============================================================
        try:
            while self.is_running:
                # --- 1. Sonraki pencereyi bekle ---
                window = self._wait_for_next_window()
                
                if not self.is_running:
                    break
                
                # --- 2. REST modunda ek veri al ---
                if self.rest_collector:
                    self._collect_rest_snapshot()
                
                # --- 3. ML Tahmini ---
                prediction = self._make_prediction()
                
                if prediction:
                    should_trade = self._should_trade(prediction, window)
                    self._print_prediction(prediction, window, should_trade)
                    
                    if should_trade:
                        self._execute_trade(prediction, window)
                else:
                    print(f"\n  ⚠ Tahmin yapılamadı (buffer: {len(self.snapshot_buffer)})")
                
                # --- 4. Süre kontrolü ---
                if duration_minutes > 0:
                    elapsed = (time.time() - start_time) / 60
                    if elapsed >= duration_minutes:
                        print(f"\n  Süre doldu ({duration_minutes} dk).")
                        break
                
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Botu düzgün kapat."""
        self.is_running = False
        
        if self.ws_stream:
            self.ws_stream.stop()
        
        self._save_trade_log()
        self._print_summary()
    
    def _save_trade_log(self):
        """İşlem geçmişini CSV'ye kaydet."""
        if not self.trade_history:
            return
        
        log_path = os.path.join(config.LOG_DIR, "bot_trades.csv")
        
        try:
            rows = []
            for t in self.trade_history:
                row = {
                    "timestamp": t.get("timestamp", ""),
                    "direction": t.get("direction", ""),
                    "amount_usdc": t.get("amount_usdc", 0),
                    "price": t.get("price", 0),
                    "success": t.get("success", False),
                    "order_id": t.get("order_id", ""),
                    "dry_run": t.get("dry_run", True),
                    "mode": t.get("mode", ""),
                    "market": t.get("market_question", "")[:80],
                    "window": t.get("window_label", ""),
                    "ml_confidence": t.get("ml_prediction", {}).get("confidence", 0),
                    "ml_direction": t.get("ml_prediction", {}).get("direction", ""),
                    "btc_mid_price": t.get("ml_prediction", {}).get("mid_price", 0),
                    "chainlink_open": t.get("chainlink_open_price", ""),
                    "chainlink_current": t.get("chainlink_current_price", ""),
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(log_path, index=False)
            logger.info(f"İşlem geçmişi kaydedildi: {log_path}")
        except Exception as e:
            logger.warning(f"Log kaydetme hatası: {e}")
    
    def _print_summary(self):
        """Çalışma özeti."""
        print(f"\n{'═'*62}")
        print(f"  BOT ÇALIŞMA ÖZETİ")
        print(f"{'═'*62}")
        print(f"  Mod:              {'DRY-RUN' if self.dry_run else 'GERÇEK'}")
        print(f"  Pencere Sayısı:   {self.windows_processed}")
        print(f"  Toplam DRY İşlem: {self.total_dry_trades}")
        print(f"  Toplam GERÇEK:    {self.total_trades}")
        print(f"  Buffer Boyutu:    {len(self.snapshot_buffer)}")
        
        if self.trade_history:
            up_count = sum(1 for t in self.trade_history if t.get("direction") == "UP")
            down_count = sum(1 for t in self.trade_history if t.get("direction") == "DOWN")
            success = sum(1 for t in self.trade_history if t.get("success"))
            print(f"  UP İşlemler:      {up_count}")
            print(f"  DOWN İşlemler:    {down_count}")
            print(f"  Başarılı:         {success}/{len(self.trade_history)}")
        
        # Son Chainlink fiyatı
        cl = get_chainlink_btc_price()
        if cl:
            print(f"  Son Chainlink:    ${cl['price']:,.2f}")
        
        print(f"{'═'*62}\n")


# ============================================================
# CLI GİRİŞ NOKTASI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="BTC Orderbook ML + Polymarket Otomatik İşlem Botu (Pencere-Senkronize)"
    )
    
    parser.add_argument(
        "--model", type=str, default="xgboost",
        choices=["xgboost", "random_forest", "ensemble"],
        help="ML model tipi (varsayılan: xgboost)"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="GERÇEK İŞLEM modu (dikkatli kullanın!)"
    )
    parser.add_argument(
        "--duration", type=int, default=0,
        help="Çalışma süresi - dakika (0 = sonsuz)"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Polymarket + Chainlink durum kontrolü"
    )
    parser.add_argument(
        "--bet", type=float, default=None,
        help=f"İşlem tutarı USDC (varsayılan: {config.POLY_BET_AMOUNT_USDC})"
    )
    
    args = parser.parse_args()
    
    # Sadece durum kontrolü
    if args.status:
        poly_print_status()
        print_price_status()
        return
    
    # Bahis tutarı override
    if args.bet:
        config.POLY_BET_AMOUNT_USDC = args.bet
    
    # Dry-run / Live seçimi
    dry_run = not args.live
    
    if not dry_run:
        print(f"\n  ⚠️  UYARI: GERÇEK İŞLEM MODU!")
        print(f"  Her işlemde ${config.POLY_BET_AMOUNT_USDC:.2f} USDC harcanacak.")
        print(f"  Resolution: Chainlink BTC/USD (Polymarket)")
        print(f"  Devam etmek için 'EVET' yazın: ", end="")
        confirm = input().strip()
        if confirm != "EVET":
            print("  İptal edildi. DRY-RUN modunda başlatılıyor...\n")
            dry_run = True
    
    # Botu başlat
    bot = TradingBot(model_type=args.model, dry_run=dry_run)
    bot.run(duration_minutes=args.duration)


if __name__ == "__main__":
    main()
