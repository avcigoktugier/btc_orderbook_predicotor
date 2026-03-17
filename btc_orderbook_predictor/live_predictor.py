"""
============================================================
BTC/USDT Orderbook ML Prediction System - Canlı Tahmin Motoru
============================================================
WebSocket üzerinden gerçek zamanlı orderbook verisi alır,
eğitilmiş modeli kullanarak 5 dakika sonrası için
fiyat yönü tahmini yapar.

Kullanım:
  python live_predictor.py
  python live_predictor.py --model ensemble --interval 15
============================================================
"""

import argparse
import logging
import sys
import time
import signal
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import Optional

import pandas as pd
import numpy as np

import config
from data_collector import OrderbookWebSocketStream, OrderbookSnapshotCollector
from features import compute_live_features, get_feature_columns
from model import OrderbookPredictor

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
# CANLI TAHMİN MOTORU
# ============================================================

class LivePredictor:
    """
    Gerçek zamanlı fiyat yönü tahmin motoru.
    
    İş akışı:
      1. Eğitilmiş modeli yükle
      2. WebSocket ile orderbook stream başlat
      3. Her N saniyede bir:
         a. Son snapshot'lardan feature vektörü hesapla
         b. Model ile tahmin yap
         c. Sonucu ekrana yazdır ve logla
    """
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Args:
            model_type: Kullanılacak model tipi
        """
        self.model_type = model_type
        self.predictor: Optional[OrderbookPredictor] = None
        self.ws_stream: Optional[OrderbookWebSocketStream] = None
        self.rest_collector: Optional[OrderbookSnapshotCollector] = None
        
        # Tahmin geçmişi
        self.prediction_history: deque = deque(maxlen=1000)
        self.snapshot_buffer: deque = deque(maxlen=5000)
        
        # Durum
        self.is_running = False
        self.total_predictions = 0
        self.correct_predictions = 0
        
        # Graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Ctrl+C ile düzgün kapanma."""
        print("\n\n  Kapatılıyor...")
        self.stop()
    
    def load_model(self):
        """Eğitilmiş modeli diskten yükle."""
        print(f"\n  Model yükleniyor: {self.model_type}")
        
        self.predictor = OrderbookPredictor(model_type=self.model_type)
        self.predictor.load()
        
        print(f"  Model yüklendi. Özellik sayısı: {len(self.predictor.feature_names)}")
    
    def _on_ws_snapshot(self, snapshot: dict):
        """WebSocket'den gelen her snapshot'ı buffer'a ekle."""
        self.snapshot_buffer.append(snapshot)
    
    def start_stream(self, use_websocket: bool = True):
        """
        Veri akışını başlat.
        
        Args:
            use_websocket: True = WebSocket, False = REST polling
        """
        if use_websocket:
            print("  WebSocket stream başlatılıyor...")
            self.ws_stream = OrderbookWebSocketStream()
            self.ws_stream.start(on_snapshot=self._on_ws_snapshot)
            
            # Bağlantının kurulmasını bekle
            print("  WebSocket bağlantısı bekleniyor", end="")
            for _ in range(30):
                if self.ws_stream.buffer_size > 0:
                    print(" OK")
                    break
                print(".", end="", flush=True)
                time.sleep(1)
            else:
                print("\n  UYARI: WebSocket bağlantısı kurulamadı, REST'e geçiliyor...")
                self.ws_stream.stop()
                self.ws_stream = None
                use_websocket = False
        
        if not use_websocket:
            print("  REST API polling modu aktif.")
            self.rest_collector = OrderbookSnapshotCollector()
    
    def _collect_rest_snapshot(self):
        """REST API ile tek snapshot al ve buffer'a ekle."""
        snapshot = self.rest_collector.fetch_single_snapshot()
        if snapshot:
            self.snapshot_buffer.append(snapshot)
    
    def make_prediction(self) -> Optional[dict]:
        """
        Mevcut verilerle tek bir tahmin yap.
        
        Returns:
            dict: Tahmin sonucu veya None
        """
        if len(self.snapshot_buffer) < config.MIN_SAMPLES_FOR_FEATURES:
            return None
        
        # Son snapshot'lardan feature vektörü oluştur
        buffer_list = list(self.snapshot_buffer)
        features = compute_live_features(buffer_list)
        
        if features is None:
            return None
        
        # Model'in beklediği feature'ları filtrele
        expected_features = self.predictor.feature_names
        available_features = [f for f in expected_features if f in features.index]
        
        if len(available_features) < len(expected_features) * 0.8:
            logger.warning(
                f"Eksik özellikler: {len(available_features)}/{len(expected_features)}"
            )
            return None
        
        # Eksik feature'ları 0 ile doldur
        feature_vector = pd.DataFrame([features])
        for col in expected_features:
            if col not in feature_vector.columns:
                feature_vector[col] = 0.0
        
        feature_vector = feature_vector[expected_features]
        
        # Tahmin
        prediction = self.predictor.predict(feature_vector)[0]
        probability = self.predictor.predict_proba(feature_vector)[0]
        
        # Mevcut fiyat bilgisi
        latest = buffer_list[-1]
        current_mid = (latest["bid_price_0"] + latest["ask_price_0"]) / 2
        
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_mid_price": current_mid,
            "prediction": int(prediction),
            "direction": "YUKARI ▲" if prediction == 1 else "AŞAĞI ▼",
            "confidence": float(max(probability)),
            "prob_up": float(probability[1]),
            "prob_down": float(probability[0]),
            "horizon_minutes": config.PREDICTION_HORIZON_MIN,
            "buffer_size": len(self.snapshot_buffer),
        }
        
        # Geçmişe ekle (doğruluk kontrolü için)
        result["verify_time"] = (
            datetime.now(timezone.utc) + 
            timedelta(minutes=config.PREDICTION_HORIZON_MIN)
        ).isoformat()
        result["verify_target_price"] = current_mid
        
        self.prediction_history.append(result)
        self.total_predictions += 1
        
        return result
    
    def verify_past_predictions(self):
        """
        Geçmiş tahminlerin doğruluğunu kontrol et.
        (Gerçek fiyat, tahmin edilen yönle eşleşiyor mu?)
        """
        now = datetime.now(timezone.utc)
        
        if not self.snapshot_buffer:
            return
        
        latest = list(self.snapshot_buffer)[-1]
        current_mid = (latest["bid_price_0"] + latest["ask_price_0"]) / 2
        
        verified_count = 0
        for pred in self.prediction_history:
            if pred.get("verified"):
                continue
            
            verify_time = datetime.fromisoformat(pred["verify_time"])
            if verify_time.tzinfo is None:
                verify_time = verify_time.replace(tzinfo=timezone.utc)
            
            if now >= verify_time:
                target = pred["verify_target_price"]
                actual_direction = 1 if current_mid > target else 0
                pred["actual_direction"] = actual_direction
                pred["actual_price"] = current_mid
                pred["correct"] = (pred["prediction"] == actual_direction)
                pred["verified"] = True
                
                if pred["correct"]:
                    self.correct_predictions += 1
                
                verified_count += 1
        
        if verified_count > 0:
            logger.info(f"{verified_count} tahmin doğrulandı.")
    
    def _print_prediction(self, result: dict):
        """Tahmin sonucunu güzelce yazdır."""
        conf = result["confidence"]
        
        # Güven seviyesi renklendirme
        if conf >= 0.7:
            strength = "GÜÇLÜ"
        elif conf >= 0.6:
            strength = "ORTA"
        else:
            strength = "ZAYIF"
        
        # Doğruluk oranı
        if self.total_predictions > 1:
            accuracy = self.correct_predictions / max(1, sum(
                1 for p in self.prediction_history if p.get("verified")
            )) * 100
            accuracy_str = f"  Canlı Doğruluk: {accuracy:.1f}%"
        else:
            accuracy_str = ""
        
        print(f"\n  {'─'*55}")
        print(f"  ⏱  {result['timestamp'][:19]}")
        print(f"  💰 BTC/USDT Mid-Price: ${result['current_mid_price']:,.2f}")
        print(f"  🎯 {config.PREDICTION_HORIZON_MIN} dk Sonra Tahmin: {result['direction']}")
        print(f"  📊 Güven: {conf:.1%} ({strength})")
        print(f"  📈 Yükseliş Olasılığı: {result['prob_up']:.1%}")
        print(f"  📉 Düşüş Olasılığı:    {result['prob_down']:.1%}")
        print(f"  🔄 Buffer: {result['buffer_size']} snapshot")
        if accuracy_str:
            print(accuracy_str)
        print(f"  {'─'*55}")
    
    def run(self, 
            prediction_interval: int = None,
            use_websocket: bool = True,
            duration_minutes: int = 0):
        """
        Canlı tahmin döngüsünü başlat.
        
        Args:
            prediction_interval: Tahmin aralığı (saniye)
            use_websocket: WebSocket mi REST mi?
            duration_minutes: Çalışma süresi (0 = sonsuz)
        """
        if prediction_interval is None:
            prediction_interval = config.LIVE_PREDICTION_INTERVAL_SEC
        
        print(f"\n{'='*60}")
        print(f"  BTC/USDT CANLI TAHMİN MOTORU")
        print(f"  Model: {self.model_type}")
        print(f"  Tahmin Aralığı: {prediction_interval} saniye")
        print(f"  Tahmin Ufku: {config.PREDICTION_HORIZON_MIN} dakika")
        print(f"  Güven Eşiği: {config.CONFIDENCE_THRESHOLD:.0%}")
        print(f"  Durdurmak için Ctrl+C")
        print(f"{'='*60}")
        
        # Model yükle
        self.load_model()
        
        # Stream başlat
        self.start_stream(use_websocket=use_websocket)
        
        # Yeterli veri birikimini bekle
        min_required = config.MIN_SAMPLES_FOR_FEATURES
        print(f"\n  Veri birikiyor... (minimum {min_required} snapshot gerekli)")
        
        self.is_running = True
        start_time = time.time()
        
        try:
            while self.is_running:
                # REST modunda snapshot al
                if self.rest_collector:
                    self._collect_rest_snapshot()
                
                # Yeterli veri var mı?
                if len(self.snapshot_buffer) < min_required:
                    remaining = min_required - len(self.snapshot_buffer)
                    print(f"\r  Birikim: {len(self.snapshot_buffer)}/{min_required} "
                          f"(~{remaining} kaldı)", end="", flush=True)
                    time.sleep(1)
                    continue
                
                # Tahmin yap
                result = self.make_prediction()
                
                if result:
                    # Güven eşiğini kontrol et
                    if result["confidence"] >= config.CONFIDENCE_THRESHOLD:
                        self._print_prediction(result)
                    else:
                        print(f"\r  [{result['timestamp'][:19]}] "
                              f"Düşük güven ({result['confidence']:.1%}), "
                              f"tahmin atlandı.", end="", flush=True)
                
                # Geçmiş tahminleri doğrula
                self.verify_past_predictions()
                
                # Süre kontrolü
                if duration_minutes > 0:
                    elapsed = (time.time() - start_time) / 60
                    if elapsed >= duration_minutes:
                        print(f"\n\n  Süre doldu ({duration_minutes} dakika).")
                        break
                
                # Sonraki tahmine kadar bekle
                time.sleep(prediction_interval)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Sistemi düzgünce kapat."""
        self.is_running = False
        
        if self.ws_stream:
            self.ws_stream.stop()
        
        # Özet yazdır
        self._print_summary()
    
    def _print_summary(self):
        """Çalışma özeti yazdır."""
        verified = [p for p in self.prediction_history if p.get("verified")]
        correct = sum(1 for p in verified if p.get("correct"))
        
        print(f"\n{'='*60}")
        print(f"  ÇALIŞMA ÖZETİ")
        print(f"{'='*60}")
        print(f"  Toplam Tahmin:       {self.total_predictions}")
        print(f"  Doğrulanan Tahmin:   {len(verified)}")
        if verified:
            print(f"  Doğru Tahmin:        {correct}")
            print(f"  Doğruluk Oranı:      {correct/len(verified)*100:.1f}%")
        print(f"  Buffer Boyutu:       {len(self.snapshot_buffer)}")
        print(f"{'='*60}\n")
        
        # Tahmin geçmişini CSV'ye kaydet
        if self.prediction_history:
            hist_df = pd.DataFrame(list(self.prediction_history))
            hist_path = os.path.join(config.LOG_DIR, "prediction_history.csv")
            hist_df.to_csv(hist_path, index=False)
            print(f"  Tahmin geçmişi kaydedildi: {hist_path}")


# ============================================================
# ANA GİRİŞ NOKTASI
# ============================================================

import os

def main():
    parser = argparse.ArgumentParser(
        description="BTC/USDT Canlı Tahmin Motoru"
    )
    
    parser.add_argument(
        "--model", type=str, default="xgboost",
        choices=["xgboost", "random_forest", "ensemble"],
        help="Kullanılacak model (varsayılan: xgboost)"
    )
    parser.add_argument(
        "--interval", type=int, default=config.LIVE_PREDICTION_INTERVAL_SEC,
        help=f"Tahmin aralığı - saniye (varsayılan: {config.LIVE_PREDICTION_INTERVAL_SEC})"
    )
    parser.add_argument(
        "--rest", action="store_true",
        help="WebSocket yerine REST API kullan"
    )
    parser.add_argument(
        "--duration", type=int, default=0,
        help="Çalışma süresi - dakika (0 = sonsuz)"
    )
    
    args = parser.parse_args()
    
    engine = LivePredictor(model_type=args.model)
    engine.run(
        prediction_interval=args.interval,
        use_websocket=not args.rest,
        duration_minutes=args.duration,
    )


if __name__ == "__main__":
    main()
