"""
============================================================
BTC/USDT Orderbook ML Prediction System - Veri Toplama Modülü
============================================================
İki yöntem sunar:
  1) REST API ile periyodik snapshot alma (eğitim verisi toplamak için)
  2) WebSocket ile gerçek zamanlı streaming (canlı tahmin için)
  
Binance public endpoint'leri kullanır - API key gerekmez.
============================================================
"""

import time
import json
import asyncio
import logging
import threading
from datetime import datetime, timezone
from collections import deque
from typing import Dict, List, Optional, Callable

import ccxt
import websockets
import pandas as pd
import numpy as np

import config

# Logger
logger = logging.getLogger(__name__)


# ============================================================
# 1. REST API ile Orderbook Snapshot Toplama
# ============================================================

class OrderbookSnapshotCollector:
    """
    ccxt kütüphanesi üzerinden Binance REST API'den
    periyodik orderbook snapshot'ları toplar.
    
    Kullanım:
        collector = OrderbookSnapshotCollector()
        collector.collect(duration_minutes=60)  # 60 dk boyunca topla
        df = collector.get_dataframe()          # Pandas DataFrame al
    """
    
    def __init__(self):
        """Exchange bağlantısını başlat (otomatik fallback destekli)."""
        self.exchange = self._create_exchange(config.EXCHANGE_ID)
        self.snapshots: List[Dict] = []     # Toplanan snapshot'lar
        self.is_collecting = False           # Toplama durumu
        
        logger.info(f"OrderbookSnapshotCollector başlatıldı: {config.SYMBOL} @ {config.EXCHANGE_ID}")
    
    @staticmethod
    def _create_exchange(exchange_id: str):
        """
        ccxt exchange nesnesi oluştur.
        Binance kısıtlı bölgelerde otomatik olarak binanceus/bybit'e geçer.
        """
        # Önce istenen borsayı dene
        fallback_order = [exchange_id, "binanceus", "bybit", "okx"]
        # Tekrarları kaldır
        seen = set()
        unique_order = []
        for eid in fallback_order:
            if eid not in seen:
                seen.add(eid)
                unique_order.append(eid)
        
        for eid in unique_order:
            try:
                exchange_class = getattr(ccxt, eid)
                ex = exchange_class({
                    "apiKey": config.API_KEY if config.API_KEY else None,
                    "secret": config.API_SECRET if config.API_SECRET else None,
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                })
                # Bağlantı testi
                ex.fetch_order_book(config.SYMBOL, limit=5)
                logger.info(f"Borsa bağlantısı başarılı: {eid}")
                return ex
            except Exception as e:
                logger.warning(f"{eid} bağlantısı başarısız: {str(e)[:100]}")
                continue
        
        # Hiçbiri çalışmazsa son denenen borsayı döndür
        logger.error("Hiçbir borsaya bağlanılamadı!")
        exchange_class = getattr(ccxt, exchange_id)
        return exchange_class({"enableRateLimit": True})
    
    def fetch_single_snapshot(self) -> Optional[Dict]:
        """
        Tek bir orderbook snapshot'ı al.
        
        Returns:
            Dict: Timestamp + 20 seviye bid/ask fiyat ve hacim verisi
            None: Hata durumunda
        """
        try:
            # ccxt ile orderbook çek (depth = 20 seviye)
            orderbook = self.exchange.fetch_order_book(
                config.SYMBOL, 
                limit=config.ORDERBOOK_DEPTH
            )
            
            timestamp = datetime.now(timezone.utc)
            
            # Snapshot'ı düz (flat) bir dict'e dönüştür
            snapshot = {
                "timestamp": timestamp.isoformat(),
                "timestamp_ms": int(timestamp.timestamp() * 1000),
            }
            
            # Bid seviyeleri (en iyi 20 alış emri)
            for i in range(config.ORDERBOOK_DEPTH):
                if i < len(orderbook["bids"]):
                    snapshot[f"bid_price_{i}"] = float(orderbook["bids"][i][0])
                    snapshot[f"bid_volume_{i}"] = float(orderbook["bids"][i][1])
                else:
                    snapshot[f"bid_price_{i}"] = np.nan
                    snapshot[f"bid_volume_{i}"] = np.nan
            
            # Ask seviyeleri (en iyi 20 satış emri)
            for i in range(config.ORDERBOOK_DEPTH):
                if i < len(orderbook["asks"]):
                    snapshot[f"ask_price_{i}"] = float(orderbook["asks"][i][0])
                    snapshot[f"ask_volume_{i}"] = float(orderbook["asks"][i][1])
                else:
                    snapshot[f"ask_price_{i}"] = np.nan
                    snapshot[f"ask_volume_{i}"] = np.nan
            
            return snapshot
            
        except ccxt.NetworkError as e:
            logger.warning(f"Ağ hatası: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"Borsa hatası: {e}")
            return None
        except Exception as e:
            logger.error(f"Beklenmeyen hata: {e}")
            return None
    
    def collect(self, duration_minutes: int = 60, 
                interval_sec: float = None,
                callback: Optional[Callable] = None) -> None:
        """
        Belirli süre boyunca periyodik snapshot topla.
        
        Args:
            duration_minutes: Toplama süresi (dakika)
            interval_sec: Her snapshot arası bekleme (saniye)
            callback: Her snapshot sonrası çağrılacak fonksiyon
        """
        if interval_sec is None:
            interval_sec = config.SNAPSHOT_INTERVAL_SEC
        
        total_seconds = duration_minutes * 60
        start_time = time.time()
        self.is_collecting = True
        snapshot_count = 0
        
        logger.info(
            f"Veri toplama başladı: {duration_minutes} dk, "
            f"her {interval_sec}s'de bir snapshot"
        )
        
        try:
            while self.is_collecting and (time.time() - start_time) < total_seconds:
                snapshot = self.fetch_single_snapshot()
                
                if snapshot is not None:
                    self.snapshots.append(snapshot)
                    snapshot_count += 1
                    
                    if callback:
                        callback(snapshot, snapshot_count)
                    
                    if snapshot_count % 100 == 0:
                        logger.info(f"Toplanan snapshot: {snapshot_count}")
                
                # Rate limit'e saygı göster
                time.sleep(interval_sec)
                
        except KeyboardInterrupt:
            logger.info("Veri toplama kullanıcı tarafından durduruldu.")
        finally:
            self.is_collecting = False
            logger.info(f"Veri toplama tamamlandı. Toplam: {snapshot_count} snapshot")
    
    def stop(self):
        """Veri toplamayı durdur."""
        self.is_collecting = False
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Toplanan snapshot'ları Pandas DataFrame'e dönüştür.
        
        Returns:
            pd.DataFrame: Tüm snapshot'lar
        """
        if not self.snapshots:
            logger.warning("Henüz snapshot toplanmamış.")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.snapshots)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        
        logger.info(f"DataFrame oluşturuldu: {df.shape[0]} satır, {df.shape[1]} sütun")
        return df
    
    def save_to_csv(self, filepath: str = None) -> str:
        """
        Toplanan veriyi CSV dosyasına kaydet.
        
        Args:
            filepath: Kayıt yolu (None ise config'den alınır)
            
        Returns:
            str: Kaydedilen dosya yolu
        """
        if filepath is None:
            filepath = config.RAW_DATA_FILE
        
        df = self.get_dataframe()
        
        if df.empty:
            logger.warning("Kaydedilecek veri yok.")
            return ""
        
        # Mevcut dosya varsa, üzerine ekle
        try:
            existing = pd.read_csv(filepath, index_col="timestamp", parse_dates=True)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]  # Tekrarları kaldır
            logger.info(f"Mevcut veriye eklendi. Toplam: {len(df)} satır")
        except FileNotFoundError:
            pass
        
        df.to_csv(filepath)
        logger.info(f"Veri kaydedildi: {filepath}")
        return filepath


# ============================================================
# 2. WebSocket ile Gerçek Zamanlı Orderbook Streaming
# ============================================================

class OrderbookWebSocketStream:
    """
    Binance WebSocket API üzerinden gerçek zamanlı 
    partial orderbook depth stream'i dinler.
    
    Kullanım:
        stream = OrderbookWebSocketStream()
        stream.start(on_snapshot=my_callback)    # Arkaplanda başlat
        latest = stream.get_latest_snapshot()     # Son snapshot'ı al
        stream.stop()                            # Durdur
    """
    
    def __init__(self, max_buffer_size: int = 10000):
        """
        Args:
            max_buffer_size: Bellekte tutulacak max snapshot sayısı
        """
        self.buffer: deque = deque(maxlen=max_buffer_size)
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._callback: Optional[Callable] = None
        
        logger.info(
            f"WebSocket stream hazır: {config.WS_STREAM_URL}"
        )
    
    async def _listen(self):
        """WebSocket bağlantısını dinle (async)."""
        reconnect_delay = 1  # İlk bağlantı gecikmesi
        max_reconnect_delay = 60
        
        while self.is_running:
            try:
                async with websockets.connect(
                    config.WS_STREAM_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info("WebSocket bağlantısı kuruldu.")
                    reconnect_delay = 1  # Başarılı bağlantıda sıfırla
                    
                    while self.is_running:
                        try:
                            raw_msg = await asyncio.wait_for(
                                ws.recv(), timeout=30
                            )
                            data = json.loads(raw_msg)
                            snapshot = self._parse_depth_message(data)
                            
                            if snapshot:
                                self.buffer.append(snapshot)
                                
                                if self._callback:
                                    self._callback(snapshot)
                                    
                        except asyncio.TimeoutError:
                            logger.debug("WebSocket timeout, ping gönderiliyor...")
                            continue
                            
            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket bağlantısı kapandı: {e}")
            except Exception as e:
                logger.error(f"WebSocket hatası: {e}")
            
            if self.is_running:
                logger.info(f"{reconnect_delay}s sonra yeniden bağlanılıyor...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
    
    def _parse_depth_message(self, data: Dict) -> Optional[Dict]:
        """
        Binance partial depth mesajını parse et.
        
        Binance partial depth format:
        {
            "lastUpdateId": 160,
            "bids": [["0.0024","10"], ...],  # [fiyat, miktar]
            "asks": [["0.0026","100"], ...]
        }
        """
        try:
            timestamp = datetime.now(timezone.utc)
            
            snapshot = {
                "timestamp": timestamp.isoformat(),
                "timestamp_ms": int(timestamp.timestamp() * 1000),
                "last_update_id": data.get("lastUpdateId", 0),
            }
            
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            
            for i in range(config.ORDERBOOK_DEPTH):
                if i < len(bids):
                    snapshot[f"bid_price_{i}"] = float(bids[i][0])
                    snapshot[f"bid_volume_{i}"] = float(bids[i][1])
                else:
                    snapshot[f"bid_price_{i}"] = np.nan
                    snapshot[f"bid_volume_{i}"] = np.nan
            
            for i in range(config.ORDERBOOK_DEPTH):
                if i < len(asks):
                    snapshot[f"ask_price_{i}"] = float(asks[i][0])
                    snapshot[f"ask_volume_{i}"] = float(asks[i][1])
                else:
                    snapshot[f"ask_price_{i}"] = np.nan
                    snapshot[f"ask_volume_{i}"] = np.nan
            
            return snapshot
            
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"Parse hatası: {e}")
            return None
    
    def _run_event_loop(self):
        """Ayrı thread'de async event loop çalıştır."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._listen())
    
    def start(self, on_snapshot: Optional[Callable] = None):
        """
        WebSocket stream'i arkaplan thread'inde başlat.
        
        Args:
            on_snapshot: Her yeni snapshot geldiğinde çağrılacak callback
        """
        if self.is_running:
            logger.warning("Stream zaten çalışıyor.")
            return
        
        self._callback = on_snapshot
        self.is_running = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        
        logger.info("WebSocket stream başlatıldı (arkaplan thread).")
    
    def stop(self):
        """WebSocket stream'i durdur."""
        self.is_running = False
        
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        
        logger.info("WebSocket stream durduruldu.")
    
    def get_latest_snapshot(self) -> Optional[Dict]:
        """Son alınan snapshot'ı döndür."""
        if self.buffer:
            return self.buffer[-1]
        return None
    
    def get_recent_snapshots(self, n: int = 100) -> List[Dict]:
        """Son n snapshot'ı döndür."""
        return list(self.buffer)[-n:]
    
    def get_dataframe(self, n: Optional[int] = None) -> pd.DataFrame:
        """
        Buffer'daki snapshot'ları DataFrame'e dönüştür.
        
        Args:
            n: Son kaç snapshot (None = tümü)
        """
        if not self.buffer:
            return pd.DataFrame()
        
        data = list(self.buffer) if n is None else list(self.buffer)[-n:]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        return df
    
    @property
    def buffer_size(self) -> int:
        """Buffer'daki snapshot sayısı."""
        return len(self.buffer)


# ============================================================
# 3. Hibrit Collector (REST + WebSocket birleşik)
# ============================================================

class HybridOrderbookCollector:
    """
    Hem REST hem WebSocket yöntemlerini birleştiren üst-seviye collector.
    
    - Eğitim verisi toplamak için REST API kullanır (yavaş ama güvenilir)
    - Canlı tahmin için WebSocket kullanır (hızlı, gerçek zamanlı)
    """
    
    def __init__(self):
        self.rest_collector = OrderbookSnapshotCollector()
        self.ws_stream = OrderbookWebSocketStream()
    
    def collect_training_data(self, duration_minutes: int = 60, 
                              interval_sec: float = 1.0) -> pd.DataFrame:
        """
        REST API ile eğitim verisi topla.
        
        Args:
            duration_minutes: Toplama süresi
            interval_sec: Snapshot aralığı
            
        Returns:
            pd.DataFrame: Toplanan ham veri
        """
        print(f"\n{'='*60}")
        print(f"  EĞİTİM VERİSİ TOPLAMA")
        print(f"  Süre: {duration_minutes} dakika")
        print(f"  Aralık: her {interval_sec} saniye")
        print(f"  Beklenen snapshot: ~{int(duration_minutes * 60 / interval_sec)}")
        print(f"{'='*60}\n")
        
        def progress_callback(snapshot, count):
            if count % 50 == 0:
                mid = (snapshot["bid_price_0"] + snapshot["ask_price_0"]) / 2
                print(f"  [{count}] Mid-Price: ${mid:,.2f}")
        
        self.rest_collector.collect(
            duration_minutes=duration_minutes,
            interval_sec=interval_sec,
            callback=progress_callback,
        )
        
        # CSV'ye kaydet
        self.rest_collector.save_to_csv()
        
        return self.rest_collector.get_dataframe()
    
    def start_live_stream(self, on_snapshot: Optional[Callable] = None):
        """WebSocket canlı stream başlat."""
        self.ws_stream.start(on_snapshot=on_snapshot)
    
    def stop_live_stream(self):
        """WebSocket canlı stream durdur."""
        self.ws_stream.stop()
    
    def get_live_dataframe(self, n: Optional[int] = None) -> pd.DataFrame:
        """Canlı stream'den DataFrame al."""
        return self.ws_stream.get_dataframe(n)


# ============================================================
# TEST / Demo
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    print("=" * 60)
    print("  ORDERBOOK VERİ TOPLAMA - DEMO")
    print("=" * 60)
    
    # REST API ile tek snapshot test
    collector = OrderbookSnapshotCollector()
    snapshot = collector.fetch_single_snapshot()
    
    if snapshot:
        mid = (snapshot["bid_price_0"] + snapshot["ask_price_0"]) / 2
        spread = snapshot["ask_price_0"] - snapshot["bid_price_0"]
        print(f"\n  Sembol:     {config.SYMBOL}")
        print(f"  Best Bid:   ${snapshot['bid_price_0']:,.2f}")
        print(f"  Best Ask:   ${snapshot['ask_price_0']:,.2f}")
        print(f"  Mid-Price:  ${mid:,.2f}")
        print(f"  Spread:     ${spread:,.2f} ({spread/mid*100:.4f}%)")
        print(f"  Zaman:      {snapshot['timestamp']}")
        print(f"\n  Toplam {config.ORDERBOOK_DEPTH} seviye bid + ask alındı.")
    else:
        print("  HATA: Snapshot alınamadı!")
