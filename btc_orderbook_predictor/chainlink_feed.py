"""
============================================================
Chainlink BTC/USD On-Chain Fiyat Okuyucu
============================================================
Polymarket, BTC 5-dakikalık Up/Down marketlerini Chainlink
Data Streams (BTC/USD) ile resolve eder.

Bu modül, Polygon Mainnet üzerindeki Chainlink BTC/USD
Price Feed'den (0xc907E116054Ad103354f2D350FD2514433D57F6f)
fiyat verisini doğrudan okur.

Kaynak: https://data.chain.link/feeds/polygon/mainnet/btc-usd
Polymarket Resolution: https://data.chain.link/streams/btc-usd

Özellikler:
  - Polygon RPC üzerinden on-chain fiyat okuma
  - Birden fazla RPC endpoint desteği (failover)
  - Fiyat geçmişi takibi (pencere başı/sonu karşılaştırma)
  - Binance ile spread takibi
============================================================
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
from collections import deque

import requests

logger = logging.getLogger(__name__)

# ============================================================
# CHAINLINK POLYGON MAINNET BTC/USD PRICE FEED
# ============================================================
# Contract: 0xc907E116054Ad103354f2D350FD2514433D57F6f
# Decimals: 8
# Deviation Threshold: 0.1%
# Heartbeat: ~27s (çok sık güncellenir)
# ============================================================

CHAINLINK_BTC_USD_POLYGON = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
CHAINLINK_DECIMALS = 8

# latestRoundData() function selector
LATEST_ROUND_DATA_SELECTOR = "0xfeaf968c"

# Polygon RPC endpoint'leri (öncelik sırasına göre)
POLYGON_RPC_ENDPOINTS = [
    "https://polygon-bor-rpc.publicnode.com",
    "https://polygon-rpc.com",
    "https://rpc-mainnet.matic.quiknode.pro",
    "https://polygon.drpc.org",
]

# Fiyat geçmişi buffer'ı
_price_history: deque = deque(maxlen=1000)
_last_rpc_index = 0


def get_chainlink_btc_price() -> Optional[Dict]:
    """
    Chainlink BTC/USD fiyatını Polygon on-chain oracle'dan oku.
    
    Polymarket'in resolution kaynağıyla aynı veriyi okur.
    Birden fazla RPC endpoint dener (failover).
    
    Returns:
        Dict: {
            price: float,          # BTC/USD fiyatı
            round_id: int,         # Chainlink round ID
            updated_at: datetime,  # Son güncelleme zamanı (UTC)
            age_seconds: float,    # Kaç saniye önce güncellendi
            source: str,           # Kullanılan RPC endpoint
        }
        None: Tüm endpoint'ler başarısız olursa
    """
    global _last_rpc_index
    
    # RPC endpoint'lerini dene (son başarılı olandan başla)
    endpoints = POLYGON_RPC_ENDPOINTS[_last_rpc_index:] + POLYGON_RPC_ENDPOINTS[:_last_rpc_index]
    
    for i, rpc_url in enumerate(endpoints):
        try:
            result = _read_latest_round_data(rpc_url)
            if result:
                # Başarılı endpoint'i hatırla
                _last_rpc_index = (POLYGON_RPC_ENDPOINTS.index(rpc_url))
                
                # Geçmişe kaydet
                _price_history.append({
                    "price": result["price"],
                    "timestamp": result["updated_at"],
                    "fetched_at": datetime.now(timezone.utc),
                })
                
                return result
                
        except Exception as e:
            logger.debug(f"RPC hatası ({rpc_url}): {e}")
            continue
    
    logger.error("Tüm Polygon RPC endpoint'leri başarısız oldu!")
    return None


def _read_latest_round_data(rpc_url: str) -> Optional[Dict]:
    """
    Tek bir RPC endpoint'inden latestRoundData() oku.
    
    Chainlink AggregatorV3Interface.latestRoundData() dönüşü:
      (uint80 roundId, int256 answer, uint256 startedAt, 
       uint256 updatedAt, uint80 answeredInRound)
    """
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_call",
        "params": [{
            "to": CHAINLINK_BTC_USD_POLYGON,
            "data": LATEST_ROUND_DATA_SELECTOR,
        }, "latest"],
        "id": 1,
    }
    
    response = requests.post(rpc_url, json=payload, timeout=5)
    response.raise_for_status()
    
    data = response.json()
    
    if "error" in data:
        logger.debug(f"RPC hata döndü: {data['error']}")
        return None
    
    result_hex = data.get("result", "0x")
    
    # latestRoundData 5 slot döndürür = 5 * 64 hex + "0x" = 322 karakter
    if len(result_hex) < 322:
        return None
    
    # Hex parse (her slot 64 hex karakter = 32 byte)
    # Slot 0: roundId (offset 2..66)
    # Slot 1: answer  (offset 66..130)
    # Slot 2: startedAt (offset 130..194)
    # Slot 3: updatedAt (offset 194..258)
    # Slot 4: answeredInRound (offset 258..322)
    
    round_id = int(result_hex[2:66], 16)
    answer_raw = int(result_hex[66:130], 16)
    
    # int256 negatif kontrol (2's complement)
    if answer_raw >= 2**255:
        answer_raw -= 2**256
    
    started_at_ts = int(result_hex[130:194], 16)
    updated_at_ts = int(result_hex[194:258], 16)
    
    # Fiyat hesapla (8 ondalık basamak)
    price = answer_raw / (10 ** CHAINLINK_DECIMALS)
    
    # Mantık kontrolü
    if price < 100 or price > 10_000_000:
        logger.warning(f"Chainlink fiyat aralık dışı: ${price:,.2f}")
        return None
    
    now_utc = datetime.now(timezone.utc)
    updated_at = datetime.fromtimestamp(updated_at_ts, tz=timezone.utc)
    age = (now_utc - updated_at).total_seconds()
    
    # Çok eski veri kontrolü (5 dakikadan eski = sorunlu)
    if age > 300:
        logger.warning(
            f"Chainlink verisi eski: {age:.0f}s "
            f"(güncelleme: {updated_at.isoformat()})"
        )
    
    return {
        "price": price,
        "round_id": round_id,
        "updated_at": updated_at,
        "age_seconds": age,
        "source": rpc_url,
    }


# ============================================================
# FİYAT GEÇMİŞİ VE PENCERE TAKİBİ
# ============================================================

def get_price_at_time(target_time: datetime, tolerance_sec: float = 30) -> Optional[float]:
    """
    Belirli bir zamandaki Chainlink fiyatını geçmişten bul.
    
    Args:
        target_time: Hedef zaman (UTC)
        tolerance_sec: Kabul edilebilir zaman farkı (saniye)
    
    Returns:
        float: En yakın fiyat, yoksa None
    """
    if not _price_history:
        return None
    
    best_price = None
    best_diff = float("inf")
    
    for entry in _price_history:
        diff = abs((entry["timestamp"] - target_time).total_seconds())
        if diff < best_diff and diff <= tolerance_sec:
            best_diff = diff
            best_price = entry["price"]
    
    return best_price


def record_window_price(label: str) -> Optional[float]:
    """
    Mevcut Chainlink fiyatını al ve pencere etiketiyle kaydet.
    Pencere başında ve sonunda çağrılarak başlangıç/bitiş fiyatı takip edilir.
    
    Args:
        label: "window_start" veya "window_end"
    
    Returns:
        float: Kaydedilen fiyat
    """
    data = get_chainlink_btc_price()
    if data:
        logger.info(f"Chainlink [{label}]: ${data['price']:,.2f} (age: {data['age_seconds']:.0f}s)")
        return data["price"]
    return None


# ============================================================
# KARŞILAŞTIRMA: CHAINLİNK vs BINANCE
# ============================================================

def get_price_comparison() -> Dict:
    """
    Chainlink ve Binance fiyatlarını yan yana getir.
    
    Neden önemli:
      - Bot Binance orderbook'tan tahmin yapıyor
      - Polymarket Chainlink ile resolve ediyor
      - İkisi arasındaki fark (spread) model doğruluğunu etkiler
    
    Returns:
        Dict: {
            chainlink: float,
            binance: float, 
            spread_usd: float,
            spread_pct: float,
            aligned: bool (<%0.05 fark = senkronize),
        }
    """
    result = {
        "chainlink": None,
        "binance": None,
        "spread_usd": None,
        "spread_pct": None,
        "aligned": False,
    }
    
    # 1. Chainlink
    cl_data = get_chainlink_btc_price()
    if cl_data:
        result["chainlink"] = cl_data["price"]
    
    # 2. Binance (sırayla dene: global → us → bybit)
    for url in [
        "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
        "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT",
    ]:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                result["binance"] = float(resp.json()["price"])
                break
        except Exception:
            continue
    
    # Bybit fallback
    if result["binance"] is None:
        try:
            resp = requests.get(
                "https://api.bybit.com/v5/market/tickers",
                params={"category": "spot", "symbol": "BTCUSDT"},
                timeout=3,
            )
            if resp.status_code == 200:
                tickers = resp.json().get("result", {}).get("list", [])
                if tickers:
                    result["binance"] = float(tickers[0]["lastPrice"])
        except Exception:
            pass
    
    # 3. Spread hesapla
    if result["chainlink"] and result["binance"]:
        result["spread_usd"] = abs(result["chainlink"] - result["binance"])
        avg = (result["chainlink"] + result["binance"]) / 2
        result["spread_pct"] = (result["spread_usd"] / avg) * 100
        result["aligned"] = result["spread_pct"] < 0.05  # %0.05'ten az = senkronize
    
    return result


# ============================================================
# DURUM GÖSTERİMİ
# ============================================================

def print_price_status():
    """Fiyat kaynaklarının durumunu göster."""
    print(f"\n  ── Chainlink BTC/USD Fiyat Durumu ──")
    
    cl = get_chainlink_btc_price()
    if cl:
        print(f"  Chainlink: ${cl['price']:,.2f}")
        print(f"  Güncelleme: {cl['age_seconds']:.0f}s önce")
        print(f"  Round ID: {cl['round_id']}")
        print(f"  RPC: {cl['source']}")
    else:
        print(f"  Chainlink: ❌ Erişilemedi")
    
    comp = get_price_comparison()
    if comp["binance"]:
        print(f"  Binance:   ${comp['binance']:,.2f}")
    
    if comp["spread_pct"] is not None:
        icon = "✅" if comp["aligned"] else "⚠️"
        print(f"  Spread:    ${comp['spread_usd']:,.2f} ({comp['spread_pct']:.4f}%) {icon}")
    
    print()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    print("Chainlink BTC/USD Price Feed Test")
    print("=" * 50)
    
    # Fiyat oku
    data = get_chainlink_btc_price()
    if data:
        print(f"\nFiyat: ${data['price']:,.2f}")
        print(f"Round: {data['round_id']}")
        print(f"Güncelleme: {data['updated_at'].isoformat()}")
        print(f"Yaş: {data['age_seconds']:.1f} saniye")
        print(f"RPC: {data['source']}")
    else:
        print("\nFiyat okunamadı!")
    
    # Karşılaştırma
    print("\n" + "=" * 50)
    print_price_status()
