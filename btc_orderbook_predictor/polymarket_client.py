"""
============================================================
BTC/USDT Orderbook ML Tahmin Sistemi - Polymarket Entegrasyonu
============================================================
Polymarket CLOB API üzerinden BTC 5-dakikalık mum 
tahmin marketlerinde otomatik işlem yapar.

Fonksiyonlar:
  - Polymarket 5-dk pencere zamanlamasını hesapla ve senkronize et
  - Aktif BTC Up/Down marketlerini bul
  - Market fiyat ve orderbook verisi çek
  - Up/Down pozisyonu aç (market order)
  - Açık pozisyonları takip et
  - Dry-run (simülasyon) modu

Fiyat Kaynakları:
  - Chainlink BTC/USD: chainlink_feed.py (Polymarket resolution kaynağı)
  - Binance Orderbook: data_collector.py (ML feature kaynağı)
============================================================
"""

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List

import requests

import config
from chainlink_feed import (
    get_chainlink_btc_price,
    get_price_comparison,
    print_price_status,
)

logger = logging.getLogger(__name__)

# Polymarket SDK - sadece gerektiğinde import et
_clob_client = None


def _get_clob_client():
    """
    Polymarket CLOB Client'ı lazy-load ile başlat.
    İlk çağrıda oluşturulur, sonraki çağrılarda cache'den döner.
    """
    global _clob_client
    
    if _clob_client is not None:
        return _clob_client
    
    if not config.POLY_PRIVATE_KEY:
        logger.warning("POLY_PRIVATE_KEY ayarlanmamış. Sadece okuma modunda çalışılacak.")
        return None
    
    try:
        from py_clob_client.client import ClobClient
        
        client = ClobClient(
            config.POLY_CLOB_API,
            key=config.POLY_PRIVATE_KEY,
            chain_id=config.POLY_CHAIN_ID,
            signature_type=config.POLY_SIGNATURE_TYPE,
            funder=config.POLY_FUNDER_ADDRESS if config.POLY_FUNDER_ADDRESS else None,
        )
        
        # L2 API kimlik bilgilerini oluştur/türet
        client.set_api_creds(client.create_or_derive_api_creds())
        
        logger.info("Polymarket CLOB client başarıyla başlatıldı.")
        _clob_client = client
        return client
        
    except ImportError:
        logger.error(
            "py-clob-client kurulu değil! "
            "Kurmak için: pip install py-clob-client"
        )
        return None
    except Exception as e:
        logger.error(f"Polymarket client başlatma hatası: {e}")
        return None


# ============================================================
# 1. POLYMARKET 5-DK PENCERE ZAMANLAMA SİSTEMİ
# ============================================================

def get_current_5min_window() -> Dict:
    """
    Şu anki Polymarket 5-dakikalık pencereyi hesapla.
    
    Polymarket 5-dk marketleri sabit pencerelerle çalışır:
      12:00-12:05, 12:05-12:10, 12:10-12:15, ... (ET zaman dilimi)
      Her pencere tam 5 dakikada bir başlar (dakika % 5 == 0)
    
    Returns:
        Dict: {
            window_start: datetime (UTC),
            window_end: datetime (UTC),
            seconds_elapsed: pencere başından beri (sn),
            seconds_remaining: pencere bitişine (sn),
            minutes_remaining: pencere bitişine (dk),
            is_early: bool (ilk 60 saniye),
            is_late: bool (son 60 saniye),
            window_label: str (örn. "4:20-4:25 AM ET"),
        }
    """
    now_utc = datetime.now(timezone.utc)
    
    # Dakikayı 5'in katına yuvarla (aşağı)
    total_minutes = now_utc.hour * 60 + now_utc.minute
    window_start_min = (total_minutes // 5) * 5
    
    window_start = now_utc.replace(
        hour=window_start_min // 60,
        minute=window_start_min % 60,
        second=0,
        microsecond=0,
    )
    window_end = window_start + timedelta(minutes=5)
    
    elapsed = (now_utc - window_start).total_seconds()
    remaining = (window_end - now_utc).total_seconds()
    
    # ET label (EDT = UTC-4, EST = UTC-5)
    # Mart-Kasım arası EDT, Kasım-Mart arası EST
    # Basitlik için EDT kullanıyoruz (Polymarket genellikle ET gösterir)
    et_offset = timedelta(hours=-4)
    et_start = window_start + et_offset
    et_end = window_end + et_offset
    
    start_str = et_start.strftime("%I:%M").lstrip("0")
    end_str = et_end.strftime("%I:%M").lstrip("0")
    ampm = et_end.strftime("%p").upper()
    
    return {
        "window_start": window_start,
        "window_end": window_end,
        "seconds_elapsed": elapsed,
        "seconds_remaining": remaining,
        "minutes_remaining": remaining / 60,
        "is_early": elapsed < 60,
        "is_late": remaining < 60,
        "window_label": f"{start_str}-{end_str} {ampm} ET",
    }


def get_next_5min_window() -> Dict:
    """
    Bir sonraki 5-dk pencereyi hesapla.
    
    Returns:
        Dict: Sonraki pencere bilgisi + seconds_until_start
    """
    current = get_current_5min_window()
    next_start = current["window_end"]
    next_end = next_start + timedelta(minutes=5)
    
    now_utc = datetime.now(timezone.utc)
    seconds_until = (next_start - now_utc).total_seconds()
    
    et_offset = timedelta(hours=-4)
    et_start = next_start + et_offset
    et_end = next_end + et_offset
    
    start_str = et_start.strftime("%I:%M").lstrip("0")
    end_str = et_end.strftime("%I:%M").lstrip("0")
    ampm = et_end.strftime("%p").upper()
    
    return {
        "window_start": next_start,
        "window_end": next_end,
        "seconds_until_start": seconds_until,
        "window_label": f"{start_str}-{end_str} {ampm} ET",
    }


# ============================================================
# 2. MARKET KEŞFETME (BTC 5-Dakika Up/Down Marketleri)
# ============================================================

def find_btc_5min_markets() -> List[Dict]:
    """
    Polymarket Gamma API üzerinden aktif BTC 5-dakikalık
    Up/Down tahmin marketlerini bul.
    """
    btc_markets = []
    
    # --- YÖNTEM 1: Events API (öncelikli) ---
    try:
        response = requests.get(
            f"{config.POLY_GAMMA_API}/events",
            params={
                "limit": 100,
                "active": True,
                "closed": False,
                "order": "startDate",
                "ascending": False,
            },
            timeout=10,
        )
        response.raise_for_status()
        events = response.json()
        
        for event in events:
            title = event.get("title", "").lower()
            slug = event.get("slug", "").lower()
            
            is_btc_5min = (
                ("btc" in slug or "bitcoin" in title)
                and ("5m" in slug or "5 min" in title or "5-min" in title)
                and ("updown" in slug or "up or down" in title or "up/down" in title)
            )
            
            if not is_btc_5min:
                continue
            
            for m in event.get("markets", []):
                market_info = _parse_market(m)
                if market_info:
                    btc_markets.append(market_info)
    
    except requests.RequestException as e:
        logger.warning(f"Events API hatası: {e}")
    
    # --- YÖNTEM 2: Markets API (fallback) ---
    if not btc_markets:
        try:
            response = requests.get(
                f"{config.POLY_GAMMA_API}/markets",
                params={
                    "limit": 100,
                    "active": True,
                    "closed": False,
                    "order": "endDate",
                    "ascending": True,
                },
                timeout=10,
            )
            response.raise_for_status()
            all_markets = response.json()
            
            for m in all_markets:
                question = m.get("question", "").lower()
                is_btc_5min = (
                    ("bitcoin" in question or "btc" in question)
                    and ("up or down" in question or "up/down" in question)
                    and ("5 min" in question or "5-min" in question or "5min" in question)
                )
                if is_btc_5min:
                    market_info = _parse_market(m)
                    if market_info:
                        btc_markets.append(market_info)
        
        except requests.RequestException as e:
            logger.warning(f"Markets API hatası: {e}")
    
    logger.info(f"{len(btc_markets)} aktif BTC 5-dk market bulundu.")
    return btc_markets


def _parse_market(m: Dict) -> Optional[Dict]:
    """Tek bir market dict'ini standart formata parse et."""
    try:
        raw_ids = m.get("clobTokenIds", "[]")
        token_ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
        
        raw_outcomes = m.get("outcomes", "[]")
        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
        
        raw_prices = m.get("outcomePrices", "[]")
        outcome_prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
        
        if len(token_ids) < 2:
            return None
        
        up_idx, down_idx = 0, 1
        for i, outcome in enumerate(outcomes):
            ol = outcome.lower()
            if ol in ("up", "yes"):
                up_idx = i
            elif ol in ("down", "no"):
                down_idx = i
        
        return {
            "condition_id": m.get("conditionId", ""),
            "question": m.get("question", ""),
            "slug": m.get("slug", ""),
            "end_date": m.get("endDate", ""),
            "up_token_id": token_ids[up_idx],
            "down_token_id": token_ids[down_idx],
            "up_price": float(outcome_prices[up_idx]) if outcome_prices else 0.5,
            "down_price": float(outcome_prices[down_idx]) if outcome_prices else 0.5,
            "outcomes": outcomes,
            "volume": float(m.get("volume", 0) or 0),
            "liquidity": float(m.get("liquidityNum", 0) or 0),
        }
    except (json.JSONDecodeError, ValueError, IndexError, KeyError) as e:
        logger.debug(f"Market parse hatası: {e}")
        return None


def find_next_market() -> Optional[Dict]:
    """Sıradaki (en yakın kapanışlı) BTC 5-dk marketini bul."""
    markets = find_btc_5min_markets()
    
    if not markets:
        logger.warning("Aktif BTC 5-dk market bulunamadı.")
        return None
    
    now = datetime.now(timezone.utc)
    upcoming = []
    
    for m in markets:
        try:
            end = datetime.fromisoformat(m["end_date"].replace("Z", "+00:00"))
            if end > now:
                m["_end_dt"] = end
                m["_minutes_left"] = (end - now).total_seconds() / 60
                upcoming.append(m)
        except (ValueError, KeyError):
            continue
    
    if not upcoming:
        return None
    
    upcoming.sort(key=lambda x: x["_end_dt"])
    next_market = upcoming[0]
    
    logger.info(
        f"Sıradaki market: {next_market['question']} "
        f"({next_market['_minutes_left']:.1f} dk kaldı)"
    )
    return next_market


def find_market_for_window(window_start: datetime) -> Optional[Dict]:
    """
    Belirli bir 5-dk pencereye ait Polymarket marketini bul.
    
    Args:
        window_start: Pencere başlangıç zamanı (UTC)
    
    Returns:
        Dict: Pencereye eşleşen market, yoksa None
    """
    markets = find_btc_5min_markets()
    window_end = window_start + timedelta(minutes=5)
    
    for m in markets:
        try:
            end = datetime.fromisoformat(m["end_date"].replace("Z", "+00:00"))
            # Pencere bitiş zamanı ±30 saniye toleransla eşleşiyor mu?
            diff = abs((end - window_end).total_seconds())
            if diff < 30:
                m["_end_dt"] = end
                m["_minutes_left"] = (end - datetime.now(timezone.utc)).total_seconds() / 60
                return m
        except (ValueError, KeyError):
            continue
    
    return None


# ============================================================
# 3. FİYAT VE ORDERBOOK VERİSİ
# ============================================================

def get_market_prices(market: Dict) -> Dict:
    """Bir market için güncel fiyat bilgilerini al."""
    client = _get_clob_client()
    
    result = {
        "up_price": market.get("up_price", 0.5),
        "down_price": market.get("down_price", 0.5),
    }
    
    if client:
        try:
            up_mid = client.get_midpoint(market["up_token_id"])
            result["up_midpoint"] = float(up_mid.get("mid", 0.5))
            
            up_spread = client.get_spread(market["up_token_id"])
            result["spread"] = float(up_spread.get("spread", 0))
        except Exception as e:
            logger.warning(f"Fiyat bilgisi alınamadı: {e}")
    
    return result


# ============================================================
# 4. İŞLEM YAPMA (Order Placement)
# ============================================================

def place_bet(
    market: Dict,
    direction: str,
    amount_usdc: float = None,
    dry_run: bool = None,
) -> Dict:
    """
    Polymarket üzerinde BTC Up/Down bahsi aç.
    """
    if amount_usdc is None:
        amount_usdc = config.POLY_BET_AMOUNT_USDC
    if dry_run is None:
        dry_run = config.POLY_DRY_RUN
    
    if direction.lower() == "up":
        token_id = market["up_token_id"]
        price = market.get("up_price", 0.5)
    else:
        token_id = market["down_token_id"]
        price = market.get("down_price", 0.5)
    
    window = get_current_5min_window()
    
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "direction": direction.upper(),
        "token_id": token_id,
        "amount_usdc": amount_usdc,
        "price": price,
        "market_question": market.get("question", ""),
        "window_label": window["window_label"],
        "window_remaining_sec": window["seconds_remaining"],
        "dry_run": dry_run,
        "success": False,
        "order_id": None,
        "error": None,
    }
    
    # --- DRY RUN ---
    if dry_run:
        estimated_shares = amount_usdc / price if price > 0 else 0
        result["success"] = True
        result["order_id"] = f"DRY-RUN-{int(time.time())}"
        result["estimated_shares"] = estimated_shares
        result["mode"] = "SIMULASYON"
        
        # Chainlink fiyatı
        cl = get_chainlink_btc_price()
        cl_price_str = f"${cl['price']:,.2f}" if cl else "N/A"
        
        print(f"\n  🔸 [DRY-RUN] İşlem Simülasyonu:")
        print(f"     Pencere:  {window['window_label']}")
        print(f"     Yön:      {direction.upper()}")
        print(f"     Tutar:    ${amount_usdc:.2f} USDC")
        print(f"     Fiyat:    {price:.4f}")
        print(f"     Pay:      {estimated_shares:.2f}")
        print(f"     Chainlink: {cl_price_str}")
        print(f"     Market:   {market.get('question', '')[:60]}")
        print(f"     Kalan:    {window['seconds_remaining']:.0f}s")
        
        logger.info(f"DRY-RUN: {direction.upper()} ${amount_usdc} @ {price}")
        return result
    
    # --- GERÇEK İŞLEM ---
    client = _get_clob_client()
    
    if not client:
        result["error"] = "Polymarket client başlatılamadı. Private key kontrol edin."
        logger.error(result["error"])
        return result
    
    try:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY
        
        market_order = MarketOrderArgs(
            token_id=token_id,
            amount=amount_usdc,
            side=BUY,
            order_type=OrderType.FOK,
        )
        
        signed_order = client.create_market_order(market_order)
        response = client.post_order(signed_order, OrderType.FOK)
        
        result["success"] = True
        result["order_id"] = response.get("orderID", response.get("id", "unknown"))
        result["status"] = response.get("status", "unknown")
        result["mode"] = "GERÇEK"
        
        print(f"\n  ✅ İŞLEM GERÇEKLEŞTİRİLDİ:")
        print(f"     Pencere: {window['window_label']}")
        print(f"     Yön:     {direction.upper()}")
        print(f"     Tutar:   ${amount_usdc:.2f} USDC")
        print(f"     Order:   {result['order_id']}")
        
        logger.info(
            f"GERÇEK İŞLEM: {direction.upper()} ${amount_usdc} "
            f"Order: {result['order_id']} Status: {result['status']}"
        )
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"İşlem hatası: {e}")
        print(f"\n  ❌ İŞLEM HATASI: {e}")
    
    return result


# ============================================================
# 5. POZİSYON TAKİBİ
# ============================================================

def get_open_positions() -> List[Dict]:
    """Mevcut açık pozisyonları getir."""
    client = _get_clob_client()
    if not client:
        return []
    try:
        from py_clob_client.clob_types import OpenOrderParams
        orders = client.get_orders(OpenOrderParams())
        return orders if isinstance(orders, list) else []
    except Exception as e:
        logger.warning(f"Pozisyon sorgulama hatası: {e}")
        return []


def get_balance() -> Optional[float]:
    """USDC bakiyesini kontrol et."""
    client = _get_clob_client()
    if not client:
        return None
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        balance = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        return float(balance.get("balance", 0))
    except Exception as e:
        logger.warning(f"Bakiye sorgulama hatası: {e}")
        return None


# ============================================================
# 6. DURUM KONTROLÜ
# ============================================================

def check_polymarket_status() -> Dict:
    """Polymarket + Chainlink bağlantı durumunu kontrol et."""
    status = {
        "api_ok": False,
        "auth_ok": False,
        "balance": None,
        "active_markets": 0,
        "next_market": None,
        "dry_run": config.POLY_DRY_RUN,
        "chainlink_price": None,
        "binance_price": None,
        "current_window": None,
    }
    
    # API erişimi
    try:
        response = requests.get(
            f"{config.POLY_GAMMA_API}/markets",
            params={"limit": 1, "active": True},
            timeout=5,
        )
        status["api_ok"] = response.status_code == 200
    except Exception:
        pass
    
    # Pencere
    window = get_current_5min_window()
    status["current_window"] = window["window_label"]
    
    # Chainlink + Binance
    comp = get_price_comparison()
    status["chainlink_price"] = comp.get("chainlink")
    status["binance_price"] = comp.get("binance")
    
    # Market keşfi
    markets = find_btc_5min_markets()
    status["active_markets"] = len(markets)
    
    next_m = find_next_market()
    if next_m:
        status["next_market"] = {
            "question": next_m["question"],
            "minutes_left": next_m.get("_minutes_left", 0),
            "up_price": next_m.get("up_price", 0),
            "down_price": next_m.get("down_price", 0),
        }
    
    # Kimlik doğrulama
    if config.POLY_PRIVATE_KEY:
        client = _get_clob_client()
        if client:
            status["auth_ok"] = True
            balance = get_balance()
            if balance is not None:
                status["balance"] = balance
    
    return status


def print_status():
    """Polymarket durumunu güzelce yazdır."""
    print(f"\n{'='*62}")
    print(f"  POLYMARKET + CHAINLINK DURUM KONTROLÜ")
    print(f"{'='*62}")
    
    status = check_polymarket_status()
    
    print(f"  Gamma API:     {'✅' if status['api_ok'] else '❌'}")
    print(f"  Kimlik:        {'✅' if status['auth_ok'] else '❌ (Private key gerekli)'}")
    print(f"  Mod:           {'🔸 DRY-RUN (Simülasyon)' if status['dry_run'] else '🔴 GERÇEK İŞLEM'}")
    
    if status["balance"] is not None:
        print(f"  Bakiye:        ${status['balance']:.2f} USDC")
    
    # Fiyat kaynakları
    print(f"\n  ── Fiyat Kaynakları ──")
    if status["chainlink_price"]:
        print(f"  🔗 Chainlink:  ${status['chainlink_price']:,.2f}  ← Polymarket resolution")
    else:
        print(f"  🔗 Chainlink:  ❌ Erişilemedi")
    
    if status["binance_price"]:
        print(f"  📊 Binance:    ${status['binance_price']:,.2f}  ← ML veri kaynağı")
    
    if status["chainlink_price"] and status["binance_price"]:
        spread = abs(status["chainlink_price"] - status["binance_price"])
        pct = spread / status["chainlink_price"] * 100
        icon = "✅" if pct < 0.05 else "⚠️"
        print(f"  {icon} Spread:    ${spread:,.2f} ({pct:.4f}%)")
    
    # Pencere & Market
    print(f"\n  ── Pencere & Market ──")
    if status["current_window"]:
        print(f"  Şu an:         {status['current_window']}")
    
    print(f"  Aktif Market:  {status['active_markets']}")
    
    if status["next_market"]:
        nm = status["next_market"]
        print(f"\n  Sıradaki Market:")
        print(f"    {nm['question'][:60]}")
        print(f"    Up: {nm['up_price']:.2f} | Down: {nm['down_price']:.2f}")
        print(f"    Kalan: {nm['minutes_left']:.1f} dakika")
    
    print(f"{'='*62}\n")
    
    return status


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    print_status()
