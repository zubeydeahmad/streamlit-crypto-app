# myfinancialapp/analysis/data_fetcher.py
# (Veri Çekme ve İşleme Modülü - API Çağrıları Devre Dışı Bırakıldı, Dummy Veri Kullanılıyor)

import pandas as pd
from datetime import datetime, timedelta
from django.utils import timezone # Django'nun zaman dilimi bilgisine sahip datetime objeleri için
import logging
import json
import numpy as np 
import io # StringIO için import eklendi

# Django settings'den API anahtarları ve diğer ayarları almak için
from django.conf import settings
# Kendi Django modellerimizi import ediyoruz
from .models import HistoricalData, PopularAssetCache

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler('crypto_app.log', encoding='utf-8'), # Dosyaya yaz
        logging.StreamHandler() # Konsola yaz
    ]
)
logger = logging.getLogger(__name__)

# --- Varlık Bilgileri ---
VARLIK_BILGILERI = {
    "Altın": {"sembol": "GC=F", "kaynak": "yfinance", "tip": "emtia"},
    "Gümüş": {"sembol": "SI=F", "kaynak": "yfinance", "tip": "emtia"},
    "Ham Petrol": {"sembol": "CL=F", "kaynak": "yfinance", "tip": "emtia"},
    "Bitcoin": {"sembol": "BTC", "kaynak": "coinapi", "tip": "kripto"},
    "Ethereum": {"sembol": "ETH", "kaynak": "coinapi", "tip": "kripto"},
    "Solana": {"sembol": "SOL", "kaynak": "coinapi", "tip": "kripto"},
    "Cardano": {"sembol": "ADA", "kaynak": "coinapi", "tip": "kripto"},
    "Dogecoin": {"sembol": "DOGE", "kaynak": "coinapi", "tip": "kripto"},
    "Binance Coin": {"sembol": "BNB", "kaynak": "coinapi", "tip": "kripto"},
    "Ripple": {"sembol": "XRP", "kaynak": "coinapi", "tip": "kripto"},
    "Euro/Dolar": {"sembol": "EURUSD=X", "kaynak": "yfinance", "tip": "doviz"},
    "Sterlin/Dolar": {"sembol": "GBPUSD=X", "kaynak": "yfinance", "tip": "doviz"},
    "Dolar/Türk Lirası": {"sembol": "TRY=X", "kaynak": "yfinance", "tip": "doviz"},
}


# --- Dummy API Veri Çekme Fonksiyonları ---
# Gerçek API çağrıları yerine sabit veri döndürecekler

def _fetch_yfinance_data_api(symbol: str, start: datetime, end: datetime, retries: int = 3, delay: int = 5) -> pd.DataFrame:
    """YFinance API'den geçmiş veri çeker (ŞİMDİLİK DUMMY VERİ)."""
    logger.info(f"DUMMY: yfinance'dan {symbol} verisi çekiliyor: {start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}")
    
    # inclusive='left' ile bitiş tarihini hariç tutarak yinelenenleri önle
    dates = pd.date_range(start=start, end=end, freq='D', inclusive='left')
    # Eğer tek bir gün isteniyorsa, o günü de dahil et
    if (end - start).days == 0:
        dates = pd.date_range(start=start, end=end, freq='D')

    if len(dates) == 0:
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    # Daha gerçekçi ve güncel dummy veri oluşturma
    # Sembole göre başlangıç fiyatı belirle
    if "GC=F" in symbol: # Altın
        base_price = 1900 + np.random.uniform(-50, 50)
    elif "SI=F" in symbol: # Gümüş
        base_price = 25 + np.random.uniform(-2, 2)
    elif "CL=F" in symbol: # Ham Petrol
        base_price = 70 + np.random.uniform(-5, 5)
    elif "EURUSD=X" in symbol: # Euro/Dolar
        base_price = 1.08 + np.random.uniform(-0.02, 0.02)
    elif "GBPUSD=X" in symbol: # Sterlin/Dolar
        base_price = 1.25 + np.random.uniform(-0.02, 0.02)
    elif "TRY=X" in symbol: # Dolar/Türk Lirası (ters kur)
        base_price = 32 + np.random.uniform(-1, 1)
    else:
        base_price = 100 + np.random.uniform(-10, 10) # Genel varsayılan

    prices = base_price + np.cumsum(np.random.normal(0, base_price * 0.005, len(dates))) # Fiyatı biraz daha gerçekçi yap
    
    data = {
        'Open': prices * np.random.uniform(0.99, 1.01, len(dates)),
        'High': prices * np.random.uniform(1.005, 1.02, len(dates)),
        'Low': prices * np.random.uniform(0.98, 0.995, len(dates)),
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date' # İndeks adını 'Date' olarak ayarla
    
    # İndeks üzerindeki yinelenenleri kaldır ve sırala
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    logger.info(f"DUMMY: {len(df)} adet {symbol} verisi başarıyla oluşturuldu.")
    return df


def _fetch_coinapi_data_api(crypto_symbol: str, api_key: str, currency: str = "USD", period_id: str = "1DAY", limit: int = 365, retries: int = 3, delay: int = 5) -> pd.DataFrame:
    """CoinAPI.io'dan kripto geçmiş veri çeker (ŞİMDİLİK DUMMY VERİ)."""
    logger.info(f"DUMMY: CoinAPI.io'dan {crypto_symbol}/{currency} verisi çekiliyor (limit={limit}, period={period_id}).")
    
    end_date = timezone.now()
    start_date = end_date - timedelta(days=limit - 1)
    # inclusive='left' ile bitiş tarihini hariç tutarak yinelenenleri önle
    dates = pd.date_range(start=start_date, end=end_date, freq='D', inclusive='left')
    # Eğer tek bir gün isteniyorsa, o günü de dahil et
    if (end_date - start_date).days == 0:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

    if len(dates) == 0:
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    # Daha gerçekçi ve güncel kripto dummy veri oluşturma
    # Sembole göre başlangıç fiyatı belirle
    if crypto_symbol == "BTC":
        base_price = 65000 + np.random.uniform(-5000, 5000)
    elif crypto_symbol == "ETH":
        base_price = 3500 + np.random.uniform(-300, 300)
    elif crypto_symbol == "SOL":
        base_price = 150 + np.random.uniform(-10, 10)
    elif crypto_symbol == "ADA":
        base_price = 0.45 + np.random.uniform(-0.05, 0.05)
    elif crypto_symbol == "DOGE":
        base_price = 0.15 + np.random.uniform(-0.02, 0.02)
    elif crypto_symbol == "BNB":
        base_price = 600 + np.random.uniform(-50, 50)
    elif crypto_symbol == "XRP":
        base_price = 0.50 + np.random.uniform(-0.05, 0.05)
    else:
        base_price = 100 + np.random.uniform(-10, 10) # Genel varsayılan

    prices = base_price + np.cumsum(np.random.normal(0, base_price * 0.01, len(dates))) # Fiyatı biraz daha gerçekçi yap
    
    data = {
        'Open': prices * np.random.uniform(0.99, 1.01, len(dates)),
        'High': prices * np.random.uniform(1.005, 1.02, len(dates)),
        'Low': prices * np.random.uniform(0.98, 0.995, len(dates)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date' # İndeks adını 'Date' olarak ayarla

    # İndeks üzerindeki yinelenenleri kaldır ve sırala
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    logger.info(f"DUMMY: {len(df)} adet {crypto_symbol} verisi başarıyla oluşturuldu.")
    return df


def _fetch_fixer_data_api(base_currency: str, target_currency: str, api_key: str, date: str, retries: int = 3, delay: int = 5) -> dict:
    """Fixer.io'dan döviz kuru çeker (ŞİMDİLİK DUMMY VERİ)."""
    logger.info(f"DUMMY: Fixer.io'dan {base_currency}/{target_currency} verisi çekiliyor (tarih={date}).")
    
    dummy_rate = np.random.uniform(1.0, 30.0) # Rastgele bir kur
    logger.info(f"DUMMY: Fixer.io'dan {base_currency}/{target_currency} için kur başarıyla oluşturuldu: {dummy_rate}.")
    return {"price": dummy_rate}


# --- Ana Veri Çekme Fonksiyonları (DB cache kontrolü ve Dummy Veri ile) ---

def fetch_all_popular_assets_and_save():
    """
    Popüler varlıkların güncel fiyatlarını çeker ve veritabanına kaydeder/günceller.
    Sadece belirli bir süre içinde güncellenmediyse yeniden çeker.
    (ŞİMDİLİK DUMMY VERİ KULLANILIYOR)
    """
    POPULAR_CACHE_DURATION_MINUTES = 5 # Önbellek süresi
    
    logger.info("DUMMY: Popüler varlıkların güncel fiyatları çekiliyor (Veritabanı öncelikli).")
    overview_data = []

    for asset_name, info in VARLIK_BILGILERI.items():
        symbol = info["sembol"] # Buradaki symbol, veritabanına kaydedilecek semboldür (BTC, GC=F gibi).

        # PopularAssetCache'teki asset_name alanı, VARLIK_BILGILERI'ndeki 'sembol' ile eşleşmelidir.
        cached_record = PopularAssetCache.objects.filter(asset_name=symbol).first()
        
        price = np.nan # Varsayılan olarak NaN
        change_percent = np.nan # Varsayılan olarak NaN

        # Eğer önbellek kaydı yoksa veya süresi dolmuşsa dummy veriyi yeniden oluştur
        if not cached_record or cached_record.last_updated < timezone.now() - timedelta(minutes=POPULAR_CACHE_DURATION_MINUTES):
            logger.info(f"DUMMY: Popüler varlık '{asset_name}' ({symbol}): Veritabanında güncel veri bulunamadı veya eski, dummy veri oluşturuluyor.")
            
            # Dummy fiyat ve değişim yüzdesi oluştur
            if "GC=F" in symbol: # Altın
                price = 1900 + np.random.uniform(-50, 50)
            elif "SI=F" in symbol: # Gümüş
                price = 25 + np.random.uniform(-2, 2)
            elif "CL=F" in symbol: # Ham Petrol
                price = 70 + np.random.uniform(-5, 5)
            elif "EURUSD=X" in symbol: # Euro/Dolar
                price = 1.08 + np.random.uniform(-0.02, 0.02)
            elif "GBPUSD=X" in symbol: # Sterlin/Dolar
                price = 1.25 + np.random.uniform(-0.02, 0.02)
            elif "TRY=X" in symbol: # Dolar/Türk Lirası (ters kur)
                price = 32 + np.random.uniform(-1, 1)
            elif "BTC" in symbol:
                price = 65000 + np.random.uniform(-5000, 5000)
            elif "ETH" in symbol:
                price = 3500 + np.random.uniform(-300, 300)
            elif "SOL" in symbol:
                price = 150 + np.random.uniform(-10, 10)
            elif "ADA" in symbol:
                price = 0.45 + np.random.uniform(-0.05, 0.05)
            elif "DOGE" in symbol:
                price = 0.15 + np.random.uniform(-0.02, 0.02)
            elif "BNB" in symbol:
                price = 600 + np.random.uniform(-50, 50)
            elif "XRP" in symbol:
                price = 0.50 + np.random.uniform(-0.05, 0.05)
            else:
                price = np.random.uniform(10, 1000) # Genel varsayılan

            change_percent = np.random.uniform(-5, 5) # -5% ile +5% arası değişim

            if price is not None and not np.isnan(price):
                PopularAssetCache.objects.update_or_create(
                    asset_name=symbol, # Veritabanına sembol olarak kaydediyoruz
                    defaults={'price': price, 'change_percent': change_percent if change_percent is not None and not np.isnan(change_percent) else np.nan, 'last_updated': timezone.now()}
                )
                logger.info(f"DUMMY: {asset_name} ({symbol}) dummy verisi veritabanına kaydedildi/güncellendi.")
            else:
                logger.info(f"DUMMY: {asset_name} ({symbol}) için fiyat oluşturulamadı veya geçersiz fiyat, veritabanına kaydedilmedi.")
        else: # Önbellek güncel ise
            price = cached_record.price
            change_percent = cached_record.change_percent
            logger.info(f"DUMMY: Popüler varlık '{asset_name}' ({symbol}): Veritabanından güncel veri yüklendi.")

        # DataFrame'e eklerken 'Varlık' sütununa tam adı, diğerlerine sembole ait veriyi ekliyoruz
        overview_data.append({
            "Varlık": asset_name, # Tabloda gösterilecek tam isim
            "Fiyat": price,
            "Değişim (%)": change_percent
        })

    df_overview = pd.DataFrame(overview_data)
    logger.info(f"DUMMY: Popüler varlıklar özeti DataFrame'e dönüştürüldü. Boyut: {df_overview.shape}")
    return df_overview


def get_historical_data_from_db_or_fetch(asset_symbol: str, start_date: datetime, end_date: datetime, force_fetch: bool = False) -> pd.DataFrame:
    """
    Belirtilen varlık için geçmiş verileri veritabanından çeker veya gerekirse dummy veri oluşturur.
    (ŞİMDİLİK DUMMY VERİ KULLANILIYOR)
    """
    logger.info(f"DUMMY: get_historical_data_from_db_or_fetch çağrıldı: Sembol={asset_symbol}, Başlangıç={start_date.strftime('%Y-%m-%d')}, Bitiş={end_date.strftime('%Y-%m-%d')}")

    HISTORY_CACHE_DURATION_HOURS = 24 * 7 

    # VARLIK_BILGILERI'nden asset_symbol'a göre varlık bilgisini bul
    asset_info = next((info for name, info in VARLIK_BILGILERI.items() if info["sembol"] == asset_symbol), None)
    if not asset_info:
        logger.error(f"Hata: '{asset_symbol}' için varlık bilgisi bulunamadı.")
        return pd.DataFrame()
    
    asset_type = asset_info["tip"]
    
    try:
        historical_record = HistoricalData.objects.get(asset_symbol=asset_symbol)
        
        # HATA DÜZELTME: timezone.now() kullanıldı
        if historical_record.last_updated < timezone.now() - timedelta(hours=HISTORY_CACHE_DURATION_HOURS) or force_fetch: 
            logger.info(f"DUMMY: '{asset_symbol}' için eski veri bulundu veya yeniden çekme isteniyor, dummy veri oluşturuluyor...")
            
            # Dummy veri oluştur
            if asset_type == "kripto":
                # settings.COINAPI_API_KEY yerine boş string, çünkü settings'e erişemiyoruz burada
                # limit parametresi, start_date ve end_date arasındaki gün sayısına göre ayarlandı
                fetched_data = _fetch_coinapi_data_api(asset_symbol, "", limit=(end_date - start_date).days + 1)
            else:
                # start_date ve end_date doğrudan yfinance dummy fonksiyonuna iletildi
                fetched_data = _fetch_yfinance_data_api(asset_symbol, start_date, end_date)

            if not fetched_data.empty:
                # Veritabanına kaydetmeden önce indeksteki potansiyel yinelenenleri kaldır
                fetched_data = fetched_data[~fetched_data.index.duplicated(keep='first')]
                fetched_data = fetched_data.sort_index() # İndeksi sırala
                historical_record.data_json = fetched_data.to_json(orient='split', date_format='iso')
                historical_record.last_updated = timezone.now() # HATA DÜZELTME: timezone.now() kullanıldı
                historical_record.save()
                logger.info(f"DUMMY: '{asset_symbol}' için geçmiş dummy veri güncellendi ve kaydedildi.")
                fetched_data.index = pd.to_datetime(fetched_data.index)
                # Tarih aralığına göre filtrelemeden önce indeksteki potansiyel yinelenenleri kaldır
                fetched_data = fetched_data[~fetched_data.index.duplicated(keep='first')]
                fetched_data = fetched_data.sort_index() # İndeksi sırala
                return fetched_data[(fetched_data.index.date >= start_date.date()) & (fetched_data.index.date <= end_date.date())].copy()
            else:
                logger.warning(f"DUMMY: '{asset_symbol}' için yeni dummy veri oluşturulamadı, eski veri kullanılıyor.")
                # StringIO kullanımı eklendi
                df_from_db_old = pd.read_json(io.StringIO(historical_record.data_json), orient='split')
                df_from_db_old.index = pd.to_datetime(df_from_db_old.index)
                # Veritabanından okuduktan sonra indeksteki potansiyel yinelenenleri kaldır
                df_from_db_old = df_from_db_old[~df_from_db_old.index.duplicated(keep='first')]
                df_from_db_old = df_from_db_old.sort_index() # İndeksi sırala
                return df_from_db_old[(df_from_db_old.index.date >= start_date.date()) & (df_from_db_old.index.date <= end_date.date())].copy()
        else:
            logger.info(f"DUMMY: '{asset_symbol}' için güncel veri veritabanından alınıyor.")
            # StringIO kullanımı eklendi
            df_from_db_fresh = pd.read_json(io.StringIO(historical_record.data_json), orient='split')
            df_from_db_fresh.index = pd.to_datetime(df_from_db_fresh.index)
            # Veritabanından okuduktan sonra indeksteki potansiyel yinelenenleri kaldır
            df_from_db_fresh = df_from_db_fresh[~df_from_db_fresh.index.duplicated(keep='first')]
            df_from_db_fresh = df_from_db_fresh.sort_index() # İndeksi sırala
            return df_from_db_fresh[(df_from_db_fresh.index.date >= start_date.date()) & (df_from_db_fresh.index.date <= end_date.date())].copy()

    except HistoricalData.DoesNotExist:
        logger.info(f"DUMMY: '{asset_symbol}' için veri veritabanında bulunamadı, dummy veri oluşturuluyor...")
        
        fetched_data = pd.DataFrame()
        if asset_type == "kripto":
            limit_days = (end_date - start_date).days + 1
            # settings.COINAPI_API_KEY yerine boş string, çünkü settings'e erişemiyoruz burada
            fetched_data = _fetch_coinapi_data_api(asset_symbol, "", limit=limit_days) 
        else:
            fetched_data = _fetch_yfinance_data_api(asset_symbol, start_date, end_date)

        if not fetched_data.empty:
            # Veritabanına kaydetmeden önce indeksteki potansiyel yinelenenleri kaldır
            fetched_data = fetched_data[~fetched_data.index.duplicated(keep='first')]
            fetched_data = fetched_data.sort_index() # İndeksi sırala
            HistoricalData.objects.create(
                asset_symbol=asset_symbol,
                data_json=fetched_data.to_json(orient='split', date_format='iso'),
                last_updated=timezone.now() # HATA DÜZELTME: timezone.now() kullanıldı
            )
            logger.info(f"DUMMY: '{asset_symbol}' için geçmiş dummy veri oluşturuldu ve veritabanına kaydedildi.")
            fetched_data.index = pd.to_datetime(fetched_data.index)
            # Tarih aralığına göre filtrelemeden önce indeksteki potansiyel yinelenenleri kaldır
            fetched_data = fetched_data[~fetched_data.index.duplicated(keep='first')]
            fetched_data = fetched_data.sort_index() # İndeksi sırala
            return fetched_data[(fetched_data.index.date >= start_date.date()) & (fetched_data.index.date <= end_date.date())].copy()
        else:
            logger.error(f"Hata: '{asset_symbol}' için geçmiş dummy veri oluşturulamadı. Boş DataFrame döndürüldü.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Hata: Geçmiş dummy veri işlenirken bir sorun oluştu: {e}", exc_info=True)
        return pd.DataFrame()
