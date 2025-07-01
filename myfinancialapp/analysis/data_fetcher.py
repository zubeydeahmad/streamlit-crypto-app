# myfinancialapp/analysis/data_fetcher.py
# (Veri Çekme ve İşleme Modülü)
# Bu modül, finansal varlıkların geçmiş verilerini çeşitli kaynaklardan çeker (yfinance, CoinAPI, Fixer.io)
# ve Django uygulamasında kullanılmak üzere işler.

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone
import requests
import logging
# import sqlite3 # Django ORM kullanıldığı için artık buna gerek yok
import json
import time # time.time() için
import numpy as np 

# Django settings'den API anahtarları ve diğer ayarları almak için
from django.conf import settings
# Kendi Django modellerimizi import ediyoruz
from .models import HistoricalData, PopularAssetCache

# Loglama yapılandırması
# Django ayarları ile de loglama yapılabilir, ancak modül bazlı loglama da geçerlidir.
logging.basicConfig(filename='crypto_app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logger = logging.getLogger(__name__)

# --- Django ORM ile Veritabanı Yönetimi ---
# Django ORM'e geçtiğimiz için bu sabitler artık models.py'deki mantık veya doğrudan kod içinde yönetilir.
# HISTORY_CACHE_DURATION_HOURS = 24 * 7 
# POPULAR_CACHE_DURATION_MINUTES = 5

# --- Varlık Bilgileri ---
# Bu sabit, Streamlit uygulamanızdan geldiği için burada tutulabilir.
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


def _fetch_yfinance_data_api(symbol: str, start: datetime, end: datetime, retries: int = 3, delay: int = 5) -> pd.DataFrame:
    """YFinance API'den geçmiş veri çeker, yeniden deneme mekanizması ve eksik sütun doldurma ile."""
    logger.info(f"yfinance'dan {symbol} verisi çekiliyor: {start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}")
    
    for attempt in range(retries):
        try:
            data = yf.download(symbol, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False, auto_adjust=False)
            
            data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]
            
            if 'Close' not in data.columns and 'Adj Close' in data.columns:
                data['Close'] = data['Adj Close']
                logger.info(f"yfinance: 'Adj Close' sütunu 'Close' olarak yeniden adlandırıldı ve kullanılıyor.")

            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            if 'Close' not in data.columns or data['Close'].empty:
                logger.error(f"yfinance: Kritik 'Close' sütunu ({symbol}) eksik veya boş. Veri kullanılamaz. (Deneme {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                else:
                    return pd.DataFrame()

            for col in required_cols:
                if col not in data.columns or data[col].empty:
                    logger.warning(f"yfinance: Çekilen veride '{col}' sütunu eksik veya boş ({symbol}). Dolduruluyor.")
                    if col in ['Open', 'High', 'Low']:
                        data[col] = data['Close']
                    elif col == 'Volume':
                        data[col] = 0
            
            data = data[required_cols]

            if data.empty:
                logger.warning(f"yfinance: {symbol} için boş veri çekildi (Deneme {attempt + 1}/{retries}).")
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                else:
                    return pd.DataFrame()
            
            logger.info(f"yfinance'dan {len(data)} adet {symbol} verisi başarıyla çekildi ve işlendi (Deneme {attempt + 1}/{retries}).")
            return data

        except Exception as e:
            logger.error(f"yfinance'dan veri çekilirken beklenmeyen hata ({symbol}, Deneme {attempt + 1}/{retries}): {e}", exc_info=True)
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.error(f"yfinance: {symbol} için tüm yeniden denemeler başarısız oldu.")
            return pd.DataFrame() # Denemeler sonrası boş DataFrame döndürür
    return pd.DataFrame() # Döngüden boş dönüyorsa


def _fetch_coinapi_data_api(crypto_symbol: str, api_key: str, currency: str = "USD", period_id: str = "1DAY", limit: int = 365, retries: int = 3, delay: int = 5) -> pd.DataFrame:
    """CoinAPI.io'dan kripto geçmiş veri çeker, yeniden deneme mekanizması ile."""
    logger.info(f"CoinAPI.io'dan {crypto_symbol}/{currency} verisi çekiliyor (limit={limit}, period={period_id}).")
    if not api_key:
        logger.error("CoinAPI anahtarı eksik, kripto veri çekilemiyor.")
        return pd.DataFrame()

    url = f"https://rest.coinapi.io/v1/ohlcv/{crypto_symbol}/{currency}/history?period_id={period_id}&limit={limit}"
    headers = {"X-CoinAPI-Key": api_key}
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status() 
            data = response.json()
            
            if not data:
                logger.warning(f"CoinAPI.io: {crypto_symbol} için boş veri çekildi (Deneme {attempt + 1}/{retries}).")
                if attempt < retries - 1:
                    time.sleep(delay)
                continue
            
            df = pd.DataFrame(data)
            df['time_period_end'] = pd.to_datetime(df['time_period_end'])
            df = df.set_index('time_period_end')
            df = df[['price_open', 'price_high', 'price_low', 'price_close', 'volume_traded']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df.index.name = 'Date'
            logger.info(f"CoinAPI.io'dan {len(df)} adet {crypto_symbol} verisi başarıyla çekildi ve işlendi (Deneme {attempt + 1}/{retries}).")
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinAPI.io'dan veri çekilirken hata ({crypto_symbol}, Deneme {attempt + 1}/{retries}): {e}", exc_info=True)
            if response.status_code == 429: # Rate limit
                logger.warning(f"CoinAPI.io: Oran limiti aşıldı ({crypto_symbol}). Deneme {attempt + 1}/{retries}")
                if attempt < retries - 1:
                    time.sleep(delay * 2) # Rate limit için daha uzun bekle
                else:
                    logger.error(f"CoinAPI.io: {crypto_symbol} için tüm yeniden denemeler başarısız oldu (oran limiti).")
                    return pd.DataFrame()
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.error(f"CoinAPI.io: {crypto_symbol} için tüm yeniden denemeler başarısız oldu.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"CoinAPI.io verisi işlenirken hata ({crypto_symbol}, Deneme {attempt + 1}/{retries}): {e}", exc_info=True)
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.error(f"CoinAPI.io: {crypto_symbol} için tüm yeniden denemeler başarısız oldu.")
                return pd.DataFrame()
    return pd.DataFrame()


def _fetch_fixer_data_api(base_currency: str, target_currency: str, api_key: str, date: str, retries: int = 3, delay: int = 5) -> dict:
    """Fixer.io'dan döviz kuru çeker, yeniden deneme mekanizması ile."""
    logger.info(f"Fixer.io'dan {base_currency}/{target_currency} verisi çekiliyor (tarih={date}).")
    if not api_key:
        logger.error("Fixer.io anahtarı eksik, döviz kuru çekilemiyor.")
        return {}
    
    url = f"http://data.fixer.io/api/{date}?access_key={api_key}&symbols={target_currency},{base_currency}&format=1"
    
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status() 
            data = response.json()
            
            if data.get("success") and "rates" in data:
                rates = data["rates"]
                if base_currency not in rates:
                    logger.warning(f"Fixer.io: Baz para birimi '{base_currency}' oranlarda bulunamadı (Deneme {attempt + 1}/{retries}).")
                    if attempt < retries - 1:
                        time.sleep(delay)
                        continue
                    else:
                        return {}
                if target_currency not in rates:
                    logger.warning(f"Fixer.io: Hedef para birimi '{target_currency}' oranlarda bulunamadı (Deneme {attempt + 1}/{retries}).")
                    if attempt < retries - 1:
                        time.sleep(delay)
                        continue
                    else:
                        return {}
                
                # Fixer.io'da tüm oranlar EUR'ya göre verilir. Bu yüzden çapraz kur hesaplaması gerekir.
                # Örnek: USD/TRY için: (EUR/TRY) / (EUR/USD)
                if base_currency == "EUR": # Eğer baz Euro ise direkt hedef kur
                    rate = rates[target_currency]
                else: # Çapraz kur hesaplaması
                    rate = rates[target_currency] / rates[base_currency]

                logger.info(f"Fixer.io'dan {base_currency}/{target_currency} için kur başarıyla çekildi: {rate} (Deneme {attempt + 1}/{retries}).")
                return {"price": rate}
            else:
                error_info = data.get('error', {}).get('info', 'Bilinmeyen hata')
                error_code = data.get('error', {}).get('code', 'N/A')
                logger.warning(f"Fixer.io'dan veri çekilirken başarısız yanıt veya başarı = false: Kod={error_code}, Hata='{error_info}' (Deneme {attempt + 1}/{retries}).")
                if error_code == 104: # Monthly usage limit reached
                    logger.error("Fixer.io: Aylık kullanım limiti aşıldı. Lütfen daha sonra tekrar deneyin veya planınızı yükseltin.")
                    return {}
                if error_code == 101: # Invalid API key
                    logger.error("Fixer.io: Geçersiz API anahtarı. Lütfen anahtarınızı kontrol edin.")
                    return {}
                if attempt < retries - 1:
                    time.sleep(delay)
                continue
        except requests.exceptions.RequestException as e:
            logger.error(f"Fixer.io'dan veri çekilirken hata ({base_currency}/{target_currency}, Deneme {attempt + 1}/{retries}): {e}", exc_info=True)
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.error(f"Fixer.io: {base_currency}/{target_currency} için tüm yeniden denemeler başarısız oldu.")
                return {}
        except Exception as e:
            logger.error(f"Fixer.io verisi işlenirken hata ({base_currency}/{target_currency}, Deneme {attempt + 1}/{retries}): {e}", exc_info=True)
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.error(f"Fixer.io: {base_currency}/{target_currency} için tüm yeniden denemeler başarısız oldu.")
                return {}
    return {}


# --- Ana Veri Çekme Fonksiyonları (DB cache kontrolü ve Django ORM ile) ---

def fetch_all_popular_assets_and_save():
    """
    Popüler varlıkların güncel fiyatlarını çeker ve veritabanına kaydeder/günceller.
    Sadece belirli bir süre içinde güncellenmediyse yeniden çeker.
    """
    POPULAR_CACHE_DURATION_MINUTES = 5 # Önbellek süresi
    
    logger.info("Popüler varlıkların güncel fiyatları çekiliyor (Veritabanı öncelikli).")
    overview_data = [] # `overview_data` tanımı fonksiyonun başında olmalı

    for asset_name, info in VARLIK_BILGILERI.items():
        symbol = info["sembol"]
        asset_type = info["tip"] # source yerine tip kullanıldı

        cached_record = PopularAssetCache.objects.filter(asset_name=asset_name).first()
        
        price = np.nan # Varsayılan olarak NaN
        change_percent = np.nan # Varsayılan olarak NaN

        # Eğer önbellek kaydı yoksa veya süresi dolmuşsa API'den çek
        if not cached_record or cached_record.last_updated < datetime.now() - timedelta(minutes=POPULAR_CACHE_DURATION_MINUTES):
            logger.info(f"Popüler varlık '{asset_name}': Veritabanında güncel veri bulunamadı veya eski, API'lerden çekiliyor.")
            fetched_via_api = False
            
            # Kriptolar için CoinAPI öncelikli
            if asset_type == "kripto":
                try:
                    # CoinAPI'den anlık fiyat almak için 2 günlük geçmiş çekip son kapanışı kullanıyoruz.
                    latest_crypto_data = _fetch_coinapi_data_api(symbol, settings.COINAPI_API_KEY, limit=2) 
                    if not latest_crypto_data.empty:
                        price = latest_crypto_data['Close'].iloc[-1]
                        if len(latest_crypto_data) > 1:
                            previous_close = latest_crypto_data['Close'].iloc[-2]
                            if previous_close != 0:
                                change_percent = ((price - previous_close) / previous_close) * 100
                            else:
                                change_percent = 0.0
                        else:
                            change_percent = np.nan # Yeterli geçmiş veri yoksa değişim NaN
                        
                        logger.info(f"Popüler varlık '{asset_name}': CoinAPI'den başarıyla çekildi.")
                        fetched_via_api = True
                    else:
                        logger.warning(f"Popüler varlık '{asset_name}': CoinAPI'den boş veri çekildi.")
                except Exception as e:
                    logger.error(f"Popüler varlık '{asset_name}': CoinAPI çekilirken hata: {e}", exc_info=True)
            
            # Dövizler için Fixer.io öncelikli (eğer anahtar varsa ve kripto çekilemediyse)
            if not fetched_via_api and asset_type == "doviz":
                try:
                    base_curr_fixer = None
                    target_curr_fixer = None
                    
                    if asset_name == "Euro/Dolar":
                        base_curr_fixer = "EUR"
                        target_curr_fixer = "USD"
                    elif asset_name == "Sterlin/Dolar":
                        base_curr_fixer = "GBP"
                        target_curr_fixer = "USD"
                    elif asset_name == "Dolar/Türk Lirası":
                        base_curr_fixer = "USD"
                        target_curr_fixer = "TRY"
                    
                    if base_curr_fixer and target_curr_fixer:
                        today_date = datetime.now().strftime('%Y-%m-%d')
                        yesterday_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

                        today_data = _fetch_fixer_data_api(base_curr_fixer, target_curr_fixer, settings.FIXER_API_KEY, today_date)
                        yesterday_data = _fetch_fixer_data_api(base_curr_fixer, target_curr_fixer, settings.FIXER_API_KEY, yesterday_date)

                        if today_data and 'price' in today_data:
                            price = today_data['price']
                            if yesterday_data and 'price' in yesterday_data and yesterday_data['price'] != 0:
                                change_percent = ((price - yesterday_data['price']) / yesterday_data['price']) * 100
                            else:
                                change_percent = 0.0
                            logger.info(f"Popüler varlık '{asset_name}': Fixer.io'dan başarıyla çekildi.")
                            fetched_via_api = True
                        else:
                            logger.warning(f"Popüler varlık '{asset_name}': Fixer.io'dan boş veri çekildi veya hata oluştu.")
                    else:
                        logger.warning(f"Popüler döviz varlığı '{asset_name}' için Fixer.io sembol ayrıştırması başarısız veya tanımlı değil.")

                except Exception as e:
                    logger.error(f"Popüler varlık '{asset_name}': Fixer.io çekilirken hata: {e}", exc_info=True)
            
            # Diğer tüm durumlar (emtia, yfinance kaynaklı olanlar veya diğer API'lerin başarısız olması) için yfinance
            if not fetched_via_api:
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30) 
                    yfinance_data = _fetch_yfinance_data_api(symbol, start_date, end_date)
                    if not yfinance_data.empty:
                        if len(yfinance_data) >= 2:
                            price = yfinance_data['Close'].iloc[-1]
                            previous_close = yfinance_data['Close'].iloc[-2]
                            if previous_close != 0:
                                change_percent = ((price - previous_close) / previous_close) * 100
                            else:
                                change_percent = 0.0
                        elif len(yfinance_data) == 1:
                            price = yfinance_data['Close'].iloc[-1]
                            change_percent = np.nan
                        else:
                            price = np.nan
                            change_percent = np.nan

                        logger.info(f"Popüler varlık '{asset_name}': yfinance'dan başarıyla çekildi.")
                        fetched_via_api = True
                    else:
                        logger.warning(f"Popüler varlık '{asset_name}': yfinance'dan boş veri çekildi.")
                except Exception as e:
                    logger.error(f"Popüler varlık '{asset_name}': yfinance çekilirken hata: {e}", exc_info=True)
            
            # Tüm API denemeleri başarısız olursa veya anahtar eksikse, veritabanındaki en eski veriye düş
            if not fetched_via_api:
                logger.warning(f"Popüler varlık '{asset_name}': API'lerden veri çekilemedi. Veritabanındaki en eski veriye düşülüyor.")
                
                cached_record_fallback = PopularAssetCache.objects.filter(asset_name=asset_name).first() 
                if cached_record_fallback:
                    price = cached_record_fallback.price
                    change_percent = cached_record_fallback.change_percent
                    logger.info(f"Popüler varlık '{asset_name}': API hatası oluştu, eski veritabanı verisi kullanılıyor.")
                else:
                    logger.warning(f"Popüler varlık '{asset_name}' için API hatası oluştu ve veritabanında eski veri de bulunamadı.")


            # Eğer başarılı bir şekilde fiyat çekildiyse veritabanına kaydet/güncelle
            if price is not None and not np.isnan(price):
                PopularAssetCache.objects.update_or_create(
                    asset_name=asset_name,
                    defaults={'price': price, 'change_percent': change_percent if change_percent is not None and not np.isnan(change_percent) else np.nan, 'last_updated': datetime.now()}
                )
                logger.info(f"{asset_name} veritabanına kaydedildi/güncellendi.")
            else:
                logger.info(f"{asset_name} için fiyat çekilemedi veya geçersiz fiyat, veritabanına kaydedilmedi.")
        else: # Önbellek güncel ise
            price = cached_record.price
            change_percent = cached_record.change_percent
            logger.info(f"Popüler varlık '{asset_name}': Veritabanından güncel veri yüklendi.")

        # Bu kısım for döngüsünün içinde kalmalı, her varlık için append ediliyor.
        overview_data.append({
            "Varlık": asset_name,
            "Fiyat": price,
            "Değişim (%)": change_percent
        })

    # Bu kısım for döngüsünün dışına, ama fetch_all_popular_assets_and_save() fonksiyonunun içine girintilenmelidir.
    df_overview = pd.DataFrame(overview_data)
    logger.info(f"Popüler varlıklar özeti DataFrame'e dönüştürüldü. Boyut: {df_overview.shape}")
    return df_overview


def get_historical_data_from_db_or_fetch(asset_symbol: str, start_date: datetime, end_date: datetime, force_fetch: bool = False) -> pd.DataFrame:
    """
    Belirtilen varlık için geçmiş verileri uygun API'lerden veya DB'den çeker.
    Öncelik sırası: DB (taze) -> (Kripto için CoinAPI, Döviz için Fixer, Diğerleri için yfinance) -> DB (eski)
    force_fetch: True ise her zaman yeniden çeker.
    """
    logger.info(f"get_historical_data_from_db_or_fetch çağrıldı: Sembol={asset_symbol}, Başlangıç={start_date.strftime('%Y-%m-%d')}, Bitiş={end_date.strftime('%Y-%m-%d')}")

    HISTORY_CACHE_DURATION_HOURS = 24 * 7 

    asset_info = next((info for name, info in VARLIK_BILGILERI.items() if info["sembol"] == asset_symbol), None)
    if not asset_info:
        logger.error(f"Hata: '{asset_symbol}' için varlık bilgisi bulunamadı.")
        return pd.DataFrame()
    
    asset_type = asset_info["tip"]
    
    try:
        historical_record = HistoricalData.objects.get(asset_symbol=asset_symbol)
        
        if historical_record.last_updated < datetime.now() - timedelta(hours=HISTORY_CACHE_DURATION_HOURS) or force_fetch: 
            logger.info(f"'{asset_symbol}' için eski veri bulundu veya yeniden çekme isteniyor, API'den yenileniyor...")
            
            fetched_data = pd.DataFrame()
            if asset_type == "kripto":
                limit_days = (end_date - start_date).days + 1
                fetched_data = _fetch_coinapi_data_api(asset_symbol, settings.COINAPI_API_KEY, limit=limit_days)
            else:
                fetched_data = _fetch_yfinance_data_api(asset_symbol, start_date, end_date)

            if not fetched_data.empty:
                historical_record.data_json = fetched_data.to_json(orient='split', date_format='iso')
                historical_record.last_updated = datetime.now()
                historical_record.save()
                logger.info(f"'{asset_symbol}' için geçmiş veri güncellendi ve kaydedildi.")
                fetched_data.index = pd.to_datetime(fetched_data.index)
                return fetched_data[(fetched_data.index.date >= start_date.date()) & (fetched_data.index.date <= end_date.date())].copy()
            else:
                logger.warning(f"Uyarı: '{asset_symbol}' için yeni veri çekilemedi, eski veri kullanılıyor.")
                df_from_db_old = pd.read_json(historical_record.data_json, orient='split')
                df_from_db_old.index = pd.to_datetime(df_from_db_old.index)
                return df_from_db_old[(df_from_db_old.index.date >= start_date.date()) & (df_from_db_old.index.date <= end_date.date())].copy()
        else:
            logger.info(f"'{asset_symbol}' için güncel veri veritabanından alınıyor.")
            df_from_db_fresh = pd.read_json(historical_record.data_json, orient='split')
            df_from_db_fresh.index = pd.to_datetime(df_from_db_fresh.index)
            return df_from_db_fresh[(df_from_db_fresh.index.date >= start_date.date()) & (df_from_db_fresh.index.date <= end_date.date())].copy()

    except HistoricalData.DoesNotExist:
        logger.info(f"'{asset_symbol}' için veri veritabanında bulunamadı, API'den çekiliyor...")
        
        fetched_data = pd.DataFrame()
        if asset_type == "kripto":
            limit_days = (end_date - start_date).days + 1
            fetched_data = _fetch_coinapi_data_api(asset_symbol, settings.COINAPI_API_KEY, limit=limit_days)
        else:
            fetched_data = _fetch_yfinance_data_api(asset_symbol, start_date, end_date)

        if not fetched_data.empty:
            HistoricalData.objects.create(
                asset_symbol=asset_symbol,
                data_json=fetched_data.to_json(orient='split', date_format='iso'),
                last_updated=datetime.now()
            )
            logger.info(f"'{asset_symbol}' için geçmiş veri çekildi ve veritabanına kaydedildi.")
            fetched_data.index = pd.to_datetime(fetched_data.index)
            return fetched_data[(fetched_data.index.date >= start_date.date()) & (fetched_data.index.date <= end_date.date())].copy()
        else:
            logger.error(f"Hata: '{asset_symbol}' için geçmiş veri çekilemedi. Boş DataFrame döndürüldü.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Hata: Geçmiş veri işlenirken bir sorun oluştu: {e}", exc_info=True)
        return pd.DataFrame()
