# data_fetcher.py
# (Veri Çekme ve Veritabanı)
# Bu dosya, çeşitli kaynaklardan (yfinance, CoinAPI.io) finansal verileri çeken fonksiyonları ve basit veritabanı başlatma mantığını içerir. Verileri önbelleğe almak için st.cache_data kullanılır.
# import streamlit as st # Streamlit artık bu modülde doğrudan kullanılmıyor
import yfinance as yf
import pandas as pd
import requests
import sqlite3
from datetime import datetime, timedelta
import logging
import os 

# Loglama yapılandırması
logging.basicConfig(filename='crypto_app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logger = logging.getLogger(__name__)

# --- Varlık Seçenekleri ve Sembolleri ---
VARLIK_BILGILERI = {
    "Altın": {"sembol": "GC=F", "kaynak": "yfinance"},
    "Gümüş": {"sembol": "SI=F", "kaynak": "yfinance"}, # Gümüş için tekrar SI=F sembolü kullanıldı
    "Ham Petrol": {"sembol": "CL=F", "kaynak": "yfinance"}, # WTI Crude Oil Futures
    "Bitcoin": {"sembol": "BTC-USD", "kaynak": "yfinance"}, # Yfinance'dan BTC/USD spot fiyatı
    "Ethereum (ETH)": {"sembol": "ETH-USD", "kaynak": "yfinance"},
    "Solana (SOL)": {"sembol": "SOL-USD", "kaynak": "yfinance"},
    "Cardano (ADA)": {"sembol": "ADA-USD", "kaynak": "yfinance"},
    "Dogecoin (DOGE)": {"sembol": "DOGE-USD", "kaynak": "yfinance"},
    "Binance Coin (BNB)": {"sembol": "BNB-USD", "kaynak": "yfinance"},
    "Ripple (XRP)": {"sembol": "XRP-USD", "kaynak": "yfinance"}
}

# --- CoinAPI Anahtarı ---
# Kendi CoinAPI anahtarınızı buraya girin veya ortam değişkeni olarak ayarlayın.
# Güvenlik için ortam değişkeni kullanılması önerilir: COINAPI_API_KEY = os.environ.get("COINAPI_API_KEY", "YOUR_API_KEY_HERE")
COINAPI_API_KEY = os.environ.get("COINAPI_API_KEY", "f970d607-417d-4767-a532-39c637b4edaa") # API Anahtarınız buraya eklendi


# --- Veritabanı Ayarları ---
DATABASE_NAME = "crypto_data_v2.db"

def init_db():
    """Veritabanını başlatır."""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_data (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (symbol, date)
            )
        ''')
        conn.commit()
        conn.close()
        logger.info(f"Veritabanı '{DATABASE_NAME}' başlatıldı veya tablolar kontrol edildi.")
    except Exception as e:
        logger.error(f"Veritabanı başlatılırken hata oluştu: {e}")


def get_yfinance_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """yfinance'dan geçmiş fiyat verilerini çeker."""
    logger.info(f"yfinance'dan {symbol} verisi çekiliyor: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval="1d", progress=False, actions=False)
        
        logger.debug(f"\n--- DEBUG (yfinance) - {symbol} Ham Veri (Raw) ---")
        logger.debug(f"Veri boş mu? {data.empty}")
        if not data.empty:
            logger.debug(f"Veri boyutu: {data.shape}")
            logger.debug("İlk 5 satır (Raw):")
            logger.debug(data.head())
            logger.debug("Sütunlar (Raw):")
            logger.debug(data.columns)
            logger.debug("Null değerler (Raw):")
            logger.debug(data.isnull().sum())
        logger.debug("--- DEBUG (yfinance) Sonu ---\n")

        if data.empty:
            logger.warning(f"yfinance'dan '{symbol}' için veri çekilemedi veya veri bulunamadı. Sembolü veya tarih aralığını kontrol edin.")
            return pd.DataFrame()
        
        data.index = pd.to_datetime(data.index)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            logger.info("yfinance: MultiIndex sütunlar tek seviyeye indirildi.")
        
        if 'Adj Close' in data.columns and 'Close' not in data.columns:
            data['Close'] = data['Adj Close']
            logger.info("yfinance: 'Adj Close' sütunu 'Close' olarak yeniden adlandırıldı.")

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data.columns = [col.capitalize() for col in data.columns]

        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"yfinance'dan çekilen veride eksik sütunlar var: {missing_cols}. Lütfen sembolü kontrol edin veya yfinance API yanıtını inceleyin.")
            return pd.DataFrame() 
        
        final_data = data[required_cols].copy()
        
        logger.debug(f"\n--- DEBUG (yfinance) - {symbol} Initial Final Data (Before dropna) ---")
        logger.debug(f"Veri boş mu? {final_data.empty}")
        if not final_data.empty:
            logger.debug(f"Veri boyutu: {final_data.shape}")
            logger.debug("Null değerler (Before dropna):")
            logger.debug(final_data.isnull().sum())
        logger.debug("--- DEBUG (yfinance) Sonu ---\n")

        if final_data.isnull().values.any():
            nan_count_before_dropna = final_data.isnull().sum().sum()
            final_data.dropna(inplace=True)
            nan_count_after_dropna = final_data.isnull().sum().sum()
            rows_after_dropna = len(final_data)
            logger.warning(f"yfinance: Çekilen veride NaN değerler bulundu. {nan_count_before_dropna} NaN değeri vardı. {nan_count_before_dropna - nan_count_after_dropna} tanesi temizlendi. Kalan satır: {rows_after_dropna}")

            logger.debug(f"\n--- DEBUG (yfinance) - {symbol} Final Data (After dropna) ---")
            logger.debug(f"Veri boş mu? {final_data.empty}")
            if not final_data.empty:
                logger.debug(f"Veri boyutu: {final_data.shape}")
                logger.debug("Null değerler (After dropna):")
                logger.debug(final_data.isnull().sum())
            else:
                logger.debug("Veri boş kaldı.")
            logger.debug("--- DEBUG (yfinance) Sonu ---\n")

            if final_data.empty:
                logger.error("NaN değerler temizlendikten sonra yfinance verisi boş kaldı. Lütfen tarih aralığını veya sembolü kontrol edin.")
                return pd.DataFrame()

        logger.info(f"yfinance'dan {len(final_data)} adet {symbol} verisi başarıyla çekildi ve işlendi.")
        
        return final_data

    except Exception as e:
        logger.error(f"yfinance'dan veri çekilirken beklenmeyen hata oluştu ({symbol}): {e}. Detay: {e}")
        return pd.DataFrame()


def get_coinapi_data(asset_id_base: str, asset_id_quote: str = "USD", period_id: str = "1DAY", days_back: int = 365) -> pd.DataFrame:
    """CoinAPI.io'dan geçmiş borsa kuru verilerini çeker."""
    if not COINAPI_API_KEY:
        logger.error("CoinAPI anahtarı belirtilmedi veya geçersiz. CoinAPI'den veri çekilemiyor.")
        return pd.DataFrame()

    time_end = datetime.utcnow()
    time_start = time_end - timedelta(days=days_back)
    time_start_iso = time_start.isoformat("T") + "Z"
    time_end_iso = time_end.isoformat("T") + "Z"

    url = f"https://rest.coinapi.io/v1/exchangerate/{asset_id_base}/{asset_id_quote}/history"
    params = {
        "period_id": period_id,
        "time_start": time_start_iso,
        "time_end": time_end_iso,
        "limit": 10000
    }
    headers = {"X-CoinAPI-Key": COINAPI_API_KEY}

    logger.info(f"CoinAPI'den {asset_id_base}/{asset_id_quote} verisi çekiliyor. Başlangıç: {time_start_iso}, Bitiş: {time_end_iso}")
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data_json = response.json()
        
        logger.debug(f"\n--- DEBUG (CoinAPI) - {asset_id_base}/{asset_id_quote} Ham JSON Verisi ---")
        logger.debug(f"JSON boş mu? {not bool(data_json)}")
        if data_json:
            logger.debug(f"JSON ilk 2 öğesi: {data_json[:2]}")
            logger.debug(f"JSON uzunluğu: {len(data_json)}")
        logger.debug("--- DEBUG (CoinAPI) Sonu ---\n")

        if not data_json:
            logger.warning(f"CoinAPI'den {asset_id_base}/{asset_id_quote} için veri bulunamadı veya yetersiz.")
            return pd.DataFrame()

        df = pd.DataFrame(data_json)
        df['Date'] = pd.to_datetime(df['time_period_end'])
        df = df.set_index('Date')
        
        required_coinapi_cols = ['rate_open', 'rate_high', 'rate_low', 'rate_close']
        missing_coinapi_cols = [col for col in required_coinapi_cols if col not in df.columns]
        if missing_coinapi_cols:
            logger.error(f"CoinAPI'den çekilen veride eksik sütunlar var: {missing_coinapi_cols}. Lütfen API yanıtını kontrol edin.")
            return pd.DataFrame()
            
        df = df[required_coinapi_cols]
        df.columns = ['Open', 'High', 'Low', 'Close']
        
        if 'Volume' not in df.columns:
            df['Volume'] = 0 
        
        df = df.resample('D').last()
        
        logger.debug(f"\n--- DEBUG (CoinAPI) - {asset_id_base}/{asset_id_quote} Initial DataFrame (Before dropna) ---")
        logger.debug(f"DataFrame boş mu? {df.empty}")
        if not df.empty:
            logger.debug(f"DataFrame boyutu: {df.shape}")
            logger.debug("Null değerler (Before dropna):")
            logger.debug(df.isnull().sum())
        logger.debug("--- DEBUG (CoinAPI) Sonu ---\n")

        if df.isnull().values.any():
            nan_count_before_dropna = df.isnull().sum().sum()
            df.dropna(inplace=True)
            nan_count_after_dropna = df.isnull().sum().sum()
            rows_after_dropna = len(df)
            logger.warning(f"CoinAPI: Çekilen veride NaN değerler bulundu. {nan_count_before_dropna} NaN değeri vardı. {nan_count_before_dropna - nan_count_after_dropna} tanesi temizlendi. Kalan satır: {rows_after_dropna}")

            logger.debug(f"\n--- DEBUG (CoinAPI) - {asset_id_base}/{asset_id_quote} Final DataFrame (After dropna) ---")
            logger.debug(f"DataFrame boş mu? {df.empty}")
            if not df.empty:
                logger.debug(f"DataFrame boyutu: {df.shape}")
                logger.debug("Null değerler (After dropna):")
                logger.debug(df.isnull().sum())
            else:
                logger.debug("Veri boş kaldı.")
            logger.debug("--- DEBUG (CoinAPI) Sonu ---\n")

            if df.empty:
                logger.error("NaN değerler temizlendikten sonra CoinAPI verisi boş kaldı. Lütfen tarih aralığını veya sembolü kontrol edin.")
                return pd.DataFrame()

        logger.info(f"CoinAPI'den {len(df)} adet {asset_id_base} verisi çekildi ve işlendi.")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"CoinAPI HTTP hatası ({asset_id_base}): {http_err}. Durum kodu: {response.status_code}. Yanıt: {response.text}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as req_err:
        logger.error(f"CoinAPI istek hatası ({asset_id_base}): {req_err}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"CoinAPI veri işlenirken beklenmeyen hata ({asset_id_base}): {e}")
        return pd.DataFrame()

def fetch_exchange_rates(base_currency="USD"):
    """Döviz kurlarını çeker (Örn: Frankfurter API)."""
    api_url = f"https://api.frankfurter.app/latest?from={base_currency}"
    try:
        response = requests.get(api_url) 
        response.raise_for_status()
        data = response.json()
        
        if 'rates' in data:
            rates = data['rates']
            rates[base_currency] = 1.0 
            logger.info(f"Döviz kurları başarıyla çekildi (Baz: {base_currency}).")
            return rates
        else:
            logger.error("Döviz kurları API'sinden 'rates' verisi alınamadı.")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Döviz kurları çekilirken hata oluştu: {e}. Bu bir API limit hatası veya ağ bağlantı sorunu olabilir.")
        return None
    except Exception as e:
        logger.error(f"Döviz kurları çekilirken beklenmeyen bir hata oluştu: {e}")
        return None

if __name__ == "__main__":
    logger.info("Veri Çekme Modülü Testi (Bağımsız Çalışma)")
    init_db()

    # yfinance Veri Testi (Gümüş sembolü ile)
    yf_symbol = "SI=F" # Gümüş için SI=F sembolü ile test ediyoruz
    yf_start_date = datetime.now() - timedelta(days=5*365) # 5 yıl geçmiş
    yf_end_date = datetime.now()
    logger.info(f"yfinance Gümüş Veri Testi: {yf_symbol}")
    data_yf = get_yfinance_data(yf_symbol, yf_start_date, yf_end_date)
    if not data_yf.empty:
        logger.info(data_yf.head())
    else:
        logger.warning(f"Gümüş ({yf_symbol}) verisi çekilemedi. Lütfen internet bağlantınızı ve sembolün geçerliliğini kontrol edin.")

    # CoinAPI Veri Testi (eğer kullanılıyorsa ve anahtar varsa)
    coin_base = "BTC"
    coin_quote = "USD"
    coin_days = 90
    logger.info(f"CoinAPI Veri Testi: {coin_base}/{coin_quote}")
    data_coinapi = get_coinapi_data(coin_base.upper(), coin_quote.upper(), days_back=coin_days)
    if not data_coinapi.empty:
        logger.info(data_coinapi.head())
    else:
        logger.warning("CoinAPI verisi çekilemedi veya CoinAPI anahtarı eksik.")

    # Döviz Kurları Testi
    logger.info("Döviz Kurları Testi:")
    rates = fetch_exchange_rates("USD")
    if rates:
        logger.info(rates)
