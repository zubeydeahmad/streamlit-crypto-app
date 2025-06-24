# data_fetcher.py
# (Veri Çekme ve Veritabanı)
# Bu dosya, çeşitli kaynaklardan (yfinance, CoinAPI.io) finansal verileri çeken fonksiyonları ve basit veritabanı başlatma mantığını içerir. Verileri önbelleğe almak için st.cache_data kullanılır.
import streamlit as st # Bu satır eklendi/doğrulandı
import yfinance as yf
import pandas as pd
import requests
import sqlite3
from datetime import datetime, timedelta
import logging
import os
import numpy as np # Bu satır eklendi/doğrulandı

# Loglama yapılandırması
logging.basicConfig(filename='crypto_app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logger = logging.getLogger(__name__)

# --- Varlık Seçenekleri ve Sembolleri ---
# Tüm uygulamada kullanılacak varlıkların tek, tutarlı listesi
VARLIK_BILGILERI = {
    "Altın": {"sembol": "GC=F", "kaynak": "yfinance"},
    "Gümüş": {"sembol": "SI=F", "kaynak": "yfinance"},
    "Ham Petrol": {"sembol": "CL=F", "kaynak": "yfinance"}, # WTI Crude Oil Futures
    "Bitcoin": {"sembol": "BTC-USD", "kaynak": "yfinance"},
    "Ethereum (ETH)": {"sembol": "ETH-USD", "kaynak": "yfinance"},
    "Solana (SOL)": {"sembol": "SOL-USD", "kaynak": "yfinance"},
    "Cardano (ADA)": {"sembol": "ADA-USD", "kaynak": "yfinance"},
    "Dogecoin (DOGE)": {"sembol": "DOGE-USD", "kaynak": "yfinance"},
    "Binance Coin (BNB)": {"sembol": "BNB-USD", "kaynak": "yfinance"},
    "Ripple (XRP)": {"sembol": "XRP-USD", "kaynak": "yfinance"},
    "Euro/Dolar (EURUSD)": {"sembol": "EURUSD=X", "kaynak": "yfinance"},
    "Sterlin/Dolar (GBPUSD)": {"sembol": "GBPUSD=X", "kaynak": "yfinance"},
    "Dolar/Türk Lirası (USDTRY)": {"sembol": "TRY=X", "kaynak": "yfinance"}
}

# --- CoinAPI Anahtarı ---
COINAPI_API_KEY = os.environ.get("COINAPI_API_KEY", "f970d607-417d-4767-a532-39c637b4edaa")


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
        data = yf.download(symbol, start=start_date, end=end_date, interval="1d", progress=False, actions=False, auto_adjust=True)

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

        # MultiIndex sütunları kontrol et ve tek seviyeye indir
        # yfinance bazen MultiIndex sütunları döndürebilir (örn: ('Close', 'SYMBOL')).
        # Bu kısım, sütunlara erişirken yaşanabilecek "truth value of a Series is ambiguous" hatalarını önler.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0].capitalize() for col in data.columns]
            logger.info(f"yfinance: MultiIndex sütunlar tek seviyeye indirildi.")
        else:
            data.columns = [col.capitalize() for col in data.columns] # Zaten tek seviyeli ise sadece capitalize et

        data.index = pd.to_datetime(data.index)

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            if 'Volume' in missing_cols:
                data['Volume'] = 0
                logger.warning(f"yfinance: '{symbol}' için Volume sütunu eksik, 0 ile dolduruldu.")
                missing_cols.remove('Volume')
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

# Popüler Varlıkların Güncel Fiyatları Çekme
@st.cache_data(ttl=timedelta(minutes=5), show_spinner="Popüler varlık fiyatları güncelleniyor...")
def get_popular_asset_overview_data():
    """
    Belirli popüler varlıkların güncel fiyatlarını ve 24 saatlik değişimlerini çeker.
    """
    logger.info("Popüler varlıkların güncel fiyatları çekiliyor.")
    overview_data = []

    # Güncel ve önceki gün kapanışlarını almak için son 7 günün verisi yeterli.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7) # Yeterli geçmiş veri sağlamak için 7 gün

    # Sadece VARLIK_BILGILERI içinde tanımlı ve yfinance kaynaklı varlıkları al
    popular_assets_to_show_filtered = {
        name: info["sembol"] for name, info in VARLIK_BILGILERI.items()
        if info["kaynak"] == "yfinance" # Sadece yfinance kaynaklı olanları göster
    }

    for asset_name, symbol in popular_assets_to_show_filtered.items():
        logger.debug(f"Popüler varlık için veri çekiliyor: {asset_name} ({symbol})")
        try:
            # yfinance'dan veri çek
            data = yf.download(symbol, start=start_date, end=end_date, interval="1d", progress=False, actions=False, auto_adjust=True)

            if not data.empty:
                # --- MultiIndex sütunları kontrol et ve tek seviyeye indir ---
                # yfinance bazen MultiIndex sütunları döndürebilir (örn: ('Close', 'SYMBOL')).
                # Bu kısım, sütunlara erişirken yaşanabilecek "truth value of a Series is ambiguous" hatalarını önler.
                if isinstance(data.columns, pd.MultiIndex):
                    # MultiIndex'in ilk seviyesini al ve her birinin ilk öğesini capitalize et (örneğin 'Close')
                    data.columns = [col[0].capitalize() for col in data.columns]
                    logger.debug(f"Popüler varlık ({symbol}): MultiIndex sütunlar tek seviyeye indirildi.")
                else:
                    # Zaten tek seviyeli ise sadece capitalize et
                    data.columns = [col.capitalize() for col in data.columns]
                # --- MultiIndex sütunları kontrol et ve tek seviyeye indir SONU ---

                if 'Close' in data.columns:
                    data_sorted = data.sort_index(ascending=True)

                    logger.debug(f"Popüler varlık ({symbol}): Çekilen veri boyutu: {data_sorted.shape}")
                    logger.debug(f"Popüler varlık ({symbol}): Son 2 satır:\n{data_sorted.tail(2)}")

                    # Calculate change based on available valid 'Close' data
                    valid_closes = data_sorted['Close'].dropna()

                    if len(valid_closes) >= 2:
                        latest_close = valid_closes.iloc[-1]
                        previous_close = valid_closes.iloc[-2]

                        # Ensure latest_close and previous_close are not NaN
                        if pd.isna(latest_close) or pd.isna(previous_close):
                            logger.warning(f"'{asset_name}' ({symbol}) için son veya önceki kapanış değeri NaN. Değişim hesaplanamadı.")
                            overview_data.append({
                                "Varlık": asset_name,
                                "Sembol": symbol,
                                "Fiyat": latest_close if pd.notna(latest_close) else np.nan,
                                "Değişim (%)": np.nan,
                                "Değişim Miktarı": np.nan
                            })
                        else:
                            price_change = latest_close - previous_close
                            percentage_change = (price_change / previous_close) * 100 if previous_close != 0 else 0

                            overview_data.append({
                                "Varlık": asset_name,
                                "Sembol": symbol,
                                "Fiyat": latest_close,
                                "Değişim (%)": percentage_change,
                                "Değişim Miktarı": price_change
                            })
                            logger.debug(f"Popüler varlık ({symbol}): Fiyat {latest_close:.2f}, Değişim {percentage_change:+.2f}%")
                    elif len(valid_closes) == 1:
                        latest_close = valid_closes.iloc[-1]
                        overview_data.append({
                            "Varlık": asset_name,
                            "Sembol": symbol,
                            "Fiyat": latest_close,
                            "Değişim (%)": np.nan,
                            "Değişim Miktarı": np.nan
                        })
                        logger.warning(f"'{asset_name}' ({symbol}) için yeterli geçmiş veri yok (sadece 1 gün). Değişim hesaplanamadı.")
                    else:
                        logger.warning(f"'{asset_name}' ({symbol}) için çekilen data boş veya 'Close' sütununda yeterli geçerli veri içermiyor. Popüler varlık listesine eklenemedi.")

                else:
                    logger.warning(f"'{asset_name}' ({symbol}) için veri çekilemedi veya 'Close' sütunu eksik.")
            else:
                logger.warning(f"'{asset_name}' ({symbol}) için veri çekilemedi (DataFrame boş).")
        except Exception as e:
            logger.error(f"'{asset_name}' ({symbol}) için güncel fiyat çekilirken hata: {e}. Detay: {e}")

    return pd.DataFrame(overview_data)
