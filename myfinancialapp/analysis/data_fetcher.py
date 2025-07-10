import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import asyncio
import aiohttp
import logging
from asgiref.sync import sync_to_async
import json
from django.conf import settings
from django.utils import timezone 
from .models import PopularAssetCache, HistoricalData
from io import StringIO # StringIO eklendi

logger = logging.getLogger(__name__)

COINAPI_API_KEY = settings.COINAPI_API_KEY
FIXER_API_KEY = settings.FIXER_API_KEY
NEWS_API_KEY = settings.NEWS_API_KEY

VARLIK_BILGILERI = {
    "Altın": {"sembol": "GC=F", "kaynak": "yfinance", "tip": "emtia"},
    "Gümüş": {"sembol": "SI=F", "kaynak": "yfinance", "tip": "emtia"},
    "Ham Petrol": {"sembol": "CL=F", "kaynak": "yfinance", "tip": "emtia"},
    "Bitcoin": {"sembol": "BTC-USD", "kaynak": "coinapi", "tip": "kripto"},
    "Ethereum": {"sembol": "ETH-USD", "kaynak": "coinapi", "tip": "kripto"},
    "Solana": {"sembol": "SOL-USD", "kaynak": "coinapi", "tip": "kripto"},
    "Cardano": {"sembol": "ADA-USD", "kaynak": "coinapi", "tip": "kripto"},
    "Dogecoin": {"sembol": "DOGE-USD", "kaynak": "coinapi", "tip": "kripto"},
    "Binance Coin": {"sembol": "BNB-USD", "kaynak": "coinapi", "tip": "kripto"},
    "Ripple": {"sembol": "XRP-USD", "kaynak": "coinapi", "tip": "kripto"},
    "Euro/Dolar": {"sembol": "EURUSD=X", "kaynak": "yfinance", "tip": "doviz"},
    "Sterlin/Dolar": {"sembol": "GBPUSD=X", "kaynak": "yfinance", "tip": "doviz"},
    "Dolar/Türk Lirası": {"sembol": "TRY=X", "kaynak": "yfinance", "tip": "doviz"},
}

@sync_to_async
def _save_popular_asset_to_db(asset_name, price, change_percent):
    try:
        obj, created = PopularAssetCache.objects.update_or_create(
            asset_name=asset_name,
            defaults={
                'price': price,
                'change_percent': change_percent,
                'last_updated': timezone.now()
            }
        )
        if created:
            logger.info(f"Popüler varlık '{asset_name}': Veritabanına yeni kaydedildi.")
        else:
            logger.info(f"Popüler varlık '{asset_name}': Veritabanında güncellendi.")
    except Exception as e:
        logger.error(f"Popüler varlık '{asset_name}': Veritabanına kaydederken hata: {e}", exc_info=True)

@sync_to_async
def _get_popular_asset_from_db(asset_name):
    try:
        asset = PopularAssetCache.objects.get(asset_name=asset_name)
        return {
            "Varlık": asset.asset_name,
            "Fiyat": asset.price,
            "Değişim_yuzdesi": asset.change_percent,
            "last_updated": asset.last_updated
        }
    except PopularAssetCache.DoesNotExist:
        logger.info(f"Popüler varlık '{asset_name}': Veritabanında bulunamadı.")
        return None
    except Exception as e:
        logger.error(f"Popüler varlık '{asset_name}': Veritabanından çekerken hata: {e}", exc_info=True)
        return None

@sync_to_async
def _save_historical_data_to_db(symbol, df):
    if df.empty:
        logger.warning(f"'{symbol}' için kaydedilecek geçmiş veri boş.")
        return

    try:
        df_to_save = df.reset_index()
        df_to_save['Date'] = df_to_save['Date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        data_json_str = df_to_save.to_json(orient="records", date_format="iso")

        obj, created = HistoricalData.objects.update_or_create(
            asset_symbol=symbol,
            defaults={
                'data_json': json.loads(data_json_str),
                'last_updated': timezone.now()
            }
        )
        if created:
            logger.info(f"'{symbol}' için geçmiş veri veritabanına yeni kaydedildi.")
        else:
            logger.info(f"'{symbol}' için geçmiş veri veritabanında güncellendi.")
    except Exception as e:
        logger.error(f"'{symbol}' için geçmiş veri veritabanına kaydederken hata: {e}", exc_info=True)

@sync_to_async
def _get_historical_data_from_db(symbol):
    try:
        data_entry = HistoricalData.objects.get(asset_symbol=symbol)
        # Düzeltme: StringIO ile sarıldı
        df = pd.read_json(StringIO(json.dumps(data_entry.data_json)), orient="records")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.sort_index()
        logger.info(f"'{symbol}' için {len(df)} adet geçmiş veri veritabanından yüklendi.")
        return df
    except HistoricalData.DoesNotExist:
        logger.info(f"'{symbol}' için veritabanında geçmiş veri bulunamadı.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"'{symbol}' için geçmiş veri veritabanından çekerken hata: {e}", exc_info=True)
        return pd.DataFrame()

async def _fetch_yfinance_data_real(symbol, period="5y"): 
    logger.info(f"Yahoo Finance'tan '{symbol}' için veri çekiliyor...")
    try:
        ticker = yf.Ticker(symbol)
        df = await sync_to_async(ticker.history)(period=period)
        if df.empty:
            logger.warning(f"Yahoo Finance'tan '{symbol}' için boş veri döndü.")
            return pd.DataFrame()
        
        df.columns = [col.capitalize() for col in df.columns]
        df.index.name = 'Date'
        logger.info(f"Yahoo Finance'tan '{symbol}' için {len(df)} adet veri çekildi. İlk 5 satır:\n{df.head()}")
        return df
    except Exception as e:
        logger.error(f"Yahoo Finance'tan '{symbol}' için veri çekerken hata: {e}", exc_info=True)
        return pd.DataFrame()

async def _fetch_coinapi_data_real(symbol, start_date, end_date):
    logger.info(f"CoinAPI'den '{symbol}' için veri çekiliyor...")
    if not COINAPI_API_KEY:
        logger.error("CoinAPI API Anahtarı ayarlanmamış.")
        return pd.DataFrame()

    coinapi_symbol = symbol.replace('-', '/')
    url = f"https://rest.coinapi.io/v1/ohlcv/{coinapi_symbol}/history"
    headers = {'X-CoinAPI-Key': COINAPI_API_KEY}
    params = {
        'period_id': '1DAY',
        'time_start': start_date.isoformat("T") + "Z",
        'time_end': end_date.isoformat("T") + "Z",
        'limit': 365
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if not data:
                    logger.warning(f"CoinAPI'den '{symbol}' için boş veri döndü.")
                    return pd.DataFrame()

                df = pd.DataFrame(data)
                df['time_period_start'] = pd.to_datetime(df['time_period_start'])
                df = df.rename(columns={
                    'time_period_start': 'Date',
                    'price_open': 'Open',
                    'price_high': 'High',
                    'price_low': 'Low',
                    'price_close': 'Close',
                    'volume_traded': 'Volume'
                })
                df = df.set_index('Date')
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df = df.sort_index()
                logger.info(f"CoinAPI'den '{symbol}' için {len(df)} adet veri çekildi. İlk 5 satır:\n{df.head()}")
                return df
    except aiohttp.ClientResponseError as e:
        logger.error(f"CoinAPI HTTP Hatası ({symbol}): {e.status} - {e.message}. URL: {e.request_info.url}", exc_info=True)
        try:
            error_response_text = await response.text()
            logger.error(f"CoinAPI Detaylı Yanıt: {error_response_text}")
        except Exception as text_e:
            logger.error(f"CoinAPI hata yanıtı okunurken hata oluştu: {text_e}")
        raise 
    except Exception as e:
        logger.error(f"CoinAPI'den '{symbol}' için veri çekerken hata: {e}", exc_info=True)
        raise 

async def fetch_all_popular_assets_and_save():
    logger.info("Popüler varlıkların güncel fiyatları çekiliyor (Veritabanı öncelikli, gerçek API denemesi).")
    popular_assets_data = []
    
    for asset_name, info in VARLIK_BILGILERI.items():
        symbol = info["sembol"]
        source = info["kaynak"]
        
        db_data = await _get_popular_asset_from_db(asset_name)
        
        if db_data and db_data["last_updated"] and (timezone.now() - db_data["last_updated"]).total_seconds() < 3600:
            popular_assets_data.append({
                "Varlık": db_data["Varlık"],
                "Fiyat": db_data["Fiyat"],
                "Değişim_yuzdesi": db_data["Değişim_yuzdesi"]
            })
            logger.info(f"Popüler varlık '{asset_name}': Veritabanından güncel veri yüklendi.")
            continue

        price = None
        change_percent = None
        
        try:
            if source == "yfinance":
                ticker = yf.Ticker(symbol)
                info_data = await sync_to_async(lambda: ticker.info)()
                if info_data:
                    price = info_data.get('currentPrice') or info_data.get('regularMarketPrice')
                    previous_close = info_data.get('previousClose') or info_data.get('regularMarketPreviousClose')
                    if price and previous_close:
                        change_percent = ((price - previous_close) / previous_close) * 100
                    logger.info(f"Yahoo Finance'tan '{asset_name}' ({symbol}): Fiyat={price}, Değişim={change_percent}")
                else:
                    logger.warning(f"Yahoo Finance'tan '{asset_name}' ({symbol}) için güncel fiyat bilgisi alınamadı.")

            elif source == "coinapi":
                if not COINAPI_API_KEY:
                    logger.error("CoinAPI API Anahtarı ayarlanmamış. Kripto varlıklar çekilemiyor.")
                    continue
                
                coinapi_symbol = symbol.replace('-', '/')
                url = f"https://rest.coinapi.io/v1/exchangerate/{coinapi_symbol}"
                headers = {'X-CoinAPI-Key': COINAPI_API_KEY}
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, headers=headers) as response:
                            response.raise_for_status()
                            data = await response.json()
                            if data and 'rate' in data:
                                price = data['rate']
                                change_percent = 0.0
                                logger.info(f"CoinAPI'den '{asset_name}' ({symbol}): Fiyat={price}, Değişim=N/A (Varsayılan 0)")
                            else:
                                logger.warning(f"CoinAPI'den '{asset_name}' ({symbol}) için güncel fiyat bilgisi alınamadı. Yanıt: {data}")
                except Exception as e:
                    logger.error(f"CoinAPI'den '{asset_name}' ({symbol}) için güncel fiyat çekerken hata: {e}. yfinance'a geri dönülüyor.", exc_info=True)
                    ticker = yf.Ticker(symbol)
                    info_data = await sync_to_async(lambda: ticker.info)()
                    if info_data:
                        price = info_data.get('currentPrice') or info_data.get('regularMarketPrice')
                        previous_close = info_data.get('previousClose') or info_data.get('regularMarketPreviousClose')
                        if price and previous_close:
                            change_percent = ((price - previous_close) / previous_close) * 100
                        logger.info(f"Yahoo Finance'tan (geri dönüş) '{asset_name}' ({symbol}): Fiyat={price}, Değişim={change_percent}")
                    else:
                        logger.warning(f"Yahoo Finance'tan (geri dönüş) '{asset_name}' ({symbol}) için güncel fiyat bilgisi alınamadı.")


            if price is not None:
                popular_assets_data.append({
                    "Varlık": asset_name,
                    "Fiyat": float(price),
                    "Değişim_yuzdesi": float(change_percent) if change_percent is not None else None
                })
                await _save_popular_asset_to_db(asset_name, float(price), float(change_percent) if change_percent is not None else None)
            else:
                logger.warning(f"'{asset_name}' ({symbol}) için güncel fiyat alınamadı, listeye eklenmedi.")

        except Exception as e:
            logger.error(f"'{asset_name}' ({symbol}) için genel hata oluştu: {e}", exc_info=True)
            if db_data:
                popular_assets_data.append({
                    "Varlık": db_data["Varlık"],
                    "Fiyat": db_data["Fiyat"],
                    "Değişim_yuzdesi": db_data["Değişim_yuzdesi"]
                })
                logger.warning(f"'{asset_name}' ({symbol}) için API hatası oluştu, veritabanındaki eski veri kullanılıyor.")
            else:
                logger.warning(f"'{asset_name}' ({symbol}) için API hatası oluştu ve veritabanında da veri yok. Atlanıyor.")

    if popular_assets_data:
        df = pd.DataFrame(popular_assets_data)
        df = df.rename(columns={"Değişim_yuzdesi": "Değişim (%)"})
        logger.info(f"Popüler varlıklar özeti DataFrame'e dönüştürüldü. Boyut: {df.shape}")
        logger.info(f"popular_assets_df ilk 5 satır:\n{df.head()}")
        return df
    logger.warning("Hiç popüler varlık verisi toplanamadı.")
    return pd.DataFrame(columns=["Varlık", "Fiyat", "Değişim (%)"])


async def get_historical_data_from_db_or_fetch(symbol, start_date, end_date):
    logger.info(f"get_historical_data_from_db_or_fetch çağrıldı: Sembol={symbol}, Başlangıç={start_date.strftime('%Y-%m-%d')}, Bitiş={end_date.strftime('%Y-%m-%d')}")

    df_historical = pd.DataFrame()
    
    logger.info(f"'{symbol}' için güncel veri veritabanından alınıyor.")
    df_historical = await _get_historical_data_from_db(symbol)

    if df_historical.empty or (not df_historical.empty and (timezone.now() - df_historical.index.max()).total_seconds() > 86400):
        logger.info(f"'{symbol}' için veritabanında yeterli veri yok veya eski. API'den çekiliyor.")
        
        source_info = next((info for name, info in VARLIK_BILGILERI.items() if info['sembol'] == symbol), None)
        
        if not source_info:
            logger.warning(f"'{symbol}' için tanımlı bir veri kaynağı bulunamadı veya sembol VARLIK_BILGILERI'nde yok.")
            return pd.DataFrame()

        primary_source = source_info.get("kaynak")
        
        if primary_source == "yfinance":
            df_historical = await _fetch_yfinance_data_real(symbol, period="5y") 
        elif primary_source == "coinapi":
            try:
                df_historical = await _fetch_coinapi_data_real(symbol, start_date, end_date)
            except Exception as e:
                logger.error(f"CoinAPI'den '{symbol}' için geçmiş veri çekerken hata: {e}. yfinance'a geri dönülüyor.", exc_info=True)
                df_historical = await _fetch_yfinance_data_real(symbol, period="5y") 
        
        if not df_historical.empty:
            await _save_historical_data_to_db(symbol, df_historical)
            logger.info(f"'{symbol}' için API'den çekilen {len(df_historical)} adet veri veritabanına kaydedildi.")
        else:
            logger.warning(f"API'den '{symbol}' için veri çekilemedi veya boş döndü. Veritabanındaki eski veri kullanılacak (varsa).")

    logger.info(f"get_historical_data_from_db_or_fetch() sonrası df_historical boş mu: {df_historical.empty}")
    if not df_historical.empty:
        logger.info(f"df_historical boyutu: {df_historical.shape}")
        logger.info(f"df_historical ilk 5 satır:\n{df_historical.head()}")
    return df_historical
