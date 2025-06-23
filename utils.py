# utils.py
# (Yardımcı Fonksiyonlar)
# Bu dosya, teknik göstergeler ve zaman özellikleri ekleme gibi yardımcı fonksiyonları içerir.
# Ayrıca genel yüzde değişim hesaplama fonksiyonunu da içerir.

import pandas as pd
import numpy as np
import ta # Teknik analiz kütüphanesi
import logging

# Loglama yapılandırması
logging.basicConfig(filename='crypto_app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logger = logging.getLogger(__name__)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame'e çeşitli teknik analiz göstergeleri ekler.

    Args:
        df (pd.DataFrame): 'Open', 'High', 'Low', 'Close', 'Volume' sütunlarına sahip ham fiyat verisi.

    Returns:
        pd.DataFrame: Teknik göstergelerle zenginleştirilmiş DataFrame.
    """
    logger.info("Teknik göstergeler ekleniyor...")

    # Kapanış, Yüksek, Düşük, Hacim sütunlarının varlığını ve NaN durumunu kontrol et
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"'{col}' sütunu eksik. Teknik göstergeler eklenemiyor.")
            return df # Eksik sütun varsa orijinal DataFrame'i döndür

    # NaN değerleri kontrol edin ve dolgu yapın
    initial_nan_count = df.isnull().sum().sum()
    if initial_nan_count > 0:
        logger.warning(f"Teknik göstergeler öncesi {initial_nan_count} NaN değeri bulundu. Dolduruluyor.")
        df.fillna(method='ffill', inplace=True) # İleriye doğru doldur
        df.fillna(method='bfill', inplace=True) # Geriye doğru doldur (başlangıçtaki NaN'lar için)
        if df.isnull().sum().sum() > 0:
            logger.warning("Teknik göstergeler öncesi NaN değerler tam olarak doldurulamadı. Kalan NaN'lar sıfırla dolduruluyor.")
            df.fillna(0, inplace=True) # Kalan NaN'ları 0 ile doldur (eğer hala varsa)

    # RSI (Relative Strength Index)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff() # Histogram

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    df['BB_Band_Width'] = bollinger.bollinger_wband()

    # SMA (Simple Moving Average)
    df['SMA_7'] = ta.trend.SMAIndicator(df['Close'], window=7).sma_indicator()
    df['SMA_25'] = ta.trend.SMAIndicator(df['Close'], window=25).sma_indicator()
    df['SMA_99'] = ta.trend.SMAIndicator(df['Close'], window=99).sma_indicator() # Daha uzun vadeli ortalama

    # EMA (Exponential Moving Average)
    df['EMA_7'] = ta.trend.EMAIndicator(df['Close'], window=7).ema_indicator()
    df['EMA_25'] = ta.trend.EMAIndicator(df['Close'], window=25).ema_indicator()
    df['EMA_99'] = ta.trend.EMAIndicator(df['Close'], window=99).ema_indicator() # Daha uzun vadeli ortalama

    # Momentum
    df['Momentum'] = ta.momentum.ROCIndicator(df['Close'], window=14).roc()

    # ATR (Average True Range)
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    # CCI (Commodity Channel Index)
    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close'], window=20).cci()

    # ADX (Average Directional Index)
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
    df['ADX'] = adx.adx()
    df['ADX_pos'] = adx.adx_pos()
    df['ADX_neg'] = adx.adx_neg()

    # OBV (On-Balance Volume)
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

    # Ultimate Oscillator
    df['UO'] = ta.momentum.UltimateOscillator(df['High'], df['Low'], df['Close']).ultimate_oscillator()

    # Money Flow Index (MFI) - Daha sağlam bir kontrol
    try:
        df['MFI'] = ta.momentum.MoneyFlowIndex(df['High'], df['Low'], df['Close'], df['Volume']).money_flow_index()
    except AttributeError:
        logger.error("ta.momentum.MoneyFlowIndex bulunamadı veya uyumsuz. Lütfen ta kütüphanesini güncelleyin: pip install --upgrade ta")
        df['MFI'] = np.nan # Hata durumunda NaN ata
    except Exception as e:
        logger.warning(f"MFI hesaplanırken hata oluştu: {e}. Bu özellik atlanıyor ve MFI sütunu çıkarılacak.")
        df['MFI'] = np.nan # Hata durumunda NaN ata

    # VWAP (Volume Weighted Average Price)
    try:
        df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
    except Exception as e:
        logger.warning(f"VWAP hesaplanırken hata oluştu: {e}. Bu özellik atlanıyor.")
        df['VWAP'] = np.nan # Hata durumunda NaN ata

    # CMF (Chaikin Money Flow)
    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume']).chaikin_money_flow()

    # TRIX
    df['TRIX'] = ta.trend.TRIXIndicator(df['Close']).trix()

    # Price Lag Features (Geçmiş fiyatları özellik olarak ekleme)
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Volume_Lag1'] = df['Volume'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Volume_Lag2'] = df['Volume'].shift(2)
    df['Close_Lag3'] = df['Close'].shift(3)
    df['Volume_Lag3'] = df['Volume'].shift(3)
    df['Close_Lag5'] = df['Close'].shift(5)
    df['Volume_Lag5'] = df['Volume'].shift(5)
    df['Close_Lag7'] = df['Close'].shift(7)
    df['Volume_Lag7'] = df['Volume'].shift(7)
    df['Close_Lag10'] = df['Close'].shift(10)
    df['Volume_Lag10'] = df['Volume'].shift(10)
    df['Close_Lag15'] = df['Close'].shift(15)
    df['Volume_Lag15'] = df['Volume'].shift(15)
    df['Close_Lag20'] = df['Close'].shift(20)
    df['Volume_Lag20'] = df['Volume'].shift(20)

    # Daily Returns (Günlük Getiriler)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Volatility (Oynaklık - Geçmiş 20 günün log getirilerinin standart sapması)
    df['Volatility_20'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252) # Yıllıklandırılmış

    logger.info("Teknik göstergeler başarıyla eklendi.")

    # EK DÜZELTME: Tüm göstergeler eklendikten sonra tamamen NaN olan sütunları düşür
    cols_to_drop = []
    # Temel sütunların her zaman korunmasını sağla
    protected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in df.columns:
        if df[col].isnull().all() and col not in protected_cols:
            cols_to_drop.append(col)
            logger.warning(f"Teknik gösterge hesaplaması sonrası '{col}' sütunu tamamen NaN olduğu için düşürüldü.")

    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    logger.info(f"DataFrame boyutu (TI sonrası, NaN sütunları temizlenmiş): {df.shape}")
    logger.info(f"Null değerler (TI sonrası - before final dropna in ai_model): \n{df.isnull().sum()}") # Daha okunabilir çıktı

    return df

def add_market_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame'e zaman tabanlı özellikleri (piyasa açılış/kapanışları vb.) ekler.

    Args:
        df (pd.DataFrame): Datetime indeksli DataFrame.

    Returns:
        pd.DataFrame: Zaman özellikleri ile zenginleştirilmiş DataFrame.
    """
    logger.info("Piyasa zamanı özellikleri ekleniyor...")

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("DataFrame indeksi DatetimeIndex değil. Zaman özellikleri eklenemedi.")
        return df

    df['HourOfDay'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfMonth'] = df.index.day
    df['MonthOfYear'] = df.index.month
    df['QuarterOfYear'] = df.index.quarter

    df['IsWeekend'] = ((df.index.dayofweek == 5) | (df.index.dayofweek == 6)).astype(int)

    # Bu özellikler genellikle günlük veriler için sabit 1 olabilir veya gerçek açılış/kapanış saatlerine göre ayarlanır.
    # Günlük veriler için, günün tamamı piyasa açık kabul edilebilir.
    df['IsUSMarketOpenDaily'] = 1
    df['IsEUMarketOpenDaily'] = 1
    df['IsAsiaMarketOpenDaily'] = 1

    logger.info("Piyasa zamanı özellikleri başarıyla eklendi.")
    return df

def calculate_volatility(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    DataFrame'deki 'Close' fiyatı üzerinden günlük ve yıllıklandırılmış oynaklığı hesaplar.

    Args:
        df (pd.DataFrame): 'Close' sütununa sahip DataFrame.
        window (int): Getirileri hesaplamak için kullanılacak gün sayısı.

    Returns:
        pd.DataFrame: 'Daily_Volatility' ve 'Annualized_Volatility' sütunlarını içeren DataFrame.
    """
    if 'Close' not in df.columns:
        logger.error("calculate_volatility: 'Close' sütunu bulunamadı.")
        return pd.DataFrame()

    df['Log_Return_Vol'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Daily_Volatility'] = df['Log_Return_Vol'].rolling(window=window).std()
    df['Annualized_Volatility'] = df['Daily_Volatility'] * np.sqrt(252) # 252 işlem günü varsayımı

    logger.info(f"Oynaklık ({window} günlük) hesaplandı.")
    # Log_Return_Vol sütununu sadece hesaplama için kullandığımızdan düşürebiliriz
    return df.drop(columns=['Log_Return_Vol'], errors='ignore')

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    İki değer arasındaki yüzdesel değişimi hesaplar.

    Args:
        old_value (float): Eski değer (başlangıç değeri).
        new_value (float): Yeni değer (bitiş değeri).

    Returns:
        float: Yüzdesel değişim. Pozitif değer yükselişi, negatif değer düşüşü gösterir.
               Eğer eski değer 0 ise 0 döner.
    """
    logger.info(f"Yüzdesel değişim hesaplanıyor: Eski değer={old_value}, Yeni değer={new_value}")
    if old_value == 0:
        if new_value == 0:
            return 0.0
        else:
            logger.warning("calculate_percentage_change: Eski değer 0 iken yüzdesel değişim hesaplanamaz, 0 döndürüldü.")
            return 0.0

    percentage_change = ((new_value - old_value) / old_value) * 100
    logger.info(f"Hesaplanan yüzdesel değişim: {percentage_change:.2f}%")
    return percentage_change

if __name__ == "__main__":
    # Bu bölüm, modülün bağımsız olarak test edilmesi içindir ve Streamlit uygulamasına dahil edilmez.
    # Bu nedenle, bu bölümde Streamlit çağrıları kullanılmamalıdır.
    logger.info("Utils Modülü Testi (Bağımsız Çalışma)")

    # Örnek DataFrame oluşturma
    data = {
        'Open': [100, 102, 105, 103, 106, 108, 110, 112, 115, 113, 116, 118, 120, 122, 125,
                 123, 126, 128, 130, 132, 135, 133, 136, 138, 140, 142, 145, 143, 146, 148,
                 150, 152, 155, 153, 156, 158, 160, 162, 165, 163, 166, 168, 170, 172, 175,
                 173, 176, 178, 180, 182, 185],
        'High': [103, 106, 108, 106, 109, 111, 112, 114, 117, 115, 118, 120, 122, 124, 127,
                 125, 128, 130, 132, 134, 137, 135, 138, 140, 142, 144, 147, 145, 148, 150,
                 152, 154, 157, 155, 158, 160, 162, 164, 167, 165, 168, 170, 172, 174, 177,
                 175, 178, 180, 182, 184, 187],
        'Low': [98, 100, 102, 101, 104, 106, 108, 110, 113, 111, 114, 116, 118, 120, 123,
                121, 124, 126, 128, 130, 133, 131, 134, 136, 138, 140, 143, 141, 144, 146,
                148, 150, 153, 151, 154, 156, 158, 160, 163, 161, 164, 166, 168, 170, 173,
                171, 174, 176, 178, 180, 183],
        'Close': [101, 104, 103, 105, 107, 109, 111, 113, 116, 114, 117, 119, 121, 123, 126,
                  124, 127, 129, 131, 133, 136, 134, 137, 139, 141, 143, 146, 144, 147, 149,
                  151, 153, 156, 154, 157, 159, 161, 163, 166, 164, 167, 169, 171, 173, 176,
                  174, 177, 179, 181, 183, 186],
        'Volume': [1000, 1200, 1100, 1300, 1050, 1150, 1250, 1350, 1100, 1200, 1300, 1400, 1150, 1250, 1350,
                   1450, 1200, 1300, 1400, 1500, 1250, 1350, 1450, 1550, 1300, 1400, 1500, 1600, 1350, 1450,
                   1550, 1650, 1400, 1500, 1600, 1700, 1450, 1550, 1650, 1750, 1500, 1600, 1700, 1800, 1550,
                   1650, 1750, 1850, 1600, 1700, 1800]
    }
    df = pd.DataFrame(data, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(data['Close']))))


    # add_technical_indicators testi
    df_ti = add_technical_indicators(df.copy())
    logger.info("\n--- add_technical_indicators Test Sonucu (Head) ---")
    logger.info(df_ti.head())
    logger.info(f"TI sonrası nihai NaN kontrolü:\n{df_ti.isnull().sum()}")


    # add_market_time_features testi
    df_time = add_market_time_features(df.copy())
    logger.info("\n--- add_market_time_features Test Sonucu (Head) ---")
    logger.info(df_time.head())

    # calculate_volatility testi
    df_vol = calculate_volatility(df.copy())
    logger.info("\n--- calculate_volatility Test Sonucu (Head) ---")
    logger.info(df_vol.head())

    # calculate_percentage_change testi
    logger.info("\n--- calculate_percentage_change Test Sonuçları ---")
    change1 = calculate_percentage_change(100, 110)
    logger.info(f"100'den 110'a değişim: {change1:.2f}%") # Beklenen: 10.00%

    change2 = calculate_percentage_change(100, 90)
    logger.info(f"100'den 90'a değişim: {change2:.2f}%")  # Beklenen: -10.00%

    change3 = calculate_percentage_change(50, 50)
    logger.info(f"50'den 50'ye değişim: {change3:.2f}%")  # Beklenen: 0.00%

    change4 = calculate_percentage_change(0, 10)
    logger.info(f"0'dan 10'a değişim: {change4:.2f}%")   # Beklenen: 0.00 (logda uyarı)

    change5 = calculate_percentage_change(10, 0)
    logger.info(f"10'dan 0'a değişim: {change5:.2f}%")    # Beklenen: -100.00%
