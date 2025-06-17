import streamlit as st
import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ta # Teknik Analiz Kütüphanesi
import sqlite3
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import numpy as np
import logging
import joblib # Modeli ve scaler'ı kaydetmek/yüklemek için
from sklearn.model_selection import GridSearchCV
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup

# --- Streamlit sayfa yapılandırması ---
st.set_page_config(page_title="Sanal Yatırım Sepeti Simülasyonu", layout="wide")

# --- Loglama Yapılandırması ---
logging.basicConfig(filename='crypto_app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Varlık Seçenekleri ve Sembolleri ---
# Yeni varlıkları buraya ekleyin
VARLIK_BILGILERI = {
    "Altın": {"sembol": "GC=F", "kaynak": "yfinance"},
    "Gümüş": {"sembol": "SI=F", "kaynak": "yfinance"},
    "Ham Petrol": {"sembol": "CL=F", "kaynak": "yfinance"}, # WTI Crude Oil Futures
    "Bitcoin": {"sembol": "BTC-USD", "kaynak": "yfinance"}, # Yfinance'dan BTC/USD spot fiyatı
    # 'Bitcoin': {'sembol': 'bitcoin', 'kaynak': 'coingecko'}, # Eğer CoinGecko gibi bir yerden çekecekseniz
}

# --- Kullanıcıdan Giriş Alma ---
st.header("Yatırım Sepetinizi Oluşturun")

baslangic_bakiyesi = st.number_input("Başlangıç Bakiyeniz (USD):", min_value=100.0, value=1000.0, step=10.0)
st.write(f"Mevcut Bakiyeniz: ${baslangic_bakiyesi:,.2f}")

secilen_varliklar = st.multiselect(
    "Yatırım yapmak istediğiniz varlıkları seçin:",
    list(VARLIK_BILGILERI.keys())
)

yatirim_tutarlari = {}
kalan_bakiye = baslangic_bakiyesi
yatirim_gecerli = True

if secilen_varliklar:
    st.subheader("Yatırım Tutarlarını Belirleyin:")
    for varlik in secilen_varliklar:
        max_tutar = kalan_bakiye if kalan_bakiye > 0 else 0
        tutar = st.number_input(
            f"{varlik} için yatırım tutarı (USD):",
            min_value=0.0,
            max_value=max_tutar, # Maksimum kalan bakiyeyi geçmesin
            value=min(10.0, max_tutar), # Varsayılan küçük bir değer
            step=1.0,
            key=f"input_{varlik}" # Her input için benzersiz anahtar
        )
        yatirim_tutarlari[varlik] = tutar
        kalan_bakiye -= tutar
        if kalan_bakiye < 0:
            st.error("Yatırım tutarı bakiyenizi aşıyor. Lütfen düzeltin.")
            yatirim_gecerli = False
        st.write(f"Kalan Bakiye: ${kalan_bakiye:,.2f}")
    
    if kalan_bakiye < 0:
        yatirim_gecerli = False

else:
    st.info("Lütfen yatırım yapmak istediğiniz varlıkları seçin.")
    yatirim_gecerli = False # Varlık seçilmediyse yatırım yapılamaz

# NLTK'nın VADER sözlüğünü indirin (ilk çalıştırmada bir kere yapılır)
# Ayrıca, indirme işlemi sırasında oluşabilecek diğer hataları da yakalıyoruz.
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError: # NLTK kaynağı bulunamadığında fırlatılan standart hata
    st.info("VADER lexicon bulunamadı, indiriliyor...")
    try:
        nltk.download('vader_lexicon')
    except Exception as e:
        st.error(f"VADER lexicon indirilirken bir hata oluştu: {e}. Lütfen internet bağlantınızı kontrol edin veya Python sürümünüzün NLTK ile uyumlu olduğundan emin olun.")
except Exception as e: # Diğer beklenmeyen başlangıç hatalarını yakala
    st.error(f"VADER lexicon kontrol edilirken beklenmeyen bir hata oluştu: {e}")

# 'punkt' tokenizer'ı da benzer şekilde indirin (metin bölme için genellikle gereklidir)
try:
    nltk.data.find('tokenizers/punkt.zip')
except LookupError:
    st.info("Punkt tokenizer bulunamadı, indiriliyor...")
    try:
        nltk.download('punkt')
    except Exception as e:
        st.error(f"Punkt tokenizer indirilirken bir hata oluştu: {e}. Lütfen internet bağlantınızı kontrol edin veya Python sürümünüzün NLTK ile uyumlu olduğundan emin olun.")
except Exception as e:
    st.error(f"Punkt tokenizer kontrol edilirken beklenmeyen bir hata oluştu: {e}")


def get_news_headlines(url):
    """Belirli bir URL'den haber başlıklarını çekmeye çalışır."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # HTTP hataları için hata fırlat
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = []
        
        # Bu kısım her web sitesi için özelleştirilmelidir!
        # CoinDesk'in HTML yapısı sıkça değişebilir, bu seçiciler güncel olmayabilir.
        # Genellikle başlıklar h2, h3 etiketleri içinde veya belirli class'lara sahip div'ler içinde yer alır.
        
        # Olası CoinDesk başlık seçicileri (güncel olanı kontrol etmeniz gerekebilir):
        # Örnek 1: h2 etiketi ve belirli bir class
        for h2_tag in soup.find_all('h2', class_='css-1a6v75g'): 
            a_tag = h2_tag.find('a')
            if a_tag and a_tag.text:
                headlines.append(a_tag.text.strip())
        
        # Örnek 2: div etiketi ve başka bir olası başlık class'ı
        if not headlines: # Eğer ilk denemede başlık bulunamazsa
            for div_tag in soup.find_all('div', class_='text-xl'): # Başka bir olası başlık etiketi
                 a_tag = div_tag.find('a')
                 if a_tag and a_tag.text:
                     headlines.append(a_tag.text.strip())
        
        # Eğer hala başlık yoksa, genel bir başlık araması deneyebilirsiniz (daha az spesifik)
        if not headlines:
            for title_tag in soup.find_all(['h1', 'h2', 'h3', 'h4']):
                if title_tag.find('a') and title_tag.find('a').text:
                    headlines.append(title_tag.find('a').text.strip())
                elif title_tag.text and len(title_tag.text.strip()) > 10: # Başlığın çok kısa olmamasını sağla
                    headlines.append(title_tag.text.strip())


        return headlines
    except requests.exceptions.RequestException as e:
        st.error(f"Haber çekilirken ağ hatası oluştu: {e}") # Streamlit ile hata göster
        return []
    except Exception as e:
        st.error(f"Haber çekilirken beklenmeyen bir hata oluştu: {e}") # Streamlit ile hata göster
        return []

def analyze_sentiment(text_list):
    """Metin listesi için duygu analizi yapar ve ortalama bileşik skoru döndürür."""
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for text in text_list:
        vs = analyzer.polarity_scores(text)
        sentiments.append(vs['compound']) # Bileşik skor (-1.0 ile +1.0 arası)
    
    if sentiments:
        return sum(sentiments) / len(sentiments)
    return 0.0 # Haber yoksa nötr


# --- Kullanım Örneği ---
if __name__ == "__main__":
    coindesk_url = "https://www.coindesk.com/"
    
    st.subheader("Piyasa Haberleri ve Duyarlılık Analizi")
    st.info("Bu bölüm, örnek bir web sitesinden (CoinDesk) haber başlıklarını çekerek duygu analizi yapar.")

    with st.spinner("Haberler çekiliyor ve analiz ediliyor..."):
        news_headlines = get_news_headlines(coindesk_url)
        if news_headlines:
            st.write(f"Çekilen {len(news_headlines)} haber başlığı:")
            for i, headline in enumerate(news_headlines[:5]): # İlk 5 başlığı göster
                st.write(f"- {headline}")
            
            avg_sentiment = analyze_sentiment(news_headlines)
            
            st.write(f"**Ortalama Duygu Skoru (VADER):** {avg_sentiment:.2f} (1.0 = çok pozitif, -1.0 = çok negatif)")

            # Duygu skoruna göre yorum yap
            if avg_sentiment > 0.1:
                st.success("Genel haber duyarlılığı pozitif görünüyor.")
            elif avg_sentiment < -0.1:
                st.error("Genel haber duyarlılığı negatif görünüyor.")
            else:
                st.info("Genel haber duyarlılığı nötr.")
        else:
            st.warning("Haber başlıkları çekilemedi veya site yapısı değişmiş olabilir.")

    st.markdown("---")
    st.caption("Not: Web kazıma kodları, hedef sitenin HTML yapısı değiştiğinde çalışmayabilir. Profesyonel uygulamalar için genellikle haber API'leri tercih edilir.")

    
    # --- coinapi.io sitesinden aldığım api ile ---
coinapi_key = "f970d607-417d-4767-a532-39c637b4edaa"  #coinapi.io sitesinden api
def get_coinapi_data(asset_id_base="BTC", asset_id_quote="USD", period_id="1DAY", days_back=365):
    """
    CoinAPI.io'dan geçmiş borsa kuru verilerini çeker.
    
    Args:
        asset_id_base (str): Temel kripto para sembolü (örn. "BTC", "ETH").
        asset_id_quote (str): Karşılaştırma para birimi sembolü (örn. "USD", "TRY").
        period_id (str): Veri aralığı (örn. "1SEC", "1MIN", "1HRS", "1DAY").
        days_back (int): Kaç gün öncesine kadar veri çekileceği.
        
    Returns:
        pd.DataFrame: Tarih ve kapanış fiyatını içeren DataFrame.
    """
    
    # Bitiş ve başlangıç zamanlarını hesapla
    time_end = datetime.utcnow() # UTC zamanını kullan
    time_start = time_end - timedelta(days=days_back)

    # ISO 8601 formatına dönüştür
    time_start_iso = time_start.isoformat("T") + "Z"
    time_end_iso = time_end.isoformat("T") + "Z"

    # CoinAPI uç noktası
    url = f"https://rest.coinapi.io/v1/exchangerate/{asset_id_base}/{asset_id_quote}/history"
    
    # API'ye gönderilecek parametreler
    params = {
        "period_id": period_id,
        "time_start": time_start_iso,
        "time_end": time_end_iso,
        "limit": 10000 # Maksimum çekilebilecek veri adeti (API limitine göre ayarla)
    }
    
    # API anahtarını içeren başlıklar
    headers = {
        "X-CoinAPI-Key": coinapi_key
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status() # HTTP hataları (4xx, 5xx) için hata fırlat
        data = response.json()
        
        if not data:
            st.warning(f"CoinAPI'den {asset_id_base}/{asset_id_quote} için veri bulunamadı veya yetersiz.")
            return pd.DataFrame()

        # Veriyi DataFrame'e dönüştür
        # CoinAPI yanıt yapısı: [{'time_period_start', 'time_period_end', 'rate_open', 'rate_high', 'rate_low', 'rate_close'}]
        df = pd.DataFrame(data)
        
        # Sütunları yeniden adlandır ve gerekli olanları seç
        df['Date'] = pd.to_datetime(df['time_period_end']) # Dönemin kapanış zamanı
        df = df.set_index('Date')
        df = df[['rate_open', 'rate_high', 'rate_low', 'rate_close']]
        df.columns = ['Open', 'High', 'Low', 'Close'] # yfinance ile uyumlu olması için adları eşleştir
        
        # Günlük veri istediğimiz için aynı güne ait birden fazla girdi varsa sonuncuyu al (nadiren olabilir)
        df = df.resample('D').last()
        
        return df

    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP hatası oluştu: {http_err}. Durum kodu: {response.status_code}. Yanıt: {response.text}")
        if response.status_code == 401:
            st.error("API anahtarınız geçersiz veya yetkilendirilmemiş. Lütfen CoinAPI.io anahtarınızı kontrol edin.")
        elif response.status_code == 429:
            st.error("API limitinize ulaşıldı. Lütfen daha sonra tekrar deneyin veya planınızı yükseltin.")
        return pd.DataFrame()
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"Bağlantı hatası oluştu: {conn_err}. İnternet bağlantınızı kontrol edin.")
        return pd.DataFrame()
    except requests.exceptions.Timeout as timeout_err:
        st.error(f"İstek zaman aşımına uğradı: {timeout_err}.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as req_err:
        st.error(f"İstek sırasında bilinmeyen bir hata oluştu: {req_err}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Veri işlenirken beklenmeyen bir hata oluştu: {e}")
        return pd.DataFrame()

#--- api'dan veri çekme ---
if __name__ == "__main__":
    st.title("CoinAPI.io Entegrasyonu Örneği")

    asset_base = st.text_input("Temel Varlık (örn. BTC):", "BTC")
    asset_quote = st.text_input("Karşılaştırma Varlık (örn. USD):", "USD")
    gun_sayisi = st.slider("Geçmiş gün sayısı:", 7, 730, 90) # CoinAPI çoğu ücretsiz planda 1 yıl civarı geçmiş veri verir

    if st.button("CoinAPI'den Veriyi Çek"):
        with st.spinner(f"{asset_base}/{asset_quote} verileri CoinAPI'den çekiliyor..."):
            coinapi_data = get_coinapi_data(
                asset_id_base=asset_base.upper(), 
                asset_id_quote=asset_quote.upper(), 
                days_back=gun_sayisi
            )
            
            if not coinapi_data.empty:
                st.subheader(f"{asset_base}/{asset_quote} Geçmiş Fiyatları (CoinAPI.io'dan)")
                st.line_chart(coinapi_data['Close'])
                st.write(coinapi_data.tail()) # Son birkaç veriyi göster
            else:
                st.warning("Veri çekilemedi. Lütfen ayarları kontrol edin veya CoinAPI.io dokümantasyonunu inceleyin.")


# --- Piyasa Zamanı Özellikleri Fonksiyonu ---
def add_market_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Veri setinizin index'inin datetime olduğundan emin olun
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        # Eğer datetime değilse, dönüştürmeyi deneyin
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            st.warning(f"Zaman tabanlı özellikler için DataFrame indeksi datetime'a dönüştürülemedi: {e}")
            return df # Dönüştürülemezse mevcut DataFrame'i döndür

    df['HourOfDay'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek # Pazartesi=0, Pazar=6
    df['DayOfMonth'] = df.index.day # Ayın günü (1-31)
    df['MonthOfYear'] = df.index.month # Yılın ayı (1-12)
    df['QuarterOfYear'] = df.index.quarter # Yılın çeyreği (1-4)
    df['IsWeekend'] = ((df.index.dayofweek == 5) | (df.index.dayofweek == 6)).astype(int)

    # Küresel Piyasa Açık Saatleri (Türkiye saati (EEST) bazında, mevcut tarih ve saate göre)
    # yfinance günlük (daily) veri çektiği için, bu piyasaların gün içinde ne zaman açık olduğu bilgisi
    # günlük veriye doğrudan yansımaz. Çünkü her gün zaten ilgili piyasalar açık oluyor.
    # Bu özellikler, eğer saatlik veya daha kısa periyotlu veri çekecek olsaydınız daha anlamlı olurdu.
    # Ancak yine de, modelin gün bazında hangi piyasaların o gün aktif olduğunu "bilmesi" için
    # bu özellikler eklenmiş olur. (Günlük veri için bu özellikler genelde 1 olacaktır, hafta sonları 0)
    
    df['IsUSMarketOpenDaily'] = ((df.index.dayofweek < 5)).astype(int) # Hafta içi 1, hafta sonu 0
    df['IsEUMarketOpenDaily'] = ((df.index.dayofweek < 5)).astype(int) # Hafta içi 1, hafta sonu 0
    df['IsAsiaMarketOpenDaily'] = ((df.index.dayofweek < 5)).astype(int) # Hafta içi 1, hafta sonu 0

    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # ... (add_technical_indicators fonksiyonunuzun içeriği) ...
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff() # MACD Histogram
    bb = ta.volatility.BollingerBands(df['Close'])
    df['Bollinger_High'] = bb.bollinger_hband()
    df['Bollinger_Low'] = bb.bollinger_lband()
    df['BB_Band_Width'] = bb.bollinger_wband()
    df['SMA_7'] = ta.trend.SMAIndicator(df['Close'], window=7).sma_indicator()
    df['SMA_25'] = ta.trend.SMAIndicator(df['Close'], window=25).sma_indicator()
    df['SMA_99'] = ta.trend.SMAIndicator(df['Close'], window=99).sma_indicator()
    df['EMA_7'] = ta.trend.EMAIndicator(df['Close'], window=7).ema_indicator()
    df['EMA_25'] = ta.trend.EMAIndicator(df['Close'], window=25).ema_indicator()
    df['EMA_99'] = ta.trend.EMAIndicator(df['Close'], window=99).ema_indicator()
    df['Momentum'] = ta.momentum.roc(df['Close'], window=14)
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
    df['ADX'] = adx.adx()
    df['ADX_pos'] = adx.adx_pos()
    df['ADX_neg'] = adx.adx_neg()
    return df

st.sidebar.title("Kriterler")
st.sidebar.markdown("""
Bu araç, kripto para fiyatlarını tahmin etmek için bir makine öğrenimi modeli kullanır. Modelin performansını değerlendirmek için aşağıdaki kriterler kullanılır:

* **Yön Tahmin Doğruluğu:** Modelin fiyatın yükseliş veya düşüş yönünü doğru tahmin etme yüzdesi.
* **R-kare (R²):** Modelin veriye ne kadar iyi uyduğunu gösterir (1'e yakın değerler daha iyidir).
* **Ortalama Mutlak Hata (MAE):** Modelin tahminlerinin ortalama ne kadar yanlış olduğunu gösterir (daha düşük değerler daha iyidir).
* **Kök Ortalama Kare Hata (RMSE):** Tahmin hatalarının kareköküdür. Büyük hataları daha çok cezalandırır (daha düşük değerler daha iyidir).
* **Ortalama Mutlak Yüzde Hata (MAPE):** Tahmin hatasını yüzde olarak ifade eder (daha düşük değerler daha iyidir).
* **Karşılaştırmalı Getiri:** Modelin tahminleriyle işlem yapsaydık ne kadar kar/zarar ederdik (sanal işlemlerle hesaplanır).
""")

st.title("📈 Gelişmiş Kripto Para Tahmin Aracı v2.1")
st.markdown("Bu araç, seçilen kripto para birimleri için geçmiş verileri analiz eder, teknik göstergeler üretir ve XGBoost makine öğrenimi modeli kullanarak bir sonraki gün için fiyat tahmini yapar.")

# --- Döviz Kurları Çekme Fonksiyonu ---
@st.cache_data(ttl=3600) # Kurları 1 saat boyunca önbelleğe al
def fetch_exchange_rates(base_currency="USD"):
    try:
        # Örnek olarak FreeCurrencyAPI'yi kullanıyorum.
        # Gerçek bir projede kendi API anahtarınızı almanız gerekebilir.
        # Bu URL, deneme amaçlıdır ve sık kullanımdan sonra sınırlanabilir.
        # Daha güvenilir bir API için bir anahtar almanız veya alternatif bir ücretsiz API bulmanız önerilir.
        api_url = f"https://api.frankfurter.app/latest?from={base_currency}"   #  !!!! API al ve bunu düzelt
        
        response = requests.get(api_url)
        response.raise_for_status() # HTTP hataları için istisna fırlatır (4xx veya 5xx)
        data = response.json()
        
        if 'rates' in data:
            rates = data['rates']
            rates[base_currency] = 1.0 # Temel para birimi için kendi kuru 1'dir
            st.success(f"Döviz kurları başarıyla çekildi (Baz: {base_currency}).")
            return rates
        else:
            st.error("Döviz kurları API'sinden 'rates' verisi alınamadı.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Döviz kurları çekilirken hata oluştu: {e}")
        st.warning("Bu bir API limit hatası veya ağ bağlantı sorunu olabilir. Lütfen daha sonra tekrar deneyin veya farklı bir API kaynağı kullanmayı düşünün.")
        return None
    except Exception as e:
        st.error(f"Beklenmeyen bir hata oluştu: {e}")
        return None
    
# --- Piyasa Zamanı Özellikleri Fonksiyonu ---
def add_market_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Veri setinizin index'inin datetime olduğundan emin olun
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        # Eğer datetime değilse, dönüştürmeyi deneyin
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            st.warning(f"Zaman tabanlı özellikler için DataFrame indeksi datetime'a dönüştürülemedi: {e}")
            return df # Dönüştürülemezse mevcut DataFrame'i döndür

    df['HourOfDay'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek # Pazartesi=0, Pazar=6
    df['DayOfMonth'] = df.index.day # Ayın günü (1-31)
    df['MonthOfYear'] = df.index.month # Yılın ayı (1-12)
    df['QuarterOfYear'] = df.index.quarter # Yılın çeyreği (1-4)
    df['IsWeekend'] = ((df.index.dayofweek == 5) | (df.index.dayofweek == 6)).astype(int)

    df['IsUSMarketOpenDaily'] = ((df.index.dayofweek < 5)).astype(int) # Hafta içi 1, hafta sonu 0
    df['IsEUMarketOpenDaily'] = ((df.index.dayofweek < 5)).astype(int) # Hafta içi 1, hafta sonu 0
    df['IsAsiaMarketOpenDaily'] = ((df.index.dayofweek < 5)).astype(int) # Hafta içi 1, hafta sonu 0

    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # --- TA Kütüphanesi ile tüm göstergeleri ekleyin ---
    # Ta kütüphanesinin kendi fonksiyonları zaten NaN'ları doldurabilir (fillna=True)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14, fillna=True).rsi()
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff() # MACD Histogram
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2, fillna=True)
    df['Bollinger_High'] = bb.bollinger_hband() # Önceki kodunuzda 'BBH' idi, burada 'Bollinger_High' yapalım
    df['Bollinger_Low'] = bb.bollinger_lband() # Önceki kodunuzda 'BBL' idi, burada 'Bollinger_Low' yapalım
    df['BB_Band_Width'] = bb.bollinger_wband() # Önceki kodunuzda 'BB_Bandwidth' idi, bu isim daha tutarlı

    df['SMA_7'] = ta.trend.SMAIndicator(df['Close'], window=7, fillna=True).sma_indicator()
    df['SMA_25'] = ta.trend.SMAIndicator(df['Close'], window=25, fillna=True).sma_indicator()
    df['SMA_99'] = ta.trend.SMAIndicator(df['Close'], window=99, fillna=True).sma_indicator()
    
    df['EMA_7'] = ta.trend.EMAIndicator(df['Close'], window=7, fillna=True).ema_indicator()
    df['EMA_25'] = ta.trend.EMAIndicator(df['Close'], window=25, fillna=True).ema_indicator()
    df['EMA_99'] = ta.trend.EMAIndicator(df['Close'], window=99, fillna=True).ema_indicator()

    df['Momentum'] = ta.momentum.roc(df['Close'], window=14, fillna=True)
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14, fillna=True).average_true_range()
    
    stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3, fillna=True)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close'], window=14, fillna=True).cci()
    
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14, fillna=True)
    df['ADX'] = adx.adx()
    df['ADX_pos'] = adx.adx_pos() # DI_plus idi
    df['ADX_neg'] = adx.adx_neg() # DI_minus idi

    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'], fillna=True).on_balance_volume()
    df['UO'] = ta.momentum.ultimate_oscillator(high=df['High'], low=df['Low'], close=df['Close'], fillna=True)
    df['MFI'] = ta.volume.money_flow_index(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14, fillna=True)
    
    # Gecikmeli Özellikler
    for lag in [1, 2, 3, 5, 7, 10, 15, 20]:
        df[f'Close_Lag{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag{lag}'] = df['Volume'].shift(lag)

    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    
    df['VWAP'] = ta.volume.volume_weighted_average_price(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14, fillna=True)
    df['CMF'] = ta.volume.chaikin_money_flow(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=20, fillna=True)
    df['TRIX'] = ta.trend.trix(close=df['Close'], window=15, fillna=True)
    
    return df

# --- YENİ FONKSİYON: prepare_data_for_model ---
def prepare_data_for_model(data: pd.DataFrame, prediction_days: int) -> tuple:
    # Veri setinizin index'inin datetime olduğundan emin olun (fetch_and_store_data'da zaten yapılıyor ama kontrol amaçlı)
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index)
    
    # Teknik göstergeleri ekle
    # Burada sizin add_technical_indicators fonksiyonunuz çağrılacak
    data = add_technical_indicators(data)
    
    # Piyasa zamanı özelliklerini ekle
    data = add_market_time_features(data)

    # Hedef değişken (y) oluştur: "prediction_days" kadar ilerideki kapanış fiyatı
    data['Target'] = data['Close'].shift(-prediction_days)
    
    # Tüm NaN içeren satırları bu noktada TEMİZLE.
    initial_rows = len(data)
    data.dropna(inplace=True) 
    rows_dropped = initial_rows - len(data)
    if rows_dropped > 0:
        st.caption(f"Veri setinden teknik göstergeler, zaman özellikleri ve hedef değişken sonrası NaN içeren {rows_dropped} satır çıkarıldı.")

    # Model eğitimi için yeterli veri kaldı mı?
    if data.empty or len(data) < 50: # Minimum 50 satır veri olmalı
        st.warning(f"Teknik göstergeler, zaman özellikleri ve NaN'lar temizlendikten sonra model eğitimi için yeterli veri kalmadı ({len(data)} satır). Lütfen veri aralığını genişletin veya farklı bir coin'i deneyin. En az 50 satır veri gereklidir.")
        return pd.DataFrame(), pd.Series(), pd.Series(), pd.Series(), None, pd.DataFrame() # Boş dönüş

    # Özellikleri (X) ve hedef değişkeni (y) ayır
    # Önceki kodunuzdaki tüm özellikleri buraya dahil ettim ve yeni zaman özelliklerini ekledim
    X_cols = [
        'Close', 'Volume', 'Open', 'High', 'Low',
        # Teknik Göstergeler
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 
        'Bollinger_High', 'Bollinger_Low', 'BB_Band_Width', # Yeni isimler
        'SMA_7', 'SMA_25', 'SMA_99', 
        'EMA_7', 'EMA_25', 'EMA_99',
        'Momentum', 'ATR', 'Stoch_K', 'Stoch_D', 'CCI', 'ADX', 'ADX_pos', 'ADX_neg',
        'OBV', 'UO', 'MFI', 'VWAP', 'CMF', 'TRIX', 
        # Gecikmeli Özellikler
        'Close_Lag1', 'Volume_Lag1',
        'Close_Lag2', 'Volume_Lag2',
        'Close_Lag3', 'Volume_Lag3',
        'Close_Lag5', 'Volume_Lag5',
        'Close_Lag7', 'Volume_Lag7',
        'Close_Lag10', 'Volume_Lag10',
        'Close_Lag15', 'Volume_Lag15',
        'Close_Lag20', 'Volume_Lag20',
        # Ek Getiri/Volatilite Özellikleri
        'Daily_Return', 'Log_Return', 'Volatility_20',
        # Yeni Piyasa Zamanı Özellikleri
        'HourOfDay', 'DayOfWeek', 'DayOfMonth', 'MonthOfYear', 'QuarterOfYear',
        'IsWeekend', 'IsUSMarketOpenDaily', 'IsEUMarketOpenDaily', 'IsAsiaMarketOpenDaily'
    ]
    
    # Sadece data DataFrame'inde mevcut olan sütunları X_cols listesine al
    X_cols_filtered = [col for col in X_cols if col in data.columns]
    X = data[X_cols_filtered]
    y = data['Target']

    # Hedef değişken NaN olan satırları kaldır (prediction_days kadar sondaki satırlar)
    # Bu zaten yukarıdaki data.dropna() içinde halledilmiş olmalı, ancak sağlamak için tekrar kontrol.
    X.dropna(inplace=True)
    y.dropna(inplace=True)

    # X ve y'nin aynı sayıda satıra sahip olduğundan emin ol
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    if X.empty or y.empty:
        st.error("Model eğitimi için yeterli ve temizlenmiş veri bulunamadı. Lütfen tarih aralığını veya veri kaynağını kontrol edin.")
        return pd.DataFrame(), pd.Series(), pd.Series(), pd.Series(), None, pd.DataFrame() 

    # Veriyi eğitim ve test setlerine ayır
    test_size_fraction = 0.2 # %20 test verisi
    split_index = int(len(X) * (1 - test_size_fraction))

    X_train_raw = X.iloc[:split_index]
    X_test_raw = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Veriyi ölçekle
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # DataFrame olarak geri dönüştür (özellik isimlerini ve indeksleri korumak için)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_raw.columns, index=X_train_raw.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_raw.columns, index=X_test_raw.index)

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler, X_test_raw   

# --- Veritabanı Başlatma ---
DATABASE_NAME = "crypto_data_v2.db"

def init_db():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.close()

init_db()

# --- Coin Seçimi ---
coins = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Cardano (ADA)": "ADA-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Binance Coin (BNB)": "BNB-USD",
    "Ripple (XRP)": "XRP-USD"
}

selected_coin_name = st.selectbox("Bir Kripto Para Seçin:", list(coins.keys()))
symbol = coins[selected_coin_name]

st.subheader("⚙️ Ayarlar") # Ayarlar başlığını ekliyoruz

prediction_days = st.number_input(
    "Kaç gün sonraki fiyatı tahmin etmek istersiniz?",
    min_value=1,
    max_value=30, # Tahmin gün sayısını 30 ile sınırlandırabiliriz
    value=1, # Varsayılan olarak 1 gün
    step=1
)

# --- Kullanıcı Girdileri ---
st.sidebar.header("📊 Analiz Ayarları")

# Tarih aralığı seçimi
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 3) # Son 3 yıl varsayılan olarak seçili gelsin

# st.date_input ile tarih aralığını kullanıcıya sun
date_range = st.date_input("Analiz için başlangıç ve bitiş tarihini seçin:", value=(start_date.date(), end_date.date()))

# Kullanıcı sadece tek tarih seçerse (örneğin başlangıç tarihi), hata vermemek için kontrol
if len(date_range) == 2:
    start_date = datetime.combine(date_range[0], datetime.min.time())
    end_date = datetime.combine(date_range[1], datetime.max.time())
else:
    st.warning("Lütfen analiz için geçerli bir başlangıç ve bitiş tarihi aralığı seçin.")
    st.stop() # Geçersiz tarih aralığında uygulamanın ilerlemesini durdur

# --- Veri Çekme, Depolama ve Önbellekleme ---
@st.cache_data(ttl=3600) # 1 saat boyunca aynı coin için cache'le
def fetch_and_store_data(symbol_param: str, start_date_param: datetime, end_date_param: datetime) -> pd.DataFrame:
    logging.info(f"{symbol_param} için veri çekme ve depolama işlemi başlatıldı. Tarih aralığı: {start_date_param.strftime('%Y-%m-%d')} - {end_date_param.strftime('%Y-%m-%d')}")

    conn = sqlite3.connect(DATABASE_NAME)
    table_name = "".join(filter(str.isalnum, symbol_param))
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            Date TEXT PRIMARY KEY, Open REAL, High REAL, Low REAL, Close REAL, Volume INTEGER
        )
    """)
    conn.commit()

    cursor.execute(f"SELECT MAX(Date) FROM {table_name}")
    last_db_date_str = cursor.fetchone()[0]

    fetch_start = start_date_param
    fetch_end = end_date_param

    if last_db_date_str:
        last_db_date = datetime.strptime(last_db_date_str, '%Y-%m-%d')
        if last_db_date < end_date_param:
            fetch_start = max(last_db_date + timedelta(days=1), start_date_param)
            st.info(f"Veritabanında '{symbol_param}' için veri bulundu. Son Kayıt: {last_db_date_str}. Eksik veriler çekiliyor (başlangıç: {fetch_start.strftime('%Y-%m-%d')})...")
        else:
            st.info(f"Veritabanında '{symbol_param}' için güncel veri bulundu. API'den yeni veri çekilmeyecek.")
            fetch_start = end_date_param + timedelta(days=1)
    else:
        st.info(f"Veritabanında '{symbol_param}' için veri bulunamadı. Tam tarih aralığı çekiliyor (başlangıç: {fetch_start.strftime('%Y-%m-%d')})...")

    if fetch_start <= fetch_end:
        try:
            new_data = yf.download(symbol_param,
                                    start=fetch_start.strftime('%Y-%m-%d'),
                                    end=fetch_end.strftime('%Y-%m-%d'),
                                    progress=False)

            if not new_data.empty:
                new_data.reset_index(inplace=True) 
                
                if isinstance(new_data.columns, pd.MultiIndex):
                    new_data.columns = new_data.columns.get_level_values(0)
                    logging.info(f"{symbol_param} için MultiIndex sütunlar düzeltildi.")
                
                new_data.columns = [col.replace(' ', '_').replace('.', '').strip() for col in new_data.columns]
                if 'index' in new_data.columns:
                    new_data.rename(columns={'index': 'Date'}, inplace=True)
                if 'Adj Close' in new_data.columns:
                    new_data.drop('Adj_Close', axis=1, inplace=True, errors='ignore') # Hata oluşursa ignore et
                    logging.info(f"{symbol_param} için 'Adj Close' sütunu kaldırıldı.")
                
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_cols:
                    if col not in new_data.columns:
                        new_data[col] = np.nan
                        logging.warning(f"'{symbol_param}' için '{col}' sütunu bulunamadı ve NaN ile dolduruldu.")
                
                new_data = new_data[required_cols]

                if pd.api.types.is_datetime64_any_dtype(new_data['Date']):
                    new_data['Date'] = new_data['Date'].dt.strftime('%Y-%m-%d')

                try:
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        new_data[col] = new_data[col].apply(lambda x: None if pd.isna(x) else x)

                    new_data.to_sql(table_name, conn, if_exists='append', index=False, dtype={'Date': 'TEXT'}, chunksize=1000, method='multi')
                    inserted_count = len(new_data)
                    logging.info(f"{symbol_param} için {inserted_count} adet yeni veri başarıyla kaydedildi (to_sql kullanılarak).")
                    if inserted_count > 0:
                        st.success(f"{inserted_count} günlük yeni '{symbol_param}' verisi başarıyla veritabanına eklendi.")
                    else:
                        st.info(f"'{symbol_param}' için yeni veri bulunamadı veya veritabanı zaten güncel.")
                except sqlite3.IntegrityError:
                    st.info(f"'{symbol_param}' için bazı tarihler veritabanında zaten mevcut. Mevcut veriler atlandı.")
                    logging.info(f"'{symbol_param}' için IntegrityError: Mevcut veriler atlandı.")
                except Exception as e:
                    logging.error(f"'{symbol_param}' için veritabanına to_sql ile yazarken hata oluştu: {e}")
                    st.error(f"Hata: Veri veritabanına kaydedilemedi: {e}")
            else:
                logging.info(f"'{symbol_param}' için {fetch_start.strftime('%Y-%m-%d')} sonrası yeni veri bulunamadı.")
                st.info(f"'{symbol_param}' için {fetch_start.strftime('%Y-%m-%d')} sonrası yeni veri bulunamadı.")

        except Exception as e:
            logging.error(f"'{symbol_param}' için yfinance veri çekme hatası: {e}")
            st.error(f"Hata: '{symbol_param}' için veri çekilemedi veya işlenemedi. Lütfen logları kontrol edin.")
    else:
        st.info(f"API'den çekilecek yeni veri aralığı bulunamadı veya istenen aralıkta zaten güncel veri var.")

    try:
        df_from_db = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE Date BETWEEN '{start_date_param.strftime('%Y-%m-%d')}' AND '{end_date_param.strftime('%Y-%m-%d')}' ORDER BY Date ASC", conn)
        df_from_db['Date'] = pd.to_datetime(df_from_db['Date'])
        df_from_db.set_index('Date', inplace=True)
        df_from_db.ffill(inplace=True)
        df_from_db.bfill(inplace=True)
        essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_from_db.dropna(subset=essential_cols, inplace=True)

        return df_from_db
    except Exception as e:
        logging.error(f"'{symbol_param}' için veritabanından veri yüklenirken hata oluştu: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# Döviz Birimi Seçimi
st.sidebar.subheader("Para Birimi Ayarları")

# Döviz kurlarını çek
# Bu fonksiyonun (fetch_exchange_rates) döviz kurlarını döndürdüğünü varsayıyoruz.
current_exchange_rates = fetch_exchange_rates(base_currency="USD") # USD bazlı kurları çekiyoruz

if current_exchange_rates is None:
    st.error("Döviz kurları çekilemedi. Yatırım simülasyonu için kurlar manuel olarak ayarlandı veya simülasyon devre dışı bırakıldı.")
    # Fallback veya uygulamanın durması
    EXCHANGE_RATES_FALLBACK = {
        "USD": 1.0,
        "TRY": 32.5, # Fallback kuru
        "EUR": 0.92, # Fallback kuru
        "GBP": 0.79  # Fallback kuru
    }
    currency_options_keys = list(EXCHANGE_RATES_FALLBACK.keys())
else:
    currency_options_keys = list(current_exchange_rates.keys())

# Session state'te daha önce seçilmiş bir para birimi var mı kontrol et
if 'selected_currency' not in st.session_state:
    # Yoksa, varsayılan olarak "USD"yi ayarla (veya ilk seçeneği)
    st.session_state.selected_currency = "USD" if "USD" in currency_options_keys else currency_options_keys[0]

# Kullanıcının daha önceki seçimini varsayılan olarak ayarla
default_index = currency_options_keys.index(st.session_state.selected_currency) \
                if st.session_state.selected_currency in currency_options_keys else 0

selected_currency_symbol = st.sidebar.selectbox(
    "Para Birimi Seçin",
    options=currency_options_keys,
    index=default_index,
    key='currency_selector' # Bu selectbox için benzersiz bir anahtar atayın
)

# Kullanıcı yeni bir seçim yaptığında, session state'i güncelle
st.session_state.selected_currency = selected_currency_symbol

# --- BURAYA EKLENECEK KISIM: selected_currency_symbol'a göre currency_symbol'u belirleme ---
if selected_currency_symbol == "USD":
    currency_symbol = "$"
elif selected_currency_symbol == "TRY":
    currency_symbol = "₺"
elif selected_currency_symbol == "EUR":
    currency_symbol = "€"
elif selected_currency_symbol == "GBP":
    currency_symbol = "£"
else:
    # Eğer başka bir para birimi eklenirse veya bilinmeyen bir durum olursa
    currency_symbol = selected_currency_symbol # Doğrudan seçilen sembolü kullan

# Sembolü kaydet
currency_symbol = selected_currency_symbol

# --- Yatırım Miktarı Girişi ---
st.sidebar.subheader("Yatırım Simülasyonu")
investment_amount = st.sidebar.number_input(
    f"Yatırım Miktarı ({selected_currency_symbol}):", # Dinamik başlık
    min_value=1.0,
    value=100.0,
    step=10.0,
    format="%.2f"
)

# Investment currency artık kullanıcıdan seçilen selected_currency_symbol olacak.
# Bu satırı kaldırın: investment_currency = st.sidebar.selectbox(...)

currency_options = {
    "USD": "USD", # Varsayılan: Amerikan Doları
    "TRY": "Türk Lirası",
    "EUR": "Euro",
    "GBP": "İngiliz Sterlini"
    # Buraya daha fazla döviz birimi eklenebilir
}
selected_currency_symbol = st.sidebar.selectbox(
    "Para Birimi Seçin",
    options=list(currency_options.keys()),
    format_func=lambda x: currency_options[x], # Gösterilen ismi güzelleştirir
    index=0 # Varsayılan olarak USD seçili gelir
)

# Sembolü kaydet
currency_symbol = selected_currency_symbol

# --- Tahmin Et Butonu ---
if st.button("Modeli Çalıştır ve Tahmin Et"):
    with st.spinner(f"'{selected_coin_name}' için veriler çekiliyor, model kontrol ediliyor ve tahmin yapılıyor... Lütfen bekleyin."):
        
        # !!! BURASI DEĞİŞTİ !!!
        # Döviz kurlarını güncel olarak ÇEK (butona basıldığında güncel kurlar alınsın)
        current_exchange_rates = fetch_exchange_rates(base_currency="USD")
        if current_exchange_rates is None:
            st.error("Döviz kurları alınamadığı için yatırım simülasyonu yapılamıyor. Lütfen internet bağlantınızı kontrol edin veya daha sonra tekrar deneyin.")
            st.stop() # Döviz kuru olmadan simülasyon yapamayız
        # !!! DEĞİŞİKLİK BİTTİ !!!

        # Model ve scaler'ı kaydetmek/yüklemek için dosya yolları
        model_path = f"model_{selected_coin_name}.joblib"
        scaler_path = f"scaler_{selected_coin_name}.joblib"
        
        model_loaded = False
        scaler_loaded = False

        # Daha önce eğitilmiş model ve scaler var mı kontrol et
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            model_loaded = True
            scaler_loaded = True
            st.info(f"Daha önce eğitilmiş '{selected_coin_name}' modeli ve ölçekleyicisi yüklendi.")
        except FileNotFoundError:
            st.info(f"'{selected_coin_name}' için daha önce eğitilmiş model bulunamadı. Yeni model eğitiliyor...")
            model_loaded = False 

        # --- Veri Çekme Fonksiyonunu Çağır ---
        data = fetch_and_store_data(symbol, start_date_param=start_date, end_date_param=end_date)
        
        # Eğer veri boşsa, test verisi oluştur
        if data.empty:
            st.warning("📉 Veri çekilemedi veya boş geldi. Test verisi oluşturuluyor...")
            dates = pd.date_range(start=start_date.date(), periods=150)
            data = pd.DataFrame({
                'Close': np.random.rand(150) * 100,
                'Volume': np.random.randint(100000, 500000, size=150),
                'Open': np.random.rand(150) * 100,
                'High': np.random.rand(150) * 100 + 10,
                'Low': np.random.rand(150) * 100 - 10,
                'Feature1': np.random.rand(150),
                'Feature2': np.random.rand(150),
            }, index=dates)
            data.index.name = 'Date' 

        if data.empty or len(data) < 50:
            st.warning(f"'{selected_coin_name}' için yeterli temel veri bulunamadı ({len(data)} satır). Lütfen daha fazla geçmiş veri olan bir coin seçin veya veri kaynağınızı kontrol edin.")
            st.stop()

        # --- Özellik Mühendisliği (Teknik İndikatörler) ---
        st.subheader("⚙ Teknik Analiz Göstergeleri (Gelişmiş)")


        # --- ZAMAN TEMELLİ YENİ ÖZELLİKLERİ BURAYA EKLE (data'ya eklenecek) ---
        st.write("Zaman temelli özellikler ekleniyor...")
        data['day_of_week'] = data.index.dayofweek          # Haftanın günü (0=Pazartesi, 6=Pazar)
        data['day_of_month'] = data.index.day              # Ayın günü (1-31)
        data['month_of_year'] = data.index.month           # Yılın ayı (1-12)
        data['quarter_of_year'] = data.index.quarter       # Yılın çeyreği (1-4)
        data['is_weekend'] = data.index.dayofweek.isin([5, 6]).astype(int) # Hafta sonu ise 1, değilse 0
        # --- ZAMAN TEMELLİ YENİ ÖZELLİKLERİN EKLENDİĞİ YERİN SONU ---

        st.write("Temel sütunlar doğrulanıyor ve dönüştürülüyor...")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in data.columns: # 'data' DataFrame'ini kullanıyoruz
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col].fillna(method='ffill', inplace=True)
                data[col].fillna(method='bfill', inplace=True)
                if data[col].isnull().any():
                    data[col].fillna(data[col].mean(), inplace=True)
        st.write("Temel sütunlar başarıyla dönüştürüldü.")

        try:
            # Tüm teknik göstergeleri tek bir seferde ekle (ta kütüphanesi)
            # fillna=True, başlangıçtaki NaN'ları doldurur
            data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
            logging.info(f"{symbol} için tüm teknik göstergeler hesaplandı.")

            # Ek teknik göstergeler (mevcut kodundan taşındı ve aynı yere eklendi)
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20, fillna=True)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50, fillna=True)
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14, fillna=True)
            macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
            data['MACD'] = macd.macd()
            data['MACD_Signal'] = macd.macd_signal()
            bb = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2, fillna=True)
            data['BBL'] = bb.bollinger_lband()
            data['BBM'] = bb.bollinger_mavg()
            data['BBH'] = bb.bollinger_hband()
            stoch = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'], window=14, smooth_window=3, fillna=True)
            data['STOCH_K'] = stoch.stoch()
            data['STOCH_D'] = stoch.stoch_signal()

            atr_indicator = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14, fillna=True)
            data['ATR'] = atr_indicator.average_true_range()
            adx_indicator = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=14, fillna=True)
            data['ADX'] = adx_indicator.adx()
            data['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume'], fillna=True).on_balance_volume()

            for lag in [1, 2, 3, 5, 7]:
                data[f'Close_Lag{lag}'] = data['Close'].shift(lag)
                data[f'Volume_Lag{lag}'] = data['Volume'].shift(lag)

            data['CCI'] = ta.trend.cci(high=data['High'], low=data['Low'], close=data['Close'], window=14, fillna=True)
            logging.info(f"{symbol} için CCI hesaplandı.")

            data['DI_plus'] = ta.trend.adx_pos(high=data['High'], low=data['Low'], close=data['Close'], window=14, fillna=True)
            data['DI_minus'] = ta.trend.adx_neg(high=data['High'], low=data['Low'], close=data['Close'], window=14, fillna=True) # DI_minus için high ve low doğru kullanıldı
            logging.info(f"{symbol} için DI_plus ve DI_minus hesaplandı.")

            data['UO'] = ta.momentum.ultimate_oscillator(high=data['High'], low=data['Low'], close=data['Close'], fillna=True)
            logging.info(f"{symbol} için Ultimate Oscillator hesaplandı.")

            data['MFI'] = ta.volume.money_flow_index(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=14, fillna=True)
            logging.info(f"{symbol} için MFI hesaplandı.")

            data['Daily_Return'] = data['Close'].pct_change() # pct_change default olarak NaN üretir
            data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1)) # log_return da NaN üretir
            logging.info(f"{symbol} için Günlük ve Log Getiriler hesaplandı.")

            data['Volatility_20'] = data['Close'].rolling(window=20).std() # rolling.std de NaN üretir
            logging.info(f"{symbol} için Volatilite_20 hesaplandı.")

            for lag in [10, 15, 20]:
                data[f'Close_Lag{lag}'] = data['Close'].shift(lag)
                data[f'Volume_Lag{lag}'] = data['Volume'].shift(lag)
            logging.info(f"{symbol} için ek gecikmeli özellikler eklendi.")

            data['VWAP'] = ta.volume.volume_weighted_average_price(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=14, fillna=True)
            logging.info(f"{symbol} için VWAP hesaplandı.")

            data['CMF'] = ta.volume.chaikin_money_flow(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=20, fillna=True)
            logging.info(f"{symbol} için CMF hesaplandı.")

            data['TRIX'] = ta.trend.trix(close=data['Close'], window=15, fillna=True)
            logging.info(f"{symbol} için TRIX hesaplandı.")

            # Bollinger Bantları zaten yukarıda hesaplandı, BB_Bandwidth için kontrol yapalım.
            if all(col in data.columns for col in ['BBH', 'BBL', 'BBM']):
                data['BB_Bandwidth'] = (data['BBH'] - data['BBL']) / data['BBM'] * 100
                logging.info(f"{symbol} için BB_Bandwidth hesaplandı.")
            else:
                logging.warning(f"{symbol} için BB_Bandwidth hesaplanamadı, Bollinger Bantları eksik.")

            # HEDEF SÜTUNUNU BURADA OLUŞTURUYORUZ! (Tüm özellikler eklendikten sonra)
            # prediction_days kadar ötele, böylece son 'prediction_days' satır NaN olacak.
            data['Next_Day_Close'] = data['Close'].shift(-prediction_days) 
            st.success(f"✅ 'Next_Day_Close' hedef sütunu başarıyla oluşturuldu (ileriye doğru {prediction_days} gün).")

            # Tüm NaN içeren satırları bu noktada TEMİZLE.
            initial_rows = len(data)
            data.dropna(inplace=True) 
            rows_dropped = initial_rows - len(data)
            st.caption(f"Veri setinden teknik göstergeler ve hedef değişken sonrası NaN içeren {rows_dropped} satır çıkarıldı. Kalan veri seti uzunluğu: {len(data)}")

        except Exception as e:
            st.error(f"Teknik göstergeler hesaplanırken veya hedef değişken oluşturulurken bir hata oluştu: {e}. Veri setinizi kontrol edin.")
            logging.error(f"{symbol} için teknik gösterge veya hedef değişken hatası: {e}")
            st.stop()

        # ÖNEMLİ KONTROL: Dropna sonrası yeterli veri kaldı mı?
        if data.empty or len(data) < 50: # Minimum 50 satır veri olmalı
            st.warning(f"Teknik göstergeler uygulandıktan ve NaN'lar temizlendikten sonra model eğitimi için yeterli veri kalmadı ({len(data)} satır). Lütfen veri aralığını genişletin veya farklı bir coin'i deneyin. En az 50 satır veri gereklidir.")
            st.stop()
        
        # Bu noktada 'Next_Day_Close' sütunu ve diğer özellik sütunları NaN içermemelidir.

                # --- Görselleştirme (Plotly ile Geliştirilmiş) ---
        st.subheader(f"📊 {selected_coin_name} Fiyat Grafiği ve Göstergeler")

        # İki satırlık subplot oluştur (Üstte fiyat, altta hacim)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_heights=[0.7, 0.3]) # Fiyat için daha fazla yer

        # Kapanış Fiyatı ve Hareketli Ortalamalar
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Kapanış Fiyatı',
                                line=dict(color='blue', width=2)), row=1, col=1)
        if 'SMA_20' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20',
                                    line=dict(color='orange', width=1)), row=1, col=1)
        if 'SMA_50' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50',
                                    line=dict(color='purple', width=1)), row=1, col=1)

        # Bollinger Bantları (Varsa)
        if all(col in data.columns for col in ['BBL', 'BBM', 'BBH']):
            fig.add_trace(go.Scatter(x=data.index, y=data['BBL'], mode='lines', name='Bollinger Alt Band',
                                    line=dict(color='gray', dash='dash'), opacity=0.5), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['BBH'], mode='lines', name='Bollinger Üst Band',
                                    line=dict(color='gray', dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', opacity=0.5), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['BBM'], mode='lines', name='Bollinger Orta Band',
                                    line=dict(color='darkgray', width=1, dash='dot')), row=1, col=1)

        # Hacim Grafiği
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Hacim', marker_color='darkgreen', opacity=0.7), row=2, col=1)

        # Düzenlemeler
        fig.update_layout(title_text=f"{selected_coin_name} Fiyat ve Hacim Grafiği",
                        height=600,
                        xaxis_rangeslider_visible=False,
                        hovermode="x unified",
                        legend_title_text="Göstergeler",
                        template="plotly_white") # Modern ve temiz bir tema
        fig.update_yaxes(title_text="Fiyat", row=1, col=1)
        fig.update_yaxes(title_text="Hacim", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Ek Göstergeler İçin Ayrı Grafikler (Hala columns kullanabiliriz)
        st.markdown("---")
        st.markdown("### Ek Teknik Göstergeler")
        col_ind1, col_ind2, col_ind3 = st.columns(3)

        with col_ind1:
            st.markdown("**RSI (Relative Strength Index)**")
            if 'RSI' in data.columns:
                fig_rsi = go.Figure(data=go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='red')))
                fig_rsi.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Aşırı Alım")
                fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Aşırı Satım")
                fig_rsi.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0), template="plotly_white")
                st.plotly_chart(fig_rsi, use_container_width=True)

            st.markdown("**MACD & Signal Line**")
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD Line', line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='red')))
                fig_macd.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0), template="plotly_white")
                st.plotly_chart(fig_macd, use_container_width=True)
        
        with col_ind2:
            st.markdown("**Stochastic Oscillator (%K & %D)**")
            if 'STOCH_K' in data.columns and 'STOCH_D' in data.columns:
                fig_stoch = go.Figure()
                fig_stoch.add_trace(go.Scatter(x=data.index, y=data['STOCH_K'], mode='lines', name='%K', line=dict(color='blue')))
                fig_stoch.add_trace(go.Scatter(x=data.index, y=data['STOCH_D'], mode='lines', name='%D', line=dict(color='red')))
                fig_stoch.add_hline(y=80, line_dash="dot", line_color="red", annotation_text="Aşırı Alım")
                fig_stoch.add_hline(y=20, line_dash="dot", line_color="green", annotation_text="Aşırı Satım")
                fig_stoch.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0), template="plotly_white")
                st.plotly_chart(fig_stoch, use_container_width=True)

            st.markdown("**ADX (Average Directional Index)**")
            if 'ADX' in data.columns:
                fig_adx = go.Figure(data=go.Scatter(x=data.index, y=data['ADX'], mode='lines', name='ADX', line=dict(color='green')))
                fig_adx.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0), template="plotly_white")
                st.plotly_chart(fig_adx, use_container_width=True)

        with col_ind3:
            st.markdown("**OBV (On-Balance Volume)**")
            if 'OBV' in data.columns:
                fig_obv = go.Figure(data=go.Scatter(x=data.index, y=data['OBV'], mode='lines', name='OBV', line=dict(color='brown')))
                fig_obv.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0), template="plotly_white")
                st.plotly_chart(fig_obv, use_container_width=True)
            
            st.markdown("**Ultimate Oscillator (UO)**")
            if 'UO' in data.columns:
                fig_uo = go.Figure(data=go.Scatter(x=data.index, y=data['UO'], mode='lines', name='UO', line=dict(color='cyan')))
                fig_uo.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Aşırı Alım")
                fig_uo.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Aşırı Satım")
                fig_uo.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0), template="plotly_white")
                st.plotly_chart(fig_uo, use_container_width=True)


        # --- Algoritmik Tahmin Modeli (XGBoost Regressor) ---
        st.subheader("🔮 Algoritmik Tahmin Modeli (XGBoost ile)")

        st.write("Veri setindeki sütunlar:", data.columns.tolist())
        st.write("Veri seti uzunluğu (teknik göstergeler sonrası):", len(data))

        # Özellik sütunlarını (X_cols) belirleme
        excluded_cols_for_features = ['Open', 'High', 'Low', 'Close', 'Next_Day_Close'] 
        
        candidate_cols = [col for col in data.columns 
                          if data[col].dtype in [np.number, np.int64, np.float64] 
                          and col not in excluded_cols_for_features
                         ]
        
        X_cols = [col for col in candidate_cols if not data[col].isnull().any()]
        
        if not X_cols:
            st.error("Model eğitimi için hiçbir özellik sütunu seçilemedi. Lütfen veri setinizi ve özellik mühendisliği adımlarını kontrol edin.")
            st.stop()

        st.write("Model için seçilen özellik sütunları (X_cols):", X_cols)

        # Şimdi X (özellikler) ve y (hedef değişken) oluşturuluyor
        X = data[X_cols]
        
        if 'Next_Day_Close' not in data.columns or data['Next_Day_Close'].isnull().any():
            st.error("Hedef değişken 'Next_Day_Close' oluşturulamadı veya NaN değerler içeriyor. Model eğitilemez.")
            st.stop()

        y = data['Next_Day_Close']

        if X.empty or y.empty:
            st.error("❌ Özellikler (X) veya hedef değişken (y) tamamen boş kaldı. Model eğitilemez.")
            st.stop()

        st.subheader("🧪 Nihai Kontroller")

        st.write(f"✅ X satır sayısı: {len(X)}, sütun sayısı: {X.shape[1] if not X.empty else 0}")
        st.write(f"✅ y satır sayısı: {len(y)}")

        required_min_rows = max(100, round(1 / 0.1)) # train_test_split için minimum 100 satır veya test_size'a göre ayarlayın
        if len(X) < required_min_rows or len(y) < required_min_rows:
            st.error(f"❌ Model eğitimi için yeterli veri yok ({len(X)} satır). En az {required_min_rows} günlük veri gereklidir.")
            st.stop()

        # train_test_split her zaman burada yapılmalı
        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=42)
            st.success("✅ Veri başarıyla eğitim/test setine ayrıldı.")
        except ValueError as e:
            st.error(f"❌ train_test_split hatası: {e}. Bu hata, veri setinin çok küçük olmasından veya boş olmasından kaynaklanabilir.")
            st.stop()

        # Normalizasyon
        re_fit_scaler = False
        if not scaler_loaded:
            re_fit_scaler = True
            st.info("Daha önce eğitilmiş ölçekleyici bulunamadı. Yeni ölçekleyici eğitiliyor.")
        else:
            if not hasattr(scaler, 'feature_names_in_') or list(scaler.feature_names_in_) != X_train_raw.columns.tolist():
                st.warning("Eğitim verisi sütunları, yüklü ölçekleyicinin eğitildiği sütunlarla eşleşmiyor. Ölçekleyici yeniden eğitiliyor.")
                logging.warning(f"Scaler uyumsuzluğu algılandı: Yüklü özellikler: {getattr(scaler, 'feature_names_in_', 'Yok')}, Mevcut özellikler: {X_train_raw.columns.tolist()}")
                re_fit_scaler = True
            else:
                st.info("Yüklü ölçekleyici ile eğitim verisi dönüştürülüyor.")
        
        if re_fit_scaler:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            scaler.feature_names_in_ = X_train_raw.columns.tolist()
            joblib.dump(scaler, scaler_path)
            st.success("✅ Ölçekleyici başarıyla eğitildi ve kaydedildi.")
        else:
            X_train_scaled = scaler.transform(X_train_raw)
            
        if hasattr(scaler, 'feature_names_in_'):
            expected_features = scaler.feature_names_in_
            X_test_aligned = X_test_raw.reindex(columns=expected_features, fill_value=0)
            
            if X_test_aligned.isnull().values.any():
                st.warning("Test verilerinde reindex sonrası hala NaN değerler var. Bu, veri temizliğinde bir sorun olabilir. NaN değerler 0 ile dolduruluyor.")
                X_test_aligned.fillna(0, inplace=True) 
            
            X_test_scaled = scaler.transform(X_test_aligned)
            st.success("✅ Test verileri ölçeklendi ve sütunlar ayarlandı.")
        else:
            st.error("Ölçekleyicinin eğitim özellik isimleri bulunamadı. Test verisi ölçeklenemiyor. Lütfen yeniden eğitin.")
            st.stop()

        st.subheader("🛠️ Model Optimizasyonu")

        # Model yükleme ve eğitim mantığı
        if not model_loaded:
            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', 
                                         eval_metric='rmse', 
                                         random_state=42)

            param_grid = {
                'n_estimators': [250, 300, 350, 400],
                'learning_rate': [0.03, 0.04, 0.05, 0.06, 0.07],
                'max_depth': [2, 3, 4, 5],
                'subsample': [0.8, 0.85, 0.9, 0.95, 1.0],
                'colsample_bytree': [0.75, 0.8, 0.85, 0.9]
            }

            st.info("GridSearchCV ile en iyi hiperparametreler aranıyor... Bu biraz zaman alabilir.")
            grid_search = GridSearchCV(estimator=xgb_model, 
                                       param_grid=param_grid,
                                       cv=3,
                                       n_jobs=-1,
                                       verbose=1,
                                       scoring='neg_mean_absolute_error')

            with st.spinner("Model için en iyi hiperparametreler aranıyor... (Bu işlem biraz zaman alabilir)"):
                grid_search.fit(X_train_scaled, y_train)

            st.success("✅ En iyi parametreler bulundu!")

            best_params = grid_search.best_params_
            st.write(f"**En İyi Model Parametreleri:** {best_params}")

            model = grid_search.best_estimator_
            joblib.dump(model, model_path)
            st.success("✅ Model başarıyla eğitildi ve kaydedildi.")
        else: # Model yüklüyse
            st.info("Model daha önce eğitildiğinden, GridSearchCV çalıştırılmadı.")
            st.write("Model daha önce kaydedilmiş parametrelerle çalıştırılıyor.")
            # Yüklü model zaten `model` değişkenine atanmış olmalı.
            # `model = joblib.load(model_path)` gibi bir satırın `main` fonksiyonunuzun başında `model_loaded = True` bloğunda olması gerekir.

        # y_pred, modelin tanımlandığı ve eğitildiği bloğun hemen dışında,
        # ancak X_test_scaled'in tanımlandığı yerden sonra tanımlanmalıdır.
        # Bu kısım tüm senaryolarda çalışacağından, koşullu blokların dışında olabilir.
        if 'model' in locals() and 'X_test_scaled' in locals():
            y_pred = model.predict(X_test_scaled)
        else:
            st.error("Model veya ölçeklenmiş test verisi tanımlanamadı. Tahmin yapılamıyor.")
            st.stop()
        
        # --- Model Performans Metrikleri ---
        st.subheader("📊 Model Performans Metrikleri")

        # MAE
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f"**Ortalama Mutlak Hata (MAE):** {mae:.2f}")

        # RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.write(f"**Kök Ortalama Kare Hata (RMSE):** {rmse:.2f}")

        # R-kare
        r2 = r2_score(y_test, y_pred)
        st.write(f"**R-kare (R²):** {r2:.2f}")

        # MAPE (Ortalama Mutlak Yüzde Hata)
        # Sıfıra bölme hatasını önlemek için küçük bir epsilon ekle
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        st.write(f"**Ortalama Mutlak Yüzde Hata (MAPE):** {mape:.2f}%")

        # Yön Tahmin Doğruluğu
        y_test_direction = np.sign(y_test.diff().dropna())
        y_pred_direction = np.sign(pd.Series(y_pred, index=y_test.index).diff().dropna())
        
        # Her iki serinin de aynı indekse sahip olduğundan emin olun
        common_index = y_test_direction.index.intersection(y_pred_direction.index)
        
        direction_accuracy = np.mean(y_test_direction[common_index] == y_pred_direction[common_index]) * 100
        st.write(f"**Yön Tahmin Doğruluğu:** {direction_accuracy:.2f}%")

        # --- Sanal İşlem Simülasyonu ---
        st.subheader("💰 Sanal İşlem Simülasyonu (Test Verisi Üzerinde)")

        initial_capital = 1000 # Başlangıç sermayesi
        capital = initial_capital
        num_trades = 0
        total_profit_loss = 0

        # DataFrame'i yeniden indeksle ve y_pred'i ekle
        test_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)
        
        # Gerçek kapanış fiyatlarındaki değişimi hesapla
        # `data['Close']` DataFrame'inizin en güncel kapanış fiyatlarını içerdiğinden emin olun.
        # Genellikle y_test, modelin eğitildiği veri setinin 'Next_Day_Close' sütunundan gelir.
        # Karşılaştırma için 'Close' sütununu kullanmak daha mantıklıdır.
        # Y_test'in bir önceki günkü değerini bulmak için kaydırma işlemi yapıyoruz
        
        # Buradaki kar/zarar hesaplamasını daha doğru hale getirelim:
        # y_test aslında X_test_raw'daki Close fiyatının 'prediction_days' sonraki halidir.
        # Dolayısıyla, y_test ile y_pred'i karşılaştırmak yeterli.
        
        # Basit bir sanal işlem: Her gün kapanışta al, ertesi gün kapanışta sat (prediction_days kadar sonra)
        # Bu, modelin tahmin ettiği yöne göre işlem yapmaktır.
        
        # Yön tahminine göre kar/zarar simülasyonu
        trade_data = pd.DataFrame({
            'Actual_Close': y_test.values,
            'Predicted_Close': y_pred
        }, index=y_test.index)

        # Önceki günün kapanış fiyatı (tahmin yapılan günkü kapanış)
        # Bu, X_test_raw'ın 'Close' sütununun son değeri veya y_test'in bir önceki değeri olmalı
        # Ancak y_test zaten 'Next_Day_Close' olduğu için, güncel 'Close' değerini almalıyız
        
        # y_test'in indeksi ile X_test_raw'ın indeksi eşleşmeli.
        # y_test, X_test_raw'ın karşılık gelen Next_Day_Close değeridir.
        
        # İşlem simülasyonu için Close fiyatlarına ihtiyacımız var.
        # X_test_raw'daki 'Close' sütununu kullanabiliriz.
        # Bu, modelin tahmin yaptığı günkü kapanış fiyatıdır.
        
        # Test verisi üzerindeki gerçek kapanış fiyatlarını alalım.
        actual_closes_for_test = data.loc[X_test_raw.index, 'Close']

        trading_results = []

        for i in range(len(test_df) - 1): # Son günden bir gün öncesine kadar döngü
            current_close = actual_closes_for_test.iloc[i] # Tahmin yapılan günkü kapanış
            
            # Modelin bir sonraki gün (veya prediction_days sonraki gün) için tahmini
            predicted_next_close = test_df['Predicted'].iloc[i]
            
            # Gerçekte prediction_days sonraki kapanış
            actual_next_close = test_df['Actual'].iloc[i]

            # Modelin tahmin ettiği yön: Yükseliş mi, Düşüş mü?
            predicted_direction = np.sign(predicted_next_close - current_close)
            
            # Gerçekte gerçekleşen yön
            actual_direction = np.sign(actual_next_close - current_close)

            if predicted_direction == 1: # Model yükseliş bekliyor (Alım işlemi)
                if actual_direction == 1: # Gerçekten yükseldi
                    profit = (actual_next_close - current_close)
                    st.write(f"🚀 Gün {i+1}: Yükseliş tahmin edildi. Gerçekten yükseldi! Kar: {currency_symbol}{profit:.2f}")
                else: # Yanlış tahmin, aslında düştü
                    profit = (actual_next_close - current_close)
                    st.write(f"🔻 Gün {i+1}: Yükseliş tahmin edildi. Ama düştü! Zarar: {currency_symbol}{profit:.2f}")
            elif predicted_direction == -1: # Model düşüş bekliyor (Kısa pozisyon veya satım işlemi, kar beklentisi)
                if actual_direction == -1: # Gerçekten düştü (kısa pozisyondan kar)
                    profit = (current_close - actual_next_close) # Kısa pozisyon karı
                    st.write(f"✅ Gün {i+1}: Düşüş tahmin edildi. Gerçekten düştü! Kar: {currency_symbol}{profit:.2f}")
                else: # Yanlış tahmin, aslında yükseldi (kısa pozisyondan zarar)
                    profit = (current_close - actual_next_close) # Kısa pozisyon zararı
                    st.write(f"❌ Gün {i+1}: Düşüş tahmin edildi. Ama yükseldi! Zarar: {currency_symbol}{profit:.2f}")
            else: # Değişim yok (sıfır) tahmin edildi
                profit = 0
                st.write(f"➖ Gün {i+1}: Değişim yok tahmin edildi. Kar/Zarar: {currency_symbol}{profit:.2f}")
            
            total_profit_loss += profit
            num_trades += 1
            capital += profit # Sermayeyi güncelle

            trading_results.append({
                'Date': test_df.index[i],
                'Current_Close': current_close,
                'Predicted_Next_Close': predicted_next_close,
                'Actual_Next_Close': actual_next_close,
                'Predicted_Direction': 'Up' if predicted_direction == 1 else ('Down' if predicted_direction == -1 else 'No Change'),
                'Actual_Direction': 'Up' if actual_direction == 1 else ('Down' if actual_direction == -1 else 'No Change'),
                'Profit_Loss': profit,
                'Capital': capital
            })

        st.write(f"**Toplam İşlem Sayısı:** {num_trades}")
        st.write(f"**Başlangıç Sermayesi:** {currency_symbol}{initial_capital:.2f}")
        st.write(f"**Toplam Kar/Zarar (Test Verisi Üzerinde):** {currency_symbol}{total_profit_loss:.2f}")
        st.write(f"**Nihai Sermaye:** {currency_symbol}{capital:.2f}")
        st.write(f"**Yüzdesel Getiri:** {((capital - initial_capital) / initial_capital * 100):.2f}%")

        if num_trades > 0:
            st.dataframe(pd.DataFrame(trading_results))
        else:
            st.info("Sanal işlem simülasyonu için yeterli veri bulunamadı.")


        # --- Gelecek Tahmini ---
        st.subheader(f"🚀 {prediction_days} Gün Sonraki Fiyat Tahmini")

        if not X.empty:
            last_day_data_raw = X.iloc[[-1]]
            
            if hasattr(scaler, 'feature_names_in_'):
                expected_features = scaler.feature_names_in_
                last_day_data_aligned = last_day_data_raw.reindex(columns=expected_features, fill_value=0)
                
                if last_day_data_aligned.isnull().values.any():
                    st.warning("Gelecek tahmin verisinde NaN değerler bulundu. Tahminler hatalı olabilir. NaN değerler 0 ile dolduruluyor.")
                    last_day_data_aligned.fillna(0, inplace=True) 

                last_day_data_scaled = scaler.transform(last_day_data_aligned)
                
                next_day_prediction = model.predict(last_day_data_scaled)[0]
                
                # BURAYI GÜNCELLEYİN: Tahmini fiyatı seçilen para birimi sembolüyle gösterin
                st.success(f"**{selected_coin_name} için gelecek {prediction_days} gün sonraki tahmini kapanış fiyatı:** **{currency_symbol}{next_day_prediction:.2f}**")

                current_price = data['Close'].iloc[-1]
                # BURAYI GÜNCELLEYİN: Mevcut fiyatı seçilen para birimi sembolüyle gösterin
                st.info(f"**Mevcut (Son) Kapanış Fiyatı:** {currency_symbol}{current_price:.2f}")

                price_change_percent = ((next_day_prediction - current_price) / current_price) * 100
                if price_change_percent > 0:
                    st.metric(label="Tahmini Fiyat Değişimi", value=f"%{price_change_percent:.2f}", delta="Yükseliş", delta_color="normal")
                else:
                    st.metric(label="Tahmini Fiyat Değişimi", value=f"%{price_change_percent:.2f}", delta="Düşüş", delta_color="inverse")
                
                st.markdown("---")

                # --- Potansiyel Kazanç/Kayıp Simülasyonu ---
                st.subheader("📊 Potansiyel Kazanç/Kayıp Simülasyonu")
                
                # currency_symbol değişkeninin bu noktada tanımlı olduğundan emin olun.
                # (Daha önce anlattığım gibi, bu değişkeni uygulamanızın başında
                #  selected_currency_symbol'a göre ayarlamış olmanız gerekir.)
                
                investment_amount = st.number_input(
                    f"Yatırım yapmayı düşündüğünüz miktar ({currency_symbol})",
                    min_value=0.0,
                    value=1000.0, # Varsayılan yatırım miktarı
                    step=100.0
                )

                if investment_amount > 0:
                    potential_change_amount = (investment_amount * price_change_percent) / 100
                    
                    if price_change_percent > 0:
                        st.success(
                            f"**{currency_symbol}{investment_amount:.2f}** tutarında bir yatırımla, tahmini olarak "
                            f"**{currency_symbol}{potential_change_amount:.2f}** kazanabilirsiniz. "
                            f"Bu, yaklaşık olarak **{currency_symbol}{investment_amount + potential_change_amount:.2f}** tutarında bir varlık değerine ulaşmanız anlamına gelir."
                        )
                    else:
                        st.error(
                            f"**{currency_symbol}{investment_amount:.2f}** tutarında bir yatırımla, tahmini olarak "
                            f"**{currency_symbol}{abs(potential_change_amount):.2f}** kaybedebilirsiniz. "
                            f"Bu, yaklaşık olarak **{currency_symbol}{investment_amount + potential_change_amount:.2f}** tutarında bir varlık değerine düşmeniz anlamına gelir."
                        )
                    st.warning("⚠️ Unutmayın: Bu tahminler modelin geçmiş verilere dayanarak yaptığı öngörülerdir ve piyasa koşulları gerçekte farklılık gösterebilir. Yatırım yapmadan önce kendi araştırmanızı yapın ve riskleri göz önünde bulundurun.")
                else:
                    st.info("Potansiyel kazanç/kayıp simülasyonunu görmek için bir yatırım miktarı girin.")
                
                st.markdown("---") # Simülasyon bölümü sonrası ayırıcı çizgi
                
                st.success("Analiz tamamlandı!")
            else:
                st.error("Ölçekleyicinin eğitim özellik isimleri bulunamadı. Gelecek fiyat tahmini yapılamıyor.")
        else:
            st.warning("Gelecek fiyat tahmini için yeterli veri bulunamadı.")