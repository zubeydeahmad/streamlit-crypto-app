# main_app.py
# (Ana Streamlit Uygulaması)
# Bu dosya Streamlit arayüzünü oluşturur, veri çekme, model eğitimi ve tahmin süreçlerini entegre eder.
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time
import joblib
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import plotly.graph_objects as go
import logging
import os
import sys
import requests # NewsAPI için

# data_fetcher.py'nin bulunduğu dizini sys.path'e ekleyin
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import myfinancialapp.analysis.data_fetcher as df # data_fetcher.py dosyasını df olarak import ediyoruz

# Loglama yapılandırması
logging.basicConfig(filename='crypto_app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logger = logging.getLogger(__name__)

# --- Sabitler ve Ayarlar ---
# Bu sabitler kullanıldıkları yerden önce tanımlanmalıdır.
FEATURE_LAG = 7 
TARGET_LAG = 1 

# İstikrarlı dönem tespiti için eşik değerler
# Bu değerleri, varlığın geçmiş davranışına göre ayarlamanız gerekebilir.
VOLATILITY_THRESHOLD = 0.005 # Normalized Volatility için %0.5 eşiği (örneğin 0.005 = 0.5%)
TREND_FLATNESS_THRESHOLD = 0.0001 # Ortalama günlük getiri için 0.0001 eşiği (örneğin 0.0001 = %0.01)
STABILITY_WINDOW = 20 # Oynaklık ve trend hesaplamak için kullanılan gün sayısı

# --- Streamlit Sayfa Yapılandırması ve CSS Yükleme ---
st.set_page_config(layout="wide", page_title="Finansal Varlık Analiz ve Tahmin Uygulaması", page_icon="📈")

# CSS dosyasını yükle
def load_css(file_name):
    try:
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"{file_name} dosyası bulunamadı. Temel stil uygulanacak.")
        logger.warning(f"{file_name} dosyası bulunamadı.")

load_css("styles.css")

# --- Yardımcı Fonksiyonlar (Veri Çekme, Preprocessing, Model Eğitimi) ---
# Bu fonksiyonlar uygulamanın herhangi bir yerinde çağrılmadan önce tanımlanmalıdır.

def get_historical_data_wrapper(asset_symbol: str, asset_source: str, start_date: datetime, end_date: datetime, asset_type: str) -> pd.DataFrame:
    """Belirtilen varlık için geçmiş verileri çeker (DB öncelikli ve API önceliklendirmesi ile)."""
    logger.info(f"get_historical_data_wrapper çağrıldı: Sembol={asset_symbol}, Kaynak={asset_source}, Başlangıç={start_date.strftime('%Y-%m-%d')}, Bitiş={end_date.strftime('%Y-%m-%d')}, Tip={asset_type}")
    
    coinapi_key = st.secrets.get("COINAPI_API_KEY") 
    fixer_key = st.secrets.get("FIXER_API_KEY") 
    
    # Yeni df.get_historical_data_for_asset fonksiyonunu çağırıyoruz
    data = df.get_historical_data_for_asset(asset_symbol, asset_source, asset_type, 
                                            start_date, end_date, coinapi_key, fixer_key)
    
    logger.debug(f"get_historical_data_wrapper: '{asset_symbol}' için çekilen veri boş mu? {data.empty}")
    if not data.empty:
        logger.debug(f"get_historical_data_wrapper: '{asset_symbol}' için çekilen veri boyutu: {data.shape}")
        logger.debug(f"get_historical_data_wrapper: '{asset_symbol}' için ilk 5 satır:\n{data.head()}")
    
    return data

@st.cache_data(show_spinner="Veriler işleniyor...", ttl=timedelta(hours=1)) 
def preprocess_data(data: pd.DataFrame, asset_symbol: str) -> pd.DataFrame:
    """Veriye özellik mühendisliği uygular ve temizler."""
    logger.info(f"preprocess_data çağrıldı for {asset_symbol}. Gelen veri boyutu: {data.shape}")
    if data.empty:
        logger.warning(f"Preprocess for {asset_symbol}: Boş DataFrame geldi.")
        return pd.DataFrame()

    df_processed = data.copy()
    
    df_processed.index = pd.to_datetime(df_processed.index)
    df_processed = df_processed.sort_index()

    min_date = df_processed.index.min()
    max_date = df_processed.index.max()
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    df_processed = df_processed.reindex(full_date_range)
    
    df_processed.fillna(method='ffill', inplace=True)
    df_processed.fillna(method='bfill', inplace=True) 

    df_processed['Avg_Price'] = (df_processed['Open'] + df_processed['Close']) / 2

    df_processed['MA7'] = df_processed['Close'].rolling(window=7).mean()
    df_processed['MA21'] = df_processed['Close'].rolling(window=21).mean()

    if len(df_processed) >= 14:
        delta = df_processed['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_processed['RSI'] = 100 - (100 / (1 + rs))
    else:
        df_processed['RSI'] = np.nan
        logger.warning(f"Preprocess for {asset_symbol}: RSI hesaplamak için yeterli veri yok (min 14 gün gerekli).")

    if len(df_processed) >= 26:
        exp1 = df_processed['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_processed['Close'].ewm(span=26, adjust=False).mean()
        df_processed['MACD'] = exp1 - exp2
        df_processed['Signal_Line'] = df_processed['MACD'].ewm(span=9, adjust=False).mean()
    else:
        df_processed['MACD'] = np.nan
        df_processed['Signal_Line'] = np.nan
        logger.warning(f"Preprocess for {asset_symbol}: MACD hesaplamak için yeterli veri yok (min 26 gün gerekli).")

    df_processed['Daily_Return'] = df_processed['Close'].pct_change()

    # --- Yeni İstikrarlı Dönem Özellikleri ---
    if len(df_processed) >= STABILITY_WINDOW:
        # Oynaklık (Rolling Standard Deviation of Close Price)
        df_processed['Rolling_Std_Close'] = df_processed['Close'].rolling(window=STABILITY_WINDOW).std()
        # Normalleştirilmiş oynaklık (fiyatın ortalamasına göre)
        df_processed['Normalized_Volatility'] = df_processed['Rolling_Std_Close'] / df_processed['Close'].rolling(window=STABILITY_WINDOW).mean()

        # Trend Düzlüğü (Rolling Mean of Daily Return)
        df_processed['Rolling_Mean_Return'] = df_processed['Daily_Return'].rolling(window=STABILITY_WINDOW).mean()
        
        # İstikrarlı bayrağı oluşturma: Düşük oynaklık VE yatay trend
        df_processed['is_stable'] = ((df_processed['Normalized_Volatility'] < VOLATILITY_THRESHOLD) &
                                     (df_processed['Rolling_Mean_Return'].abs() < TREND_FLATNESS_THRESHOLD)).astype(int)
        
        # İlk STABILITY_WINDOW gün için NaN değerleri 0 ile doldur (istikrarsız varsayım)
        df_processed['is_stable'] = df_processed['is_stable'].fillna(0)

    else:
        df_processed['Rolling_Std_Close'] = np.nan
        df_processed['Normalized_Volatility'] = np.nan
        df_processed['Rolling_Mean_Return'] = np.nan
        df_processed['is_stable'] = 0 # Yeterli veri yoksa istikrarsız kabul et
        logger.warning(f"Preprocess for {asset_symbol}: İstikrarlı dönem özellikleri hesaplamak için yeterli veri yok (min {STABILITY_WINDOW} gün gerekli).")


    for i in range(1, FEATURE_LAG + 1):
        df_processed[f'Close_Lag_{i}'] = df_processed['Close'].shift(i)
        df_processed[f'Open_Lag_{i}'] = df_processed['Open'].shift(i)
        df_processed[f'High_Lag_{i}'] = df_processed['High'].shift(i)
        df_processed[f'Low_Lag_{i}'] = df_processed['Low'].shift(i)
        df_processed[f'Volume_Lag_{i}'] = df_processed['Volume'].shift(i)
        df_processed[f'Avg_Price_Lag_{i}'] = df_processed['Avg_Price'].shift(i)

    df_processed['Target_Close'] = df_processed['Close'].shift(-TARGET_LAG)

    # Minimum satır sayısı ihtiyacını yeni özelliklere göre güncelle
    # En uzun pencere (RSI 14, MACD 26, Stability_Window 20) ve lag feature'ı (FEATURE_LAG) dikkate alınır.
    min_rows_needed = max(FEATURE_LAG, 26, STABILITY_WINDOW) + 1 
    if len(df_processed) < min_rows_needed:
        logger.warning(f"Preprocess for {asset_symbol}: NaN temizliği sonrası veri boş kalabilir, minimum {min_rows_needed} satır gerekli, {len(df_processed)} mevcut.")
        st.warning(f"Seçilen tarih aralığı için yeterli veri bulunamadı. Lütfen daha uzun bir tarih aralığı seçin (en az {min_rows_needed} gün).")
        return pd.DataFrame()

    df_processed.dropna(inplace=True)
    
    if df_processed.empty:
        logger.error(f"Preprocess for {asset_symbol}: İşleme sonrası DataFrame boş kaldı.")
        st.error("Veri işleme sonrası boş kaldı. Lütfen veri kaynağını ve tarih aralığını kontrol edin.")
    
    logger.info(f"preprocess_data tamamlandı for {asset_symbol}. Son veri boyutu: {df_processed.shape}")
    logger.info(f"preprocess_data - df_processed.columns: {df_processed.columns.tolist()}") 
    
    return df_processed

@st.cache_data(show_spinner="Model eğitiliyor ve tahminler yapılıyor...", ttl=timedelta(hours=6)) 
def model_training_and_prediction(df_processed: pd.DataFrame, asset_name: str, num_future_days_to_predict: int):
    """Modeli eğitir ve tahminler yapar."""
    logger.info(f"model_training_and_prediction çağrıldı for {asset_name}. Gelen veri boyutu: {df_processed.shape}, Tahmin gün sayısı: {num_future_days_to_predict}")
    
    if df_processed.empty:
        st.warning(f"İşlenmiş veri boş, model {asset_name} için eğitilemiyor veya tahmin yapılamıyor.")
        logger.warning(f"Model eğitimi/tahmini: {asset_name} için işlenmiş veri boş.")
        return None, None, None, None, None

    # Beklenen temel özellikler listesi
    expected_base_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Avg_Price', 'MA7', 'MA21', 'RSI', 'MACD', 'Signal_Line', 
                             'Daily_Return', 'Rolling_Std_Close', 'Normalized_Volatility', 'Rolling_Mean_Return', 'is_stable']
    
    # Beklenen lag özellikleri listesi (FEATURE_LAG'a göre kesin)
    expected_lag_features = []
    for i in range(1, FEATURE_LAG + 1):
        for col in ['Close', 'Open', 'High', 'Low', 'Volume', 'Avg_Price']:
            expected_lag_features.append(f'{col}_Lag_{i}')
    
    # Modelin kullanacağı nihai özellikler listesini oluştur
    # Sadece `df_processed` içinde gerçekten var olan beklenen özellikleri al
    features = []
    for f in expected_base_features + expected_lag_features:
        if f in df_processed.columns:
            features.append(f)
        else:
            logger.warning(f"Beklenen özellik '{f}' '{asset_name}' için işlenmiş veride bulunamadı. Devam ediliyor ancak bu bir soruna yol açabilir.")

    target = 'Target_Close'

    if target in features: 
        features.remove(target)
    
    X = df_processed[features]
    y = df_processed[target]

    train_size = len(X) - num_future_days_to_predict 
    if train_size <= 0:
        st.warning(f"Eğitim için yeterli veri yok. En az {num_future_days_to_predict + 1} gün gereklidir.")
        logger.warning(f"Model eğitimi: {asset_name} için yeterli veri yok. {len(X)} satır var, {num_future_days_to_predict} gerekli.")
        return None, None, None, None, None

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    logger.info(f"Model Eğitimi İçin {asset_name}: X_train boyutu: {X_train.shape}, y_train boyutu: {y_train.shape}")
    logger.info(f"Model Tahmini İçin {asset_name}: X_test boyutu: {X_test.shape}, y_test boyutu: {y_test.shape}")
    
    logger.info(f"X_train sütunları: {X_train.columns.tolist()}") 
    logger.info(f"X_test sütunları: {X_test.columns.tolist()}")   


    scaler_path = "scaler.joblib"
    features_path = "features.joblib"

    try:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if not X_test.empty else np.array([])
        
        joblib.dump(scaler, scaler_path)
        joblib.dump(features, features_path) 
        logger.info(f"Ölçekleyici ve özellik listesi kaydedildi: {scaler_path}, {features_path}")

        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train_scaled, y_train)

        model_path = "xgboost_model.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Model kaydedildi: {model_path}")

        rmse = None
        if not X_test.empty:
            y_pred_test = model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            logger.info(f"Model test seti üzerinde RMSE: {rmse}")
        else:
            logger.info("Test seti boş olduğu için RMSE hesaplanamadı.")


        last_data_point_raw = df_processed.iloc[[-1]][features].copy() 
        
        if last_data_point_raw.empty:
            logger.error(f"Gelecek tahmin için son veri noktası {asset_name} bulunamadı.")
            return model, scaler, features, None, rmse

        future_predictions = []
        current_features_df = last_data_point_raw.copy() 

        for i in range(num_future_days_to_predict): 
            logger.info(f"Tahmin döngüsü {i+1}. adım. current_features_df sütunları: {current_features_df.columns.tolist()}") 
            
            if not all(col in current_features_df.columns for col in features):
                missing_cols_for_predict = [col for col in features if col not in current_features_df.columns]
                logger.error(f"Tahmin döngüsünde current_features_df'de eksik sütunlar: {missing_cols_for_predict}. Adım: {i+1}")
                st.error(f"Tahmin sırasında dahili bir hata oluştu: Gerekli sütunlar eksik ({', '.join(missing_cols_for_predict)}). Lütfen log dosyasına bakın.")
                return None, None, None, None, None


            current_features_scaled = scaler.transform(current_features_df)
            
            next_day_prediction = model.predict(current_features_scaled)[0]
            future_predictions.append(next_day_prediction)
            
            new_row_dict = {}
            
            current_actual_close_for_lag = current_features_df['Close'].iloc[0] 
            
            new_row_dict['Close'] = next_day_prediction
            new_row_dict['Open'] = next_day_prediction * (1 + np.random.uniform(-0.005, 0.005)) 
            new_row_dict['High'] = next_day_prediction * (1 + np.random.uniform(0.001, 0.01))
            new_row_dict['Low'] = next_day_prediction * (1 - np.random.uniform(0.001, 0.01))
            new_row_dict['Volume'] = current_features_df['Volume'].iloc[0] 
            new_row_dict['Avg_Price'] = (new_row_dict['Open'] + new_row_dict['Close']) / 2
            
            # LAG ÖZELLİKLERİNİ GÜNCELLEME
            for lag_num in range(FEATURE_LAG, 1, -1): 
                for col_name in ['Close', 'Open', 'High', 'Low', 'Volume', 'Avg_Price']:
                    lag_col_source = f'{col_name}_Lag_{lag_num-1}'
                    if lag_col_source in current_features_df.columns:
                        new_row_dict[f'{col_name}_Lag_{lag_num}'] = current_features_df[lag_col_source].iloc[0]
                    else:
                        new_row_dict[f'{col_name}_Lag_{lag_num}'] = np.nan 

            # Lag_1 için güncel değerleri ata
            new_row_dict['Close_Lag_1'] = current_actual_close_for_lag
            new_row_dict['Open_Lag_1'] = current_features_df['Open'].iloc[0]
            new_row_dict['High_Lag_1'] = current_features_df['High'].iloc[0]
            new_row_dict['Low_Lag_1'] = current_features_df['Low'].iloc[0]
            new_row_dict['Volume_Lag_1'] = current_features_df['Volume'].iloc[0]
            new_row_dict['Avg_Price_Lag_1'] = current_features_df['Avg_Price'].iloc[0]

            # Hareketli Ortalamaları Güncelle
            new_row_dict['MA7'] = (current_features_df['MA7'].iloc[0] * 6 + next_day_prediction) / 7
            new_row_dict['MA21'] = (current_features_df['MA21'].iloc[0] * 20 + next_day_prediction) / 21
            
            # Yeni bir gün için Daily_Return hesapla
            new_row_dict['Daily_Return'] = (next_day_prediction - current_actual_close_for_lag) / current_actual_close_for_lag if current_actual_close_for_lag != 0 else 0

            # Oynaklık ve trend düzlüğü metriklerini tahmin edilen kapanış ile güncelle
            temp_close_history = [next_day_prediction] 
            for k in range(1, STABILITY_WINDOW):
                lag_col_name = f'Close_Lag_{k}'
                if lag_col_name in current_features_df.columns:
                    temp_close_history.append(current_features_df[lag_col_name].iloc[0])
                else:
                    temp_close_history.append(np.nan) 

            temp_close_series = pd.Series(temp_close_history[:STABILITY_WINDOW]).dropna() 
            
            if len(temp_close_series) >= STABILITY_WINDOW: 
                new_row_dict['Rolling_Std_Close'] = temp_close_series.rolling(window=STABILITY_WINDOW).std().iloc[-1]
                new_row_dict['Normalized_Volatility'] = new_row_dict['Rolling_Std_Close'] / temp_close_series.rolling(window=STABILITY_WINDOW).mean().iloc[-1]
            else:
                new_row_dict['Rolling_Std_Close'] = np.nan
                new_row_dict['Normalized_Volatility'] = np.nan

            temp_return_history = [new_row_dict['Daily_Return']]
            for k in range(1, STABILITY_WINDOW):
                if 'Daily_Return' in current_features_df.columns: 
                    if k == 1: 
                        temp_return_history.append(current_features_df['Daily_Return'].iloc[0])
                    else: 
                        temp_return_history.append(np.nan)
                else:
                    temp_return_history.append(np.nan)

            temp_return_series = pd.Series(temp_return_history[:STABILITY_WINDOW]).dropna()
            if len(temp_return_series) >= STABILITY_WINDOW:
                 new_row_dict['Rolling_Mean_Return'] = temp_return_series.rolling(window=STABILITY_WINDOW).mean().iloc[-1]
            else:
                new_row_dict['Rolling_Mean_Return'] = np.nan

            new_row_dict['is_stable'] = int((new_row_dict.get('Normalized_Volatility', np.inf) < VOLATILITY_THRESHOLD) &
                                            (abs(new_row_dict.get('Rolling_Mean_Return', np.inf)) < TREND_FLATNESS_THRESHOLD))

            new_row_dict['RSI'] = current_features_df['RSI'].iloc[0] if 'RSI' in current_features_df.columns and not pd.isna(current_features_df['RSI'].iloc[0]) else np.nan
            new_row_dict['MACD'] = current_features_df['MACD'].iloc[0] if 'MACD' in current_features_df.columns and not pd.isna(current_features_df['MACD'].iloc[0]) else np.nan
            new_row_dict['Signal_Line'] = current_features_df['Signal_Line'].iloc[0] if 'Signal_Line' in current_features_df.columns and not pd.isna(current_features_df['Signal_Line'].iloc[0]) else np.nan
            
            cols_for_next_df = {col: new_row_dict.get(col, np.nan) for col in features}
            current_features_df = pd.DataFrame([cols_for_next_df], columns=features)
            
        return model, scaler, features, future_predictions, rmse

    except Exception as e:
        logger.error(f"Model eğitimi veya tahmini sırasında hata oluştu ({asset_name}): {e}", exc_info=True) 
        st.error(f"Model eğitimi veya tahmini sırasında hata oluştu: {e}. Lütfen log dosyasına bakın.")
        return None, None, None, None, None

# Session state'i ilk çalıştırmada güvenli bir şekilde başlat
if 'run_analysis' not in st.session_state:
    st.session_state['run_analysis'] = False
if 'selected_asset_for_run' not in st.session_state:
    st.session_state['selected_asset_for_run'] = None
if 'start_date_for_run' not in st.session_state:
    st.session_state['start_date_for_run'] = None
if 'end_date_for_run' not in st.session_state:
    st.session_state['end_date_for_run'] = None
if 'prediction_days_for_run' not in st.session_state:
    st.session_state['prediction_days_for_run'] = None


# --- Sidebar (Yan Panel) Tanımı ---
with st.sidebar:
    st.header("Varlık Seçimi ve Ayarlar")

    asset_options = list(df.VARLIK_BILGILERI.keys())
    selected_asset = st.selectbox("Analiz edilecek varlığı seçin:", asset_options, key="asset_select")

    end_date = datetime.now()
    min_days_for_training = max(FEATURE_LAG, 26, STABILITY_WINDOW) + TARGET_LAG + 5 
    start_date_default = end_date - timedelta(days=365*5) 

    start_date_input = st.date_input("Başlangıç Tarihi:", value=start_date_default, max_value=end_date - timedelta(days=min_days_for_training), key="start_date_input") 
    end_date_input = st.date_input("Bitiş Tarihi:", value=end_date, max_value=end_date, min_value=start_date_input + timedelta(days=min_days_for_training), key="end_date_input")

    if start_date_input >= end_date_input:
        st.error("Bitiş tarihi başlangıç tarihinden sonra olmalıdır.")
    
    if (end_date_input - start_date_input).days < min_days_for_training:
        st.warning(f"Model eğitimi için en az {min_days_for_training} günlük veri gereklidir. Lütfen tarih aralığını genişletin.")

    prediction_days = st.number_input(
        "Gelecek kaç gün için tahmin yapılsın?",
        min_value=1,
        max_value=30, 
        value=7,
        step=1,
        help="Modelin kaç gün sonrasını tahmin etmesini istediğinizi belirtin. (1-30 gün arası)"
    )

    st.markdown("---")
    st.write("Verileri Çek, Modeli Eğit ve Tahmin Yap")
    
    if st.button("Uygulamayı Çalıştır", use_container_width=True, key="run_button"):
        logger.info(" 'Uygulamayı Çalıştır' düğmesine basıldı. Session state güncelleniyor.") 
        st.session_state['run_analysis'] = True
        st.session_state['selected_asset_for_run'] = selected_asset
        st.session_state['start_date_for_run'] = start_date_input
        st.session_state['end_date_for_run'] = end_date_input
        st.session_state['prediction_days_for_run'] = prediction_days
        # st.rerun() # Bu kaldırıldı, çünkü sürekli yeniden çalıştırmaya neden oluyordu.


# --- Ana Sayfa İçeriği ---
st.title("📈 Finansal Varlık Analiz ve Tahmin Uygulaması")
st.markdown("Bu uygulama ile çeşitli finansal varlıkların geçmiş verilerini analiz edebilir, modelleyebilir ve gelecek fiyatlarını tahmin edebilirsiniz.")

st.write(f"Analiz Çalıştırma Durumu (Session State): **{st.session_state['run_analysis']}**")


# Popüler varlıklar panelini göstermek için fonksiyonu çağır
def display_popular_assets_panel():
    st.subheader("Popüler Varlıklar Piyasasına Bakış")

    # Varlık ikonları için bir sözlük tanımlayın
    ASSET_ICONS = {
        "Bitcoin": "https://s2.coinmarketcap.com/static/img/coins/64x64/1.png",
        "Ethereum (ETH)": "https://s2.coinmarketcap.com/static/img/coins/64x64/1027.png",
        "Altın": "💰", 
        "Gümüş": "🪙", 
        "Ham Petrol": "🛢️", 
        "Binance Coin (BNB)": "https://s2.coinmarketcap.com/static/img/coins/64x64/1839.png",
        "Ripple (XRP)": "https://s2.coinmarketcap.com/static/img/coins/64x64/52.png",
        "Solana (SOL)": "https://s2.coinmarketcap.com/static/img/coins/64x64/5426.png",
        "Cardano (ADA)": "https://s2.coinmarketcap.com/static/img/coins/64x64/2010.png",
        "Dogecoin (DOGE)": "https://s2.coinmarketcap.com/static/img/coins/64x64/74.png",
        "Euro/Dolar (EURUSD)": "https://flagcdn.com/w160/eu.png",
        "Sterlin/Dolar (GBPUSD)": "https://flagsapi.com/GB/flat/64.png",
        "Dolar/Türk Lirası (USDTRY)": "https://flagcdn.com/w160/tr.png"
    }

    # API anahtarlarını st.secrets'tan çekiyoruz ve data_fetcher'a iletiyoruz
    coinapi_key = st.secrets.get("COINAPI_API_KEY") 
    fixer_key = st.secrets.get("FIXER_API_KEY")

    # CoinAPI veya Fixer.io anahtarları eksikse uyarı göster
    if not coinapi_key:
        st.warning("CoinAPI anahtarı eksik olduğu için kripto varlıklar için güncel fiyatlar gösterilemeyebilir.")
    if not fixer_key:
        st.warning("Fixer API anahtarı eksik olduğu için döviz varlıkları için güncel fiyatlar gösterilemeyebilir.")


    popular_assets_df = st.cache_data(ttl=timedelta(minutes=df.POPULAR_CACHE_DURATION_MINUTES), show_spinner="Popüler varlık fiyatları güncelleniyor...")(
        df.get_popular_asset_overview_data
    )(coinapi_key=coinapi_key, fixer_key=fixer_key) 

    try:
        with open("asset_cards.html", "r", encoding="utf-8") as f:
            html_template = f.read()
        
        if not popular_assets_df.empty: 
            asset_cards_html_parts = []

            for index, row in popular_assets_df.iterrows():
                varlik = row.get('Varlık', 'Bilinmeyen Varlık') 
                fiyat = row.get('Fiyat', np.nan)
                degisim_yuzde = row.get('Değişim (%)', np.nan)

                icon_info = ASSET_ICONS.get(varlik, "❓") 
                
                if icon_info.startswith("http"): 
                    icon_html = f'<img src="{icon_info}" onerror="this.onerror=null;this.src=\'https://placehold.co/30x30/cccccc/ffffff?text=?\';" alt="{varlik} İkon" class="asset-image-icon">'
                else: 
                    icon_html = f'<div class="asset-emoji-icon">{icon_info}</div>'
                
                fiyat_str = f"{fiyat:,.2f} $" if pd.notna(fiyat) else "N/A"
                if "Euro" in varlik or "Sterlin" in varlik or "Dolar/Türk Lirası" in varlik:
                    fiyat_str = f"{fiyat:,.4f}" if pd.notna(fiyat) else "N/A"
                elif "Bitcoin" in varlik or "Ethereum" in varlik or "Solana" in varlik or "Cardano" in varlik or "Dogecoin" in varlik or "Binance Coin" in varlik or "Ripple" in varlik:
                    fiyat_str = f"{fiyat:,.2f} $" if pd.notna(fiyat) else "N/A"
                elif "Altın" in varlik or "Gümüş" in varlik or "Ham Petrol" in varlik: 
                    fiyat_str = f"{fiyat:,.2f} $" if pd.notna(fiyat) else "N/A"

                change_class = ""
                change_icon = ""
                if pd.notna(degisim_yuzde):
                    if degisim_yuzde >= 0:
                        change_class = "change-positive"
                        change_icon = "⬆️"
                    else:
                        change_class = "change-negative"
                        change_icon = "⬇️"
                    degisim_str = f"{degisim_yuzde:+.2f}%"
                else:
                    degisim_str = "N/A"
                
                card_html = f"""
            <div class="asset-card">
                {icon_html}
                <div class="text-content">
                    <span class="asset-name">{varlik}</span>
                    <div class="price-and-change">
                        <span class="asset-price">{fiyat_str}</span>
                        <span class="asset-change {change_class}">{degisim_str} <span class="change-icon">{change_icon}</span></span>
                    </div>
                </div>
            </div>"""
                asset_cards_html_parts.append(card_html)
            
            all_cards_html = f'<div class="asset-cards-container">{"".join(asset_cards_html_parts)}</div>' 

            placeholder = "<!-- INJECT_ASSET_CARDS_HERE -->"
            final_html_output = html_template.replace(placeholder, all_cards_html)
            
            st.html(final_html_output)

        else: 
            st.warning("Popüler varlıkların güncel fiyatları çekilemedi veya veritabanından yüklenemedi.")

    except FileNotFoundError:
        st.error("asset_cards.html dosyası bulunamadı. Lütfen aynı dizinde olduğundan emin olun.")
        logger.error("asset_cards.html dosyası bulunamadı.")
        st.warning("Popüler varlıkların güncel fiyatları çekilemedi.")
    except Exception as e:
        logger.error(f"HTML içeriği oluşturulurken bir hata oluştu: {e}", exc_info=True)
        st.error(f"HTML içeriği oluşturulurken bir hata oluştu: {e}. Popüler varlıklar gösterilemiyor.")
        st.warning("Popüler varlıkların güncel fiyatları çekilemedi.")

display_popular_assets_panel() 

st.markdown("---")

# --- NewsAPI Entegrasyonu için fonksiyon tanımı ---
@st.cache_data(ttl=timedelta(hours=1), show_spinner="Haberler çekiliyor...")
def get_news(query: str, api_key: str, page_size: int = 5) -> list:
    """NewsAPI.org'dan ilgili haberleri çeker."""
    if not api_key:
        logger.error("NewsAPI anahtarı belirtilmedi. Haberler çekilemiyor.")
        return []

    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize={page_size}&apiKey={api_key}"
    
    logger.info(f"NewsAPI.org'dan haberler çekiliyor. Sorgu: '{query}'")
    try:
        response = requests.get(url)
        response.raise_for_status() 
        articles = response.json().get('articles', [])
        logger.info(f"NewsAPI.org'dan {len(articles)} haber başarıyla çekildi.")
        return articles
    except requests.exceptions.RequestException as e:
        logger.error(f"NewsAPI.org'dan haber çekilirken hata: {e}")
        st.error(f"Haberler çekilirken bir hata oluştu: {e}. Lütfen API anahtarınızı veya internet bağlantınızı kontrol edin.")
        return []
    except Exception as e:
        logger.error(f"NewsAPI.org verisi işlenirken beklenmeyen hata: {e}")
        st.error(f"Haberler işlenirken bir hata oluştu: {e}.")
        return []

# --- NewsAPI Entegrasyonu ---
news_api_key = st.secrets.get("NEWS_API_KEY") 

if news_api_key: 
    st.subheader("Güncel Haberler")
    news_queries = ["gold price", "precious metals", "gold market", "stock market", "crypto market", "economic news"]
    articles = get_news(", ".join(news_queries), news_api_key, page_size=6) 

    if articles:
        st.markdown('<div class="news-container">', unsafe_allow_html=True) 
        for i, article in enumerate(articles):
            title = article.get('title', 'Başlık Yok')
            source_name = article.get('source', {}).get('name', 'Bilinmeyen Kaynak')
            published_at = article.get('publishedAt')
            description = article.get('description', 'Açıklama Yok')
            url = article.get('url', '#')

            if published_at:
                try:
                    published_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    formatted_date = published_date.strftime('%Y-%m-%d %H:%M')
                except ValueError:
                    formatted_date = "Tarih Bilgisi Yok"
            else:
                formatted_date = "Tarih Bilgisi Yok"
            
            news_card_html = f"""
            <div class="news-card">
                <h3 class="news-title">{title}</h3>
                <p class="news-meta">Kaynak: {source_name} - Tarih: {formatted_date}</p>
                <p class="news-description">{description}</p>
                <a href="{url}" target="_blank" class="news-link">Haberi Oku</a>
            </div>
            """
            st.markdown(news_card_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True) 
    else:
        st.info("Güncel haberler çekilemedi.")
else: 
    st.warning("NewsAPI anahtarı eksik olduğu için haberler gösterilemiyor.")


st.markdown("---")

# --- Ana Bölüm - Analiz ve Tahmin ---

logger.info(f"DEBUG (Before Analysis Block): st.session_state['run_analysis'] = {st.session_state['run_analysis']}")


if st.session_state['run_analysis']: 
    logger.info("DEBUG (Inside Analysis Block): Analiz bölümüne girildi.") 
    st.info("Analiz ve tahminleriniz hazırlanıyor...") 

    current_selected_asset = st.session_state['selected_asset_for_run']
    current_start_date = st.session_state['start_date_for_run']
    current_end_date = st.session_state['end_date_for_run']
    current_prediction_days = st.session_state['prediction_days_for_run']

    st.subheader(f"Seçilen Varlık: {current_selected_asset} İçin Analiz Sonuçları")
    
    asset_info = df.VARLIK_BILGILERI[current_selected_asset]
    asset_symbol = asset_info["sembol"]
    asset_source = asset_info["kaynak"]
    asset_type = asset_info["tip"] # Varlık tipini de geçiyoruz

    st.info(f"'{current_selected_asset}' için geçmiş veriler çekiliyor...")
    logger.info(f"Analiz: Geçmiş veri çekiliyor: {asset_symbol}, Kaynak: {asset_source}, Başlangıç: {current_start_date}, Bitiş: {current_end_date}, Tip: {asset_type}")

    # get_historical_data_wrapper artık coinapi_key ve fixer_key parametrelerini de alıyor.
    historical_data = get_historical_data_wrapper(asset_symbol, asset_source, 
                                          datetime.combine(current_start_date, time.min), 
                                          datetime.combine(current_end_date, time.max), asset_type)
    logger.info(f"Analiz: Geçmiş veri çekme tamamlandı. Veri boyutu: {historical_data.shape if not historical_data.empty else 'boş'}")

    if not historical_data.empty:
        st.success(f"'{current_selected_asset}' için {len(historical_data)} günlük veri başarıyla çekildi.")
        
        if len(historical_data) >= 2:
            latest_close = historical_data['Close'].iloc[-1]
            previous_close = historical_data['Close'].iloc[-2]
            price_change = latest_close - previous_close
            price_change_percent = (price_change / previous_close) * 100 if previous_close != 0 else 0

            delta_color = "inverse" if price_change < 0 else "normal" 
            
            def format_price(price, asset_name):
                if "Euro" in asset_name or "Sterlin" in asset_name or "Dolar/Türk Lirası" in asset_name:
                    return f"{price:,.4f}"
                else:
                    return f"${price:,.2f}"

            st.metric(
                label=f"Son Kapanış Fiyatı ({historical_data.index.max().strftime('%Y-%m-%d')})",
                value=format_price(latest_close, current_selected_asset),
                delta=f"{price_change:,.2f} ({price_change_percent:+.2f}%)",
                delta_color=delta_color
            )
        else:
            st.info("Fiyat değişimini göstermek için yeterli geçmiş veri yok (en az 2 gün).")

        st.write("İlk 5 veri satırı:")
        if not historical_data.empty: 
            st.dataframe(historical_data.head())
        
        st.info("Veriler işleniyor ve özellikler oluşturuluyor...")
        logger.info(f"Analiz: Veri işleme başlatılıyor: {asset_symbol}")
        processed_data = preprocess_data(historical_data, asset_symbol)
        logger.info(f"Analiz: Veri işleme tamamlandı. İşlenmiş veri boyutu: {processed_data.shape if not processed_data.empty else 'boş'}")


        if not processed_data.empty:
            st.success("Veriler başarıyla işlendi ve özellikler oluşturuldu.")
            st.write("İşlenmiş verinin son 5 satırı (özelliklerle birlikte):")
            if not processed_data.empty: 
                st.dataframe(processed_data.tail())

            st.info(f"'{current_selected_asset}' için model eğitiliyor ve tahminler yapılıyor...")
            logger.info(f"Analiz: Model eğitimi ve tahmin başlatılıyor: {asset_symbol}")
            model, scaler, features_list, future_predictions, rmse = model_training_and_prediction(processed_data, current_selected_asset, current_prediction_days)
            logger.info(f"Analiz: Model eğitimi ve tahmin tamamlandı. RMSE: {rmse}")

            if model and future_predictions is not None:
                st.success("Model başarıyla eğitildi ve gelecek tahminler yapıldı.")
                st.metric("Model RMSE (Ortalama Kare Hata):", f"{rmse:.4f}")

                prediction_dates = pd.date_range(start=historical_data.index.max() + timedelta(days=1), periods=current_prediction_days)
                
                if future_predictions:
                    next_day_date = prediction_dates[0]
                    next_day_prediction_value = future_predictions[0]
                    
                    def format_price(price, asset_name):
                        if "Euro" in asset_name or "Sterlin" in asset_name or "Dolar/Türk Lirası" in asset_name:
                            return f"{price:,.4f}"
                        else:
                            return f"${price:,.2f}"

                    st.subheader(f"Yarınki Tahmini Kapanış Fiyatı ({next_day_date.strftime('%Y-%m-%d')}) :orange[{format_price(next_day_prediction_value, current_selected_asset)}]")
                    
                    if not historical_data.empty and 'Close' in historical_data.columns and len(historical_data) > 0:
                        last_real_close = historical_data['Close'].iloc[-1]
                        if last_real_close != 0:
                            change_pct = ((next_day_prediction_value - last_real_close) / last_real_close) * 100
                            change_amount = next_day_prediction_value - last_real_close
                            
                            st.markdown(f"**Değişim:** :{'red' if change_pct < 0 else 'green'}["
                                        f"{change_pct:+.2f}% ({change_amount:+.2f})] " 
                                        f"{'⬇️' if change_pct < 0 else '⬆️'}")
                st.markdown("---")

                st.subheader(f"Tüm {current_prediction_days} Günlük Gelecek Fiyat Tahminleri ({current_selected_asset})")
                predictions_df = pd.DataFrame({
                    "Tarih": prediction_dates,
                    "Tahmini Kapanış Fiyatı": future_predictions
                })
                if not predictions_df.empty: 
                    st.dataframe(predictions_df)

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], mode='lines', name='Geçmiş Kapanış Fiyatı', line=dict(color='blue')))

                if len(processed_data) > FEATURE_LAG: 
                    test_real_dates = processed_data.index[len(processed_data) - current_prediction_days:]
                    test_real_values = processed_data['Close'].iloc[len(processed_data) - current_prediction_days:]
                    fig.add_trace(go.Scatter(x=test_real_dates, y=test_real_values, mode='lines', name='Gerçek Değerler (Test)', line=dict(color='green', dash='dot')))


                fig.add_trace(go.Scatter(x=prediction_dates, y=future_predictions, mode='lines+markers', name=f'Tahmini Fiyat ({current_prediction_days} Gün)', line=dict(color='red', dash='dash')))

                fig.update_layout(
                    title=f'{current_selected_asset} Fiyat Tahmini',
                    xaxis_title='Tarih',
                    yaxis_title='Fiyat',
                    hovermode='x unified',
                    legend_title="Veri Tipi",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Model ve Ölçekleyici Yolu")
                st.write(f"- Model yolu: `xgboost_model.joblib`")
                st.write(f"- Ölçekleyici yolu: `scaler.joblib`")
                st.write(f"- Özellikler listesi yolu: `features.joblib`")
                st.warning("Not: Bu dosyalar uygulamanın çalıştığı dizinde oluşturulur.")

            else:
                st.error("Model eğitimi veya tahmini sırasında bir sorun oluştu. Lütfen log dosyasına bakın.")
        else:
            st.error("Veri işleme sonrası boş kaldı. Lütfen veri kaynağını ve tarih aralığını kontrol edin.")
    else:
        st.error(f"'{current_selected_asset}' için geçmiş veri çekilemedi. Lütfen internet bağlantınızı veya seçilen varlığın sembolünü/kaynağını kontrol edin. Log dosyasında daha fazla detay bulabilirsiniz.")
    
    # Analiz tamamlandığında session state'i sıfırla ki, kullanıcı tekrar çalıştırmadan aynı sonucu görmesin
    st.session_state['run_analysis'] = False
    # st.experimental_rerun() # Bu satır kaldırıldı.
