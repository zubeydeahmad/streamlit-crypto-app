# main_app.py
# (Ana Streamlit Uygulaması)
# Bu dosya Streamlit arayüzünü oluşturur, veri çekme, model eğitimi ve tahmin süreçlerini entegre eder.
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time # 'time' objesini import etmeyi unutmayın
import joblib
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import plotly.graph_objects as go
import logging
import os
import sys

# data_fetcher.py'nin bulunduğu dizini sys.path'e ekleyin
# (Proje yapısına göre bu yolu ayarlamanız gerekebilir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_fetcher as df # data_fetcher.py dosyasını df olarak import ediyoruz

# Loglama yapılandırması
logging.basicConfig(filename='crypto_app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logger = logging.getLogger(__name__)

# Streamlit sayfa yapılandırması
st.set_page_config(layout="wide", page_title="Finansal Varlık Analiz ve Tahmin Uygulaması", page_icon="📈")

# Başlık
st.title("📈 Finansal Varlık Analiz ve Tahmin Uygulaması")
st.markdown("Bu uygulama ile çeşitli finansal varlıkların geçmiş verilerini analiz edebilir, modelleyebilir ve gelecek fiyatlarını tahmin edebilirsiniz.")

# --- Sabitler ve Ayarlar ---
FEATURE_LAG = 7 # Modelin kullanacağı geçmiş gün sayısı (özellikler için)
TARGET_LAG = 1 # 1 gün sonrası için tahmin

# --- Yardımcı Fonksiyonlar ---
# Caching key'e asset_symbol ve tarih aralıkları ekleyerek, her farklı çağrıda yeniden veri çekilmesini sağla
@st.cache_data(show_spinner="Geçmiş veriler çekiliyor...", ttl=timedelta(hours=1))
def get_historical_data(asset_symbol: str, asset_source: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Belirtilen varlık için geçmiş verileri çeker."""
    logger.info(f"get_historical_data çağrıldı: Sembol={asset_symbol}, Kaynak={asset_source}, Başlangıç={start_date.strftime('%Y-%m-%d')}, Bitiş={end_date.strftime('%Y-%m-%d')}")
    try:
        if asset_source == "yfinance":
            data = df.get_yfinance_data(asset_symbol, start_date, end_date)
        elif asset_source == "coinapi":
            # CoinAPI sembolleri genellikle BTC, ETH gibi 'base' currency formatındadır.
            # Yfinance'daki BTC-USD gibi sembolleri burada da uyumlu hale getirmemiz gerekebilir.
            # Şimdilik, CoinAPI için doğrudan asset_symbol kullanacağız.
            base_currency = asset_symbol.split('-')[0] if '-' in asset_symbol else asset_symbol
            data = df.get_coinapi_data(base_currency, days_back=(end_date - start_date).days)
        else:
            st.error("Bilinmeyen veri kaynağı.")
            return pd.DataFrame()
        
        logger.debug(f"get_historical_data: '{asset_symbol}' için çekilen veri boş mu? {data.empty}")
        if not data.empty:
            logger.debug(f"get_historical_data: '{asset_symbol}' için çekilen veri boyutu: {data.shape}")
            logger.debug(f"get_historical_data: '{asset_symbol}' için ilk 5 satır:\n{data.head()}")
        
        return data
    except Exception as e:
        logger.error(f"Geçmiş veri çekilirken hata oluştu ({asset_symbol}): {e}")
        st.error(f"Geçmiş veri çekilirken hata oluştu: {e}")
        return pd.DataFrame()

# Preprocessing ve Feature Engineering
@st.cache_data(show_spinner="Veriler işleniyor...", ttl=timedelta(hours=1))
def preprocess_data(data: pd.DataFrame, asset_symbol: str) -> pd.DataFrame:
    """Veriye özellik mühendisliği uygular ve temizler."""
    logger.info(f"preprocess_data çağrıldı for {asset_symbol}. Gelen veri boyutu: {data.shape}")
    if data.empty:
        logger.warning(f"Preprocess for {asset_symbol}: Boş DataFrame geldi.")
        return pd.DataFrame()

    df_processed = data.copy()
    
    # Tarih indeksini sağlamlaştır
    df_processed.index = pd.to_datetime(df_processed.index)
    df_processed = df_processed.sort_index()

    # Eksik günleri tamamla ve NaN değerleri önceki değerle doldur
    # Bu adım, özellikle yfinance'dan gelen ve haftasonu/tatil eksikliği olan veriler için önemli
    # Reindex yapmadan önce en küçük ve en büyük tarihi belirle
    min_date = df_processed.index.min()
    max_date = df_processed.index.max()
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    df_processed = df_processed.reindex(full_date_range)
    
    # İleriye doğru doldur (ffill) ve sonra geriye doğru doldur (bfill)
    # Bu, herhangi bir başlangıçtaki NaN'ları da doldurur.
    df_processed.fillna(method='ffill', inplace=True)
    df_processed.fillna(method='bfill', inplace=True) # Başlangıçtaki NaN'ları doldurmak için

    # Ortalama Fiyat
    df_processed['Avg_Price'] = (df_processed['Open'] + df_processed['Close']) / 2

    # Hareketli Ortalamalar
    df_processed['MA7'] = df_processed['Close'].rolling(window=7).mean()
    df_processed['MA21'] = df_processed['Close'].rolling(window=21).mean()

    # RSI (Göreceli Güç Endeksi)
    # RSI hesaplaması için gerekli minimum veri: 14 periyot
    if len(df_processed) >= 14:
        delta = df_processed['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_processed['RSI'] = 100 - (100 / (1 + rs))
    else:
        df_processed['RSI'] = np.nan
        logger.warning(f"Preprocess for {asset_symbol}: RSI hesaplamak için yeterli veri yok (min 14 gün gerekli).")

    # MACD
    # MACD hesaplaması için gerekli minimum veri: 26 periyot
    if len(df_processed) >= 26:
        exp1 = df_processed['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_processed['Close'].ewm(span=26, adjust=False).mean()
        df_processed['MACD'] = exp1 - exp2
        df_processed['Signal_Line'] = df_processed['MACD'].ewm(span=9, adjust=False).mean()
    else:
        df_processed['MACD'] = np.nan
        df_processed['Signal_Line'] = np.nan
        logger.warning(f"Preprocess for {asset_symbol}: MACD hesaplamak için yeterli veri yok (min 26 gün gerekli).")

    # Günlük Getiri
    df_processed['Daily_Return'] = df_processed['Close'].pct_change()

    # Belirli bir gecikme için kaydırma
    # Son Fiyat (t-1), (t-2), ... (t-FEATURE_LAG)
    for i in range(1, FEATURE_LAG + 1):
        df_processed[f'Close_Lag_{i}'] = df_processed['Close'].shift(i)
        df_processed[f'Open_Lag_{i}'] = df_processed['Open'].shift(i)
        df_processed[f'High_Lag_{i}'] = df_processed['High'].shift(i)
        df_processed[f'Low_Lag_{i}'] = df_processed['Low'].shift(i)
        df_processed[f'Volume_Lag_{i}'] = df_processed['Volume'].shift(i)
        df_processed[f'Avg_Price_Lag_{i}'] = df_processed['Avg_Price'].shift(i)

    # Hedef Değişken (1 gün sonraki kapanış fiyatı)
    df_processed['Target_Close'] = df_processed['Close'].shift(-TARGET_LAG)

    # NaN değerleri temizle (özellikle ilk FEATURE_LAG günleri ve göstergelerin başlangıcı)
    # Gerekli minimum veri miktarını hesapla
    min_rows_needed = max(FEATURE_LAG, 26) + 1 # En büyük lag (26 for MACD) + 1 (target)
    if len(df_processed) < min_rows_needed:
        logger.warning(f"Preprocess for {asset_symbol}: NaN temizliği sonrası veri boş kalabilir, minimum {min_rows_needed} satır gerekli, {len(df_processed)} mevcut.")
        st.warning(f"Seçilen tarih aralığı için yeterli veri bulunamadı. Lütfen daha uzun bir tarih aralığı seçin (en az {min_rows_needed} gün).")
        return pd.DataFrame()

    df_processed.dropna(inplace=True)
    
    if df_processed.empty:
        logger.error(f"Preprocess for {asset_symbol}: İşleme sonrası DataFrame boş kaldı.")
        st.error("Veri işleme sonrası boş kaldı. Lütfen veri kaynağını ve tarih aralığını kontrol edin.")

    logger.info(f"preprocess_data tamamlandı for {asset_symbol}. Son veri boyutu: {df_processed.shape}")
    logger.debug(f"preprocess_data for {asset_symbol}: İşlenmiş verinin ilk 5 satırı:\n{df_processed.head()}")
    logger.debug(f"preprocess_data for {asset_symbol}: İşlenmiş verinin son 5 satırı:\n{df_processed.tail()}")
    
    return df_processed

# Model Eğitimi ve Tahmin
def model_training_and_prediction(df_processed: pd.DataFrame, asset_name: str, num_future_days_to_predict: int):
    """Modeli eğitir ve tahminler yapar."""
    logger.info(f"model_training_and_prediction çağrıldı for {asset_name}. Gelen veri boyutu: {df_processed.shape}, Tahmin gün sayısı: {num_future_days_to_predict}")
    
    if df_processed.empty:
        st.warning(f"İşlenmiş veri boş, model {asset_name} için eğitilemiyor veya tahmin yapılamıyor.")
        logger.warning(f"Model eğitimi/tahmini: {asset_name} için işlenmiş veri boş.")
        return None, None, None, None, None

    # Özellikler listesini tanımla
    base_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Avg_Price', 'MA7', 'MA21', 'RSI', 'MACD', 'Signal_Line', 'Daily_Return']
    lag_features = [f'{col}_Lag_{i}' for i in range(1, FEATURE_LAG + 1) for col in ['Close', 'Open', 'High', 'Low', 'Volume', 'Avg_Price']]
    features = [f for f in base_features + lag_features if f in df_processed.columns] # Sadece DataFrame'de olanları al

    target = 'Target_Close'

    # Hedef değişkenin özellik listesinde olmadığından emin ol
    if target in features:
        features.remove(target)
    
    # Tüm gerekli sütunların DataFrame'de olduğundan emin ol
    missing_features = [f for f in features if f not in df_processed.columns]
    if missing_features:
        logger.error(f"Model eğitimi: {asset_name} için eksik özellik sütunları: {missing_features}")
        st.error(f"Model eğitimi için gerekli bazı veri sütunları eksik: {', '.join(missing_features)}. Lütfen veri işleme adımını kontrol edin.")
        return None, None, None, None, None

    # NaN içeren feature'ları kontrol et
    nan_in_features = df_processed[features].isnull().sum().sum()
    if nan_in_features > 0:
        logger.error(f"Model eğitimi: {asset_name} için özelliklerde NaN değerler var. Eğitim durduruldu. Toplam NaN: {nan_in_features}. Detay: {df_processed[features].isnull().sum()[df_processed[features].isnull().sum() > 0].index.tolist()}")
        st.error(f"Model eğitimi için gerekli verilerde eksiklikler var. Lütfen verilerinizi kontrol edin. Eksik Özellikler: {df_processed[features].isnull().sum()[df_processed[features].isnull().sum() > 0].index.tolist()}")
        return None, None, None, None, None

    X = df_processed[features]
    y = df_processed[target]

    # Veriyi eğitim ve test setlerine ayır
    train_size = len(X) - FEATURE_LAG
    if train_size <= 0:
        st.warning(f"Eğitim için yeterli veri yok. En az {FEATURE_LAG + 1} gün gereklidir.")
        logger.warning(f"Model eğitimi: {asset_name} için yeterli veri yok. {len(X)} satır var, {FEATURE_LAG} gerekli.")
        return None, None, None, None, None

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    logger.info(f"Model Eğitimi İçin {asset_name}: X_train boyutu: {X_train.shape}, y_train boyutu: {y_train.shape}")
    logger.info(f"Model Tahmini İçin {asset_name}: X_test boyutu: {X_test.shape}, y_test boyutu: {y_test.shape}")


    # Ölçekleyiciyi kaydetmek için dosya yolu
    scaler_path = "scaler.joblib"
    features_path = "features.joblib"

    try:
        # Veriyi ölçekle
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Ölçekleyiciyi ve özellik listesini kaydet
        joblib.dump(scaler, scaler_path)
        joblib.dump(features, features_path)
        logger.info(f"Ölçekleyici ve özellik listesi kaydedildi: {scaler_path}, {features_path}")

        # XGBoost modelini eğit
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Modeli kaydet
        model_path = "xgboost_model.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Model kaydedildi: {model_path}")

        # Test seti üzerinde tahmin yap
        y_pred_test = model.predict(X_test_scaled)

        # RMSE hesapla
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        logger.info(f"Model eğitimi ve tahmini tamamlandı for {asset_name}. RMSE: {rmse}")

        # Gelecek Tahmini için son gün verisini al
        last_data_point = df_processed[features].iloc[[-1]].copy() # Son satırı DataFrame olarak al
        
        if last_data_point.empty:
            logger.error(f"Gelecek tahmin için son veri noktası {asset_name} bulunamadı.")
            return model, scaler, features, None, rmse

        future_predictions = []
        current_features_df = last_data_point # İlk tahmin için son gerçek veri

        for i in range(num_future_days_to_predict): # Kullanıcının belirlediği gün sayısı kadar tahmin yap
            # Özellikleri ölçekle
            current_features_scaled = scaler.transform(current_features_df)
            
            # Tahmin yap
            next_day_prediction = model.predict(current_features_scaled)[0]
            future_predictions.append(next_day_prediction)
            
            # Bir sonraki tahmin için 'current_features_df'yi güncelle
            new_row_dict = {}
            
            # Mevcut kapanış değerini al (ilk iterasyon için son gerçek kapanış, sonrası için önceki tahmin)
            current_actual_close_for_lag = current_features_df['Close'].iloc[0] 
            
            # Güncel Kapanış, Açılış, Yüksek, Düşük, Hacim, Ortalama Fiyat (tahmin edilen değerler)
            new_row_dict['Close'] = next_day_prediction
            new_row_dict['Open'] = next_day_prediction * (1 + np.random.uniform(-0.005, 0.005)) 
            new_row_dict['High'] = next_day_prediction * (1 + np.random.uniform(0.001, 0.01))
            new_row_dict['Low'] = next_day_prediction * (1 - np.random.uniform(0.001, 0.01))
            new_row_dict['Volume'] = current_features_df['Volume'].iloc[0] # Hacmi sabit tut (basitlik için)

            new_row_dict['Avg_Price'] = (new_row_dict['Open'] + new_row_dict['Close']) / 2
            
            # Lag özelliklerini güncelle (t-1'den t-FEATURE_LAG'e kadar kaydır)
            for lag in range(1, FEATURE_LAG):
                for col_name in ['Close', 'Open', 'High', 'Low', 'Volume', 'Avg_Price']:
                    new_row_dict[f'{col_name}_Lag_{lag+1}'] = current_features_df[f'{col_name}_Lag_{lag}'].iloc[0]
            
            # Lag_1 (yani bir önceki günün değeri) şimdiki unlagged değerler olur
            new_row_dict['Close_Lag_1'] = current_actual_close_for_lag
            new_row_dict['Open_Lag_1'] = current_features_df['Open'].iloc[0]
            new_row_dict['High_Lag_1'] = current_features_df['High'].iloc[0]
            new_row_dict['Low_Lag_1'] = current_features_df['Low'].iloc[0]
            new_row_dict['Volume_Lag_1'] = current_features_df['Volume'].iloc[0]
            new_row_dict['Avg_Price_Lag_1'] = current_features_df['Avg_Price'].iloc[0]

            # Göstergeleri güncelle (basit yaklaşımlar)
            # MA'lar: Tahmin edilen kapanışı kullanarak basit bir şekilde güncelle
            new_row_dict['MA7'] = (current_features_df['MA7'].iloc[0] * 6 + next_day_prediction) / 7
            new_row_dict['MA21'] = (current_features_df['MA21'].iloc[0] * 20 + next_day_prediction) / 21
            
            # RSI, MACD, Signal_Line: Bunlar için karmaşık geçmiş veri serisi gerektiğinden, 
            # şimdilik son bilinen değerleri kullanacağız veya basit bir eğilim varsayacağız.
            new_row_dict['RSI'] = current_features_df['RSI'].iloc[0] if 'RSI' in current_features_df.columns and not pd.isna(current_features_df['RSI'].iloc[0]) else np.nan
            new_row_dict['MACD'] = current_features_df['MACD'].iloc[0] if 'MACD' in current_features_df.columns and not pd.isna(current_features_df['MACD'].iloc[0]) else np.nan
            new_row_dict['Signal_Line'] = current_features_df['Signal_Line'].iloc[0] if 'Signal_Line' in current_features_df.columns and not pd.isna(current_features_df['Signal_Line'].iloc[0]) else np.nan
            
            # Günlük Getiri
            new_row_dict['Daily_Return'] = (next_day_prediction - current_actual_close_for_lag) / current_actual_close_for_lag if current_actual_close_for_lag != 0 else 0

            # Yeni DataFrame'i oluştururken indeks ve sütun sırasını koru
            current_features_df = pd.DataFrame([new_row_dict], columns=features)
            
        return model, scaler, features, future_predictions, rmse

    except Exception as e:
        logger.error(f"Model eğitimi veya tahmini sırasında hata oluştu ({asset_name}): {e}")
        st.error(f"Model eğitimi veya tahmini sırasında hata oluştu: {e}. Lütfen log dosyasına bakın.")
        return None, None, None, None, None

# --- Streamlit Arayüzü ---

# Sol sütun: Varlık Seçimi ve Tarih Aralığı
with st.sidebar:
    st.header("Varlık Seçimi ve Ayarlar")

    # Varlık Seçimi
    asset_options = list(df.VARLIK_BILGILERI.keys())
    selected_asset = st.selectbox("Analiz edilecek varlığı seçin:", asset_options, key="asset_select")

    # Tarih Aralığı
    end_date = datetime.now()
    min_days_for_training = max(FEATURE_LAG, 26) + TARGET_LAG + 5 
    start_date_default = end_date - timedelta(days=365*5) 

    start_date_input = st.date_input("Başlangıç Tarihi:", value=start_date_default, max_value=end_date - timedelta(days=min_days_for_training), key="start_date_input") 
    end_date_input = st.date_input("Bitiş Tarihi:", value=end_date, max_value=end_date, min_value=start_date_input + timedelta(days=min_days_for_training), key="end_date_input")

    if start_date_input >= end_date_input:
        st.error("Bitiş tarihi başlangıç tarihinden sonra olmalıdır.")
    
    if (end_date_input - start_date_input).days < min_days_for_training:
        st.warning(f"Model eğitimi için en az {min_days_for_training} günlük veri gereklidir. Lütfen tarih aralığını genişletin.")

    # Gelecek Tahmini İçin Gün Sayısı
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
    
    # Butona basıldığında session_state'i güncelle
    if st.button("Uygulamayı Çalıştır", use_container_width=True, key="run_button"):
        st.session_state['run_analysis'] = True
        st.session_state['selected_asset_for_run'] = selected_asset
        st.session_state['start_date_for_run'] = start_date_input
        st.session_state['end_date_for_run'] = end_date_input
        st.session_state['prediction_days_for_run'] = prediction_days
    
    # Eğer ilk kez açılıyorsa veya run_analysis False ise, hiçbir şey yapma
    if 'run_analysis' not in st.session_state:
        st.session_state['run_analysis'] = False

# Ana Bölüm
if st.session_state.get('run_analysis', False):
    current_selected_asset = st.session_state['selected_asset_for_run']
    current_start_date = st.session_state['start_date_for_run']
    current_end_date = st.session_state['end_date_for_run']
    current_prediction_days = st.session_state['prediction_days_for_run']

    st.subheader(f"Seçilen Varlık: {current_selected_asset}")
    
    asset_info = df.VARLIK_BILGILERI[current_selected_asset]
    asset_symbol = asset_info["sembol"]
    asset_source = asset_info["kaynak"]

    st.info(f"'{current_selected_asset}' ({asset_symbol}) için veriler çekiliyor ve analiz ediliyor...")

    # datetime.combine için datetime.time.max kullanıldı
    historical_data = get_historical_data(asset_symbol, asset_source, 
                                          datetime.combine(current_start_date, time.min), # time.min kullanıldı
                                          datetime.combine(current_end_date, time.max)) # time.max kullanıldı

    if not historical_data.empty:
        st.success(f"'{current_selected_asset}' için {len(historical_data)} günlük veri başarıyla çekildi.")
        
        # Güncel Fiyat Değişimini Göster
        if len(historical_data) >= 2:
            latest_close = historical_data['Close'].iloc[-1]
            previous_close = historical_data['Close'].iloc[-2]
            price_change = latest_close - previous_close
            price_change_percent = (price_change / previous_close) * 100 if previous_close != 0 else 0

            delta_color = "inverse" if price_change < 0 else "normal"
            st.metric(
                label=f"Son Kapanış Fiyatı ({historical_data.index.max().strftime('%Y-%m-%d')})",
                value=f"{latest_close:.2f}",
                delta=f"{price_change:.2f} ({price_change_percent:.2f}%)",
                delta_color=delta_color
            )
        else:
            st.info("Fiyat değişimini göstermek için yeterli geçmiş veri yok (en az 2 gün).")

        st.write("İlk 5 veri satırı:")
        st.dataframe(historical_data.head())
        
        processed_data = preprocess_data(historical_data, asset_symbol)

        if not processed_data.empty:
            st.success("Veriler başarıyla işlendi ve özellikler oluşturuldu.")
            st.write("İşlenmiş verinin son 5 satırı (özelliklerle birlikte):")
            st.dataframe(processed_data.tail())

            st.info(f"'{current_selected_asset}' için model eğitiliyor ve tahminler yapılıyor...")
            
            model, scaler, features_list, future_predictions, rmse = model_training_and_prediction(processed_data, current_selected_asset, current_prediction_days)

            if model and future_predictions is not None:
                st.success("Model başarıyla eğitildi ve gelecek tahminler yapıldı.")
                st.metric("Model RMSE (Ortalama Kare Hata):", f"{rmse:.4f}")

                # Tahmin tarihlerini oluştur
                prediction_dates = pd.date_range(start=historical_data.index.max() + timedelta(days=1), periods=current_prediction_days)
                
                # Bir sonraki günün tahminini açıkça göster
                if future_predictions:
                    next_day_date = prediction_dates[0]
                    next_day_prediction_value = future_predictions[0]
                    st.subheader(f"Yarınki Tahmini Kapanış Fiyatı ({next_day_date.strftime('%Y-%m-%d')}) :orange[${next_day_prediction_value:,.2f}]") # Renklendirildi
                    
                    # Yüzde değişimini hesapla ve göster
                    if not historical_data.empty and 'Close' in historical_data.columns and len(historical_data) > 0:
                        last_real_close = historical_data['Close'].iloc[-1]
                        if last_real_close != 0:
                            change_pct = ((next_day_prediction_value - last_real_close) / last_real_close) * 100
                            delta_color_pred = "inverse" if change_pct < 0 else "normal"
                            st.markdown(f"**Değişim:** :{'red' if change_pct < 0 else 'green'}["
                                        f"{change_pct:+.2f}% ({next_day_prediction_value - last_real_close:+.2f})] "
                                        f"{'⬇️' if change_pct < 0 else '⬆️'}")
                st.markdown("---")

                st.subheader(f"Tüm {current_prediction_days} Günlük Gelecek Fiyat Tahminleri ({current_selected_asset})")
                predictions_df = pd.DataFrame({
                    "Tarih": prediction_dates,
                    "Tahmini Kapanış Fiyatı": future_predictions
                })
                predictions_df["Tahmini Kapanış Fiyatı"] = predictions_df["Tahmini Kapanış Fiyatı"].round(2)
                st.dataframe(predictions_df)

                # Tahminleri görselleştirme
                fig = go.Figure()

                # Geçmiş kapanış fiyatları
                fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], mode='lines', name='Geçmiş Kapanış Fiyatı', line=dict(color='blue')))

                # Gerçek değerler (test setinden) - sadece görselleştirme için, modelin kullandığı gerçek test setidir.
                if len(processed_data) > FEATURE_LAG:
                    test_real_dates = processed_data.index[-FEATURE_LAG:]
                    test_real_values = processed_data['Close'].iloc[-FEATURE_LAG:]
                    fig.add_trace(go.Scatter(x=test_real_dates, y=test_real_values, mode='lines', name='Gerçek Değerler (Test)', line=dict(color='green', dash='dot')))


                # Gelecek Tahminleri
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
