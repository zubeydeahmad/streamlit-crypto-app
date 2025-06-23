import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3 # Hala sqlite3.Error yakalamak için gerekli olabilir
import logging
import os
import datetime

# --- Loglama Ayarları ---
LOG_FILE = "streamlit_altin_ai.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=LOG_FILE,
                    filemode='a')
logger = logging.getLogger(__name__)

# --- Veritabanı Ayarları ---
# DB_FILE ve TABLE_NAME st.connection tarafından yönetilecek
TABLE_NAME = "gld_prices_streamlit"

# --- get_db_connection fonksiyonu kaldırıldı ---
# st.connection, bağlantı yönetimini üstleniyor

def save_data_to_db(df_to_save): # _conn argümanı kaldırıldı
    """Pandas DataFrame'i veritabanına kaydeder."""
    try:
        # st.connection kullanarak bağlantıyı al
        conn = st.connection("sqlite", type="sql")
        
        # Tabloyu oluştur (eğer yoksa) - ilk bağlantıda yapılmalı
        # Bu işlem, Streamlit'in secrets.toml dosyasındaki init_commands ile de yapılabilir.
        # Basitlik için ilk veri kaydetme girişiminde kontrol edip oluşturuyoruz.
        conn.cursor().execute(f'''
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                Date TEXT PRIMARY KEY,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                Volume INTEGER
            )
        ''')
        conn.commit() # Tablo oluşturma veya kontrol etme işlemini kaydet

        df_to_save['Date'] = df_to_save.index.strftime('%Y-%m-%d')
        df_selected = df_to_save[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Mevcut verileri çekmek için conn.query kullanın
        # TTL=0 her zaman en güncel veriyi çekmesini sağlar
        existing_dates_df = conn.query(f"SELECT Date FROM {TABLE_NAME}", ttl=0)
        existing_dates = set(existing_dates_df['Date'].tolist())
        
        df_new_data = df_selected[~df_selected['Date'].isin(existing_dates)]

        if not df_new_data.empty:
            # conn.write kullanarak veriyi veritabanına yazın
            # if_exists='append' ve primary_key='Date' Streamlit'in verileri doğru yönetmesini sağlar
            conn.write(df_new_data, table_name=TABLE_NAME, if_exists='append', primary_key='Date')
            logger.info(f"{len(df_new_data)} adet yeni veri veritabanına kaydedildi.")
            return True
        else:
            logger.info("Veritabanına eklenecek yeni veri bulunamadı.")
            return False
    except Exception as e:
        logger.error(f"Veritabanına veri kaydederken hata oluştu: {e}")
        st.error(f"Veri kaydederken hata oluştu: {e}. Log dosyasına bakın.")
        return False

@st.cache_data(ttl=3600) # Verilerin önbelleğe alınma süresi: 1 saat
def load_data_from_db(): # _conn argümanı kaldırıldı
    """Verileri veritabanından yükler."""
    try:
        conn = st.connection("sqlite", type="sql")
        # conn.query kullanarak verileri yükleyin
        df_loaded = conn.query(f"SELECT * FROM {TABLE_NAME} ORDER BY Date", ttl=3600, index_col='Date', parse_dates=['Date'])
        logger.info(f"Veritabanından {len(df_loaded)} adet veri yüklendi.")
        return df_loaded
    except Exception as e:
        logger.error(f"Veritabanından veri yüklerken hata oluştu: {e}")
        st.error(f"Veritabanından veri yüklerken hata oluştu: {e}. Log dosyasına bakın.")
        return pd.DataFrame()


@st.cache_data(ttl=600) # Veriyi 10 dakika önbellekte tut
def fetch_current_market_data():
    """Güncel döviz kurlarını ve ons altın fiyatını yfinance'dan çeker."""
    market_data = {}
    tickers = {
        "Dolar/TL": {"symbol": "TRY=X", "currency": "TL"},
        "Euro/TL": {"symbol": "EURTRY=X", "currency": "TL"},
        "Ons Altın": {"symbol": "GC=F", "currency": "$"}, # İsimden parantezi kaldırdık
        # "Gram Altın (XAUUSD)": {"symbol": "XAUUSD=X", "currency": "$"}, # Yorum satırı kaldı
        "Gümüş": {"symbol": "SI=F", "currency": "$"} # İsimden parantezi kaldırdık
    }

    st.sidebar.subheader("Piyasa Özeti (Canlı)")
    data_fetched_successfully = False

    for name, details in tickers.items():
        symbol = details["symbol"]
        currency = details["currency"]
        
        try:
            # XAUUSD=X kontrolü kaldırıldı (devre dışı)
            data = yf.Ticker(symbol).history(period="1d", interval="1m") 
            
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                
                prev_close = None
                if len(data) >= 2:
                    prev_close = data['Close'].iloc[-2] 
                
                if prev_close is not None and prev_close != 0:
                    change_percent = ((current_price - prev_close) / prev_close) * 100
                else:
                    change_percent = 0.0

                market_data[name] = {
                    "price": current_price,
                    "change_percent": change_percent
                }
                
                color = "green" if change_percent >= 0 else "red"
                
                # HTML stringini daha temiz oluştur, name'i direkt kullan
                price_text = f"{current_price:,.2f} ({change_percent:.2f}%)"
                if currency == "$":
                    price_text = f"${price_text}"
                else: # TL
                    price_text = f"{price_text} TL"
                
                st.sidebar.markdown(
                    f"<b>{name}:</b> <span style='color:{color}'>{price_text}</span>", 
                    unsafe_allow_html=True
                )
                data_fetched_successfully = True
            else:
                market_data[name] = {"price": "N/A", "change_percent": "N/A"}
                st.sidebar.warning(f"<b>{name}:</b> Veri çekilemedi veya boş döndü. (Sembol: {symbol})", unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"'{name}' ({symbol}) için piyasa verisi çekilirken hata: {e}")
            market_data[name] = {"price": "N/A", "change_percent": "N/A"}
            st.sidebar.error(f"<b>{name}:</b> Veri çekilemedi. Hata: {e}", unsafe_allow_html=True)
    
    if data_fetched_successfully:
        st.sidebar.caption(f"Veriler {datetime.datetime.now().strftime('%d/%m %H:%M')} itibarıyla günceldir.")
    else:
        st.sidebar.error("Canlı piyasa verileri çekilemedi. Lütfen internet bağlantınızı ve yfinance sembollerini kontrol edin.")
        
    return market_data

@st.cache_data
def fetch_and_prepare_data(ticker, period, interval, n_adim): # _conn argümanı kaldırıldı
    """Veriyi çeker, indikatörleri hesaplar ve modeli hazırlar."""
    
    st.info(f"{ticker} için veri yükleniyor veya çekiliyor...")
    logger.info(f"Veri çekme ve hazırlama başlatıldı. Ticker: {ticker}, Dönem: {period}")

    df_data = load_data_from_db() # _conn argümanı kaldırıldı

    fetch_from_yfinance = False
    if df_data.empty:
        fetch_from_yfinance = True
        logger.info("Veritabanında hiç veri bulunamadı, yfinance'dan çekilecek.")
    else:
        last_db_date = df_data.index.max()
        current_date = pd.Timestamp.now().normalize()
        if (current_date - last_db_date).days > 2 and current_date.dayofweek > 0:
            fetch_from_yfinance = True
            logger.info(f"Veritabanındaki veri ({last_db_date.strftime('%Y-%m-%d')}) güncel değil, yfinance'dan çekilecek.")
        elif current_date.dayofweek == 0 and (current_date - last_db_date).days > 3:
             fetch_from_yfinance = True
             logger.info(f"Veritabanındaki veri ({last_db_date.strftime('%Y-%m-%d')}) Pazartesi için güncel değil, yfinance'dan çekilecek.")
        else:
            logger.info(f"Veritabanındaki veri ({last_db_date.strftime('%Y-%m-%d')}) güncel, kullanılacak.")
            st.success("Veriler veritabanından başarıyla yüklendi.")


    if fetch_from_yfinance:
        try:
            hisse = yf.Ticker(ticker)
            new_df = hisse.history(period=period, interval=interval)

            if new_df.empty:
                raise ValueError(f"'{ticker}' için yfinance'dan veri çekilemedi. Lütfen sembolü kontrol edin.")

            if not new_df.index.is_monotonic_increasing:
                new_df = new_df.sort_index()

            df_to_save_yf = new_df[['Open', 'High', 'Low', 'Close', 'Volume']]
            if save_data_to_db(df_to_save_yf): # _conn argümanı kaldırıldı
                df_data = load_data_from_db() # _conn argümanı kaldırıldı
                st.success("Yeni veriler yfinance'dan çekildi ve veritabanına kaydedildi.")
            else:
                st.warning("yfinance'dan veri çekildi ancak veritabanına kaydedilemedi veya güncel veri yoktu. Çekilen veri geçici olarak kullanılıyor.")
                df_data = pd.concat([df_data, df_to_save_yf[~df_to_save_yf.index.isin(df_data.index)]]).sort_index()

        except Exception as e:
            logger.error(f"yfinance'dan veri çekerken hata: {e}")
            st.error(f"Veri çekme hatası: {e}. Lütfen log dosyasına bakın.")
            st.warning("Veri çekilemedi veya sembol bulunamadı. Lütfen başka bir sembol deneyin veya daha sonra tekrar deneyin.")
            return None, None, None, None, None, None, None, None, None
            
    if df_data.empty:
        st.error("Veri yüklenemedi veya çekilemedi. Lütfen tekrar deneyin.")
        logger.critical("Hiç veri yüklenemedi. Uygulama devam edemiyor.")
        return None, None, None, None, None, None, None, None, None

    min_data_for_indicators = max(10, 20, 14+3-1)
    if len(df_data) < min_data_for_indicators:
        st.error(f"Teknik indikatörleri hesaplamak için yeterli veri yok. En az {min_data_for_indicators} gün veri gerekiyor. Mevcut: {len(df_data)} gün.")
        return None, None, None, None, None, None, None, None, None

    df_data['SMA_10'] = ta.trend.sma_indicator(df_data['Close'], window=10)
    df_data['EMA_20'] = ta.trend.ema_indicator(df_data['Close'], window=20)
    df_data['RSI'] = ta.momentum.rsi(df_data['Close'], window=14)
    df_data['Stoch_K'] = ta.momentum.stoch(df_data['High'], df_data['Low'], df_data['Close'], window=14, smooth_window=3)
    df_data['Stoch_D'] = ta.momentum.stoch_signal(df_data['High'], df_data['Low'], df_data['Close'], window=14, smooth_window=3)
    df_data.dropna(inplace=True)
    logger.info(f"Teknik indikatörler hesaplandı. Kalan veri satırı: {len(df_data)}")

    df_data['Next_Day_Close'] = df_data['Close'].shift(-1)
    df_data['Target'] = (df_data['Next_Day_Close'] > df_data['Close']).astype(int)
    df_data.dropna(inplace=True)
    logger.info("Hedef değişken oluşturuldu.")

    ozellik_sutunlari = ['Close', 'SMA_10', 'EMA_20', 'RSI', 'Stoch_K', 'Stoch_D']
    
    X = []
    y = []

    if len(df_data) < n_adim + 1: 
        logger.error(f"Veri seti boyutu ({len(df_data)}) model eğitimi için gerekli minimum {n_adim + 1} günden az.")
        st.error(f"Model eğitimi için yeterli veri yok! En az {n_adim + 1} gün veri gerekiyor. Mevcut: {len(df_data)} gün.")
        return None, None, None, None, None, None, None, None, None

    for i in range(len(df_data) - n_adim):
        X.append(df_data[ozellik_sutunlari].iloc[i : i + n_adim].values.flatten())
        y.append(df_data['Target'].iloc[i + n_adim -1])

    X = np.array(X)
    y = np.array(y)

    if X.shape[0] == 0:
        logger.error("Özellik ve hedef dizileri boş. Uygulama devam edemiyor.")
        st.error("Veri hazırlığı sırasında bir sorun oluştu. Tahmin yapılamıyor.")
        return None, None, None, None, None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Veri eğitim ve test setlerine ayrıldı. Eğitim seti boyutu: {len(X_train)}, Test seti boyutu: {len(X_test)}")

    model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000, C=0.1, class_weight='balanced')
    logger.info("Model eğitiliyor (Lojistik Regresyon)...")
    model.fit(X_train, y_train)
    logger.info("Model eğitimi tamamlandı.")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Düşüş/Sabit', 'Yükseliş'], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Model Doğruluk: {accuracy:.2f}")

    return df_data, model, X_test, y_test, accuracy, class_report, cm, ozellik_sutunlari, n_adim


# --- Fiyat Tahmin Fonksiyonu ---
def predict_future_price_iterative(model, initial_df, features_columns, n_steps, n_adim):
    """
    Belirli sayıda adım (n_steps) için gelecekteki kapanış fiyatını tahmin eder.
    Bu, mevcut yön tahmin modelini kullanarak iteratif bir yaklaşımdır.
    """
    
    required_cols_for_iterative = list(set(features_columns + ['Open', 'High', 'Low', 'Close', 'Volume'])) 
    current_df = initial_df[required_cols_for_iterative].copy()
    
    predicted_prices = [current_df['Close'].iloc[-1]] 
    predicted_dates = []

    last_known_date = initial_df.index[-1]

    max_window_size = max(10, 20, 14, 16)

    for step in range(n_steps):
        
        temp_input_df = current_df.tail(n_adim + max_window_size).copy()
        
        temp_input_df['SMA_10'] = ta.trend.sma_indicator(temp_input_df['Close'], window=10)
        temp_input_df['EMA_20'] = ta.trend.ema_indicator(temp_input_df['Close'], window=20)
        temp_input_df['RSI'] = ta.momentum.rsi(temp_input_df['Close'], window=14)
        temp_input_df['Stoch_K'] = ta.momentum.stoch(temp_input_df['High'], temp_input_df['Low'], temp_input_df['Close'], window=14, smooth_window=3)
        temp_input_df['Stoch_D'] = ta.momentum.stoch_signal(temp_input_df['High'], temp_input_df['Low'], temp_input_df['Close'], window=14, smooth_window=3)
        
        clean_input_df = temp_input_df[features_columns].dropna()

        if len(clean_input_df) < n_adim:
            st.warning(f"Gelecek {n_steps} günlük tahmin için iteratif modelde yeterli temiz veri yok. En az {n_adim} gün gerekiyor. Tahmin {step+1}. adımda kesildi.")
            logger.warning(f"Iteratif tahmin için yetersiz temiz veri, {step+1}. adımda durduruldu. Mevcut temiz veri: {len(clean_input_df)}, Gerekli: {n_adim}")
            return predicted_prices, predicted_dates 

        input_for_prediction = clean_input_df.tail(n_adim).values.flatten().reshape(1, -1)
        
        direction_prediction = model.predict(input_for_prediction)[0]
        
        last_close = current_df['Close'].iloc[-1]
        
        if len(current_df) >= 20:
            daily_returns = current_df['Close'].pct_change().dropna()
            std_dev_returns = daily_returns.std()
            if pd.isna(std_dev_returns) or std_dev_returns == 0:
                std_dev_returns = 0.005
        else:
            std_dev_returns = 0.005

        change_percentage = np.random.uniform(0.5, 1.5) * std_dev_returns 
        
        if direction_prediction == 1:
            predicted_close = last_close * (1 + change_percentage)
        else:
            predicted_close = last_close * (1 - change_percentage)
            if predicted_close <= 0:
                predicted_close = last_close * 0.99 

        predicted_prices.append(predicted_close)
        
        next_date = last_known_date + pd.Timedelta(days=1)
        while next_date.dayofweek > 4:
            next_date += pd.Timedelta(days=1)
        predicted_dates.append(next_date)
        last_known_date = next_date

        new_row_data = {
            'Open': predicted_close,
            'High': predicted_close * 1.005, 
            'Low': predicted_close * 0.995,   
            'Close': predicted_close,
            'Volume': current_df['Volume'].iloc[-1] if not current_df.empty else 0
        }
        
        new_row_df = pd.DataFrame([new_row_data], index=[last_known_date])
        
        current_df = pd.concat([current_df[['Open', 'High', 'Low', 'Close', 'Volume']], new_row_df])
        current_df = current_df.sort_index()

        current_df = current_df.tail(n_adim + max_window_size).copy()
        
    return predicted_prices, predicted_dates

# --- Streamlit Uygulaması ---
st.set_page_config(layout="wide", page_title="Altın Piyasası Yön Tahmini AI")

st.title("💰 Altın Piyasası Yön Tahmini AI")
st.markdown("Bu uygulama, altın ETF'si GLD'nin geçmiş kapanış fiyatları ve teknik indikatörleri kullanarak bir sonraki günkü fiyat yönünü tahmin eder.")

# --- Sidebar Ayarları ---
st.sidebar.header("Uygulama Ayarları")

# --- Güncel Piyasa Verileri (daha yukarı taşındı) ---
st.sidebar.markdown("---")
fetch_current_market_data() # Fonksiyon çağrısı burada
st.sidebar.markdown("---")
# --- Güncel Piyasa Verileri SONU ---

selected_ticker = st.sidebar.selectbox("Hisse/ETF Sembolü Seçin:", ["GLD", "SPY", "QQQ", "DIA"], index=0)

selected_period = st.sidebar.selectbox("Veri Geçmişi (Yıl):", ["1y", "2y", "3y", "5y", "10y", "max"], index=2)

n_adim_input = st.sidebar.slider("Model Girdisi İçin Önceki Gün Sayısı:", min_value=5, max_value=20, value=10)

st.sidebar.markdown("---")
st.sidebar.write("**Model Detayları:**")
st.sidebar.write("- **Algoritma:** Lojistik Regresyon (Yön Tahmini)")
st.sidebar.write("- **İndikatörler:** SMA (10), EMA (20), RSI (14), Stochastic (14,3)")
st.sidebar.write(f"- **Girdi Periyodu:** Son {n_adim_input} günün verileri")

predict_days_ahead = st.sidebar.slider("Kaç İş Günü Sonrası Fiyat Tahmin Edilsin?", min_value=1, max_value=20, value=3)

# --- Veritabanı bağlantısı artık Streamlit tarafından yönetiliyor ---

st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Veri ve Model Eğitimi")
    # fetch_and_prepare_data fonksiyonuna _conn argümanı artık gönderilmiyor
    df_data, model, X_test, y_test, accuracy, class_report, cm, ozellik_sutunlari, n_adim_used = \
        fetch_and_prepare_data(selected_ticker, selected_period, "1d", n_adim=n_adim_input)

    if df_data is not None and model is not None:
        st.subheader("📊 Tarihsel Kapanış Fiyatları ve İndikatörler")
        fig_prices = plt.figure(figsize=(12, 6))
        plt.plot(df_data.index, df_data['Close'], label=f'{selected_ticker} Kapanış Fiyatı', color='blue', linewidth=1)
        plt.plot(df_data.index, df_data['SMA_10'], label='10 Günlük SMA', color='orange', linestyle='--')
        plt.plot(df_data.index, df_data['EMA_20'], label='20 Günlük EMA', color='green', linestyle='-.')
        plt.title(f'{selected_ticker} Kapanış Fiyatı ve Teknik İndikatörler')
        plt.xlabel('Tarih')
        plt.ylabel('Fiyat')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig_prices)

        st.subheader("🚀 Model Performansı (Test Verisi)")
        st.write(f"**Doğruluk (Accuracy):** **{accuracy:.2%}**")

        st.write("**Sınıflandırma Raporu:**")
        report_df = pd.DataFrame(class_report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0, subset=['precision', 'recall', 'f1-score']))

        st.write("**Hata Matrisi (Confusion Matrix):**")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Tahmini Düşüş/Sabit', 'Tahmini Yükseliş'],
                    yticklabels=['Gerçek Düşüş/Sabit', 'Gerçek Yükseliş'], ax=ax_cm)
        ax_cm.set_title('Hata Matrisi')
        ax_cm.set_ylabel('Gerçek Sınıf')
        ax_cm.set_xlabel('Tahmini Sınıf')
        st.pyplot(fig_cm)

        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"**Model Yorumu:**")
        st.write(f"- Model, {len(y_test)} test örneği üzerinde eğitildi.")
        st.write(f"- **{tp}** adet gerçek yükselişi doğru bildi.")
        st.write(f"- **{tn}** adet gerçek düşüş/sabit kalışı doğru bildi.")
        st.write(f"- **{fp}** adet durumda, **yükseliş beklerken düşüş/sabit kalış** gerçekleşti (False Positive).")
        st.write(f"- **{fn}** adet durumda, **düşüş/sabit kalış beklerken yükseliş** gerçekleşti (False Negative).")
        st.write(f"- **Toplam Yanlış Tahmin Sayısı (Hata):** {fp + fn}")
        st.write(f"- **Modelin Başarısızlık Oranı:** **{(fp + fn) / len(y_test):.2%}**")


with col2:
    st.header("📈 Gelecek Gün Tahmini")
    if df_data is not None and model is not None and len(df_data) >= n_adim_used:
        try:
            current_close_price = df_data['Close'].iloc[-1]
            st.markdown(f"Bugünün (Son Bilinen) Kapanış Fiyatı: **${current_close_price:.2f}**")

            predicted_prices_list, predicted_dates_list = predict_future_price_iterative(
                model, df_data, ozellik_sutunlari, predict_days_ahead, n_adim_used
            )
            
            if predicted_prices_list and len(predicted_prices_list) > predict_days_ahead:
                predicted_final_price = predicted_prices_list[predict_days_ahead]
                
                st.markdown(f"**{predict_days_ahead} İş Günü Sonraki Tahmini Fiyat:**")
                st.success(f"### **${predicted_final_price:.2f}**")

                if predicted_final_price > current_close_price:
                    st.markdown("Tahmini Yön: **YÜKSELİŞ** ⬆️")
                else:
                    st.markdown("Tahmini Yön: **DÜŞÜŞ / SABİT KALMA** ⬇️")

                fig_future = plt.figure(figsize=(10, 5))
                
                past_days_to_show = 30
                plot_past_data = df_data['Close'].tail(past_days_to_show).tolist()
                
                all_prices_for_plot = plot_past_data + predicted_prices_list[1:] 
                
                last_known_date_for_plot = df_data.index[-1]
                past_dates_for_plot = pd.to_datetime(df_data.index[-past_days_to_show:].tolist())
                
                all_dates_for_plot = past_dates_for_plot.tolist() + predicted_dates_list

                plt.plot(all_dates_for_plot, all_prices_for_plot, label=f'{selected_ticker} Kapanış ve Tahmin', color='purple', marker='o', markersize=3, linestyle='-')
                plt.axvline(x=last_known_date_for_plot, color='red', linestyle='--', label='Son Bilinen Gün')
                plt.title(f'{selected_ticker} Kapanış Fiyatı ve {predict_days_ahead} Gün Sonraki Tahmini')
                plt.xlabel('Tarih')
                plt.ylabel('Fiyat')
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig_future)

            else:
                st.warning("Gelecek fiyat tahmini yapılamadı. Yeterli veri veya model hatası. (Fonksiyondan erken çıkış olabilir)")

            st.markdown("---")
            st.subheader("Son Bilinen Veriler (Model Girişi İçin):")
            st.dataframe(df_data[ozellik_sutunlari].tail(n_adim_used))


        except Exception as e:
            st.error(f"Tahmin yapılırken hata oluştu: {e}")
            logger.error(f"Streamlit arayüzünde tahmin hatası: {e}")
    else:
        st.warning("Modeli eğitmek için yeterli veri yok veya veri yüklenemedi.")


st.markdown("---")
st.info("""
**Önemli Notlar:**
- Bu model finansal tavsiye **değildir**.
- **Gelecek fiyat tahmini, modelin bir sonraki günün yön tahminini ardışık olarak kullanarak yapılan basit bir simülasyondur.** Bu, piyasadaki karmaşıklığı tam olarak yansıtmaz ve sadece gösterim amaçlıdır.
- Finansal piyasalar karmaşık ve değişkendir; geçmiş performans gelecek için garanti değildir.
- Model, haberler, ekonomik veriler veya ani olaylar gibi dış faktörleri dikkate almaz.
""")