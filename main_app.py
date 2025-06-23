# main_app.py
# (Ana Streamlit Uygulaması)
# Bu, kullanıcı arayüzünü ve tüm diğer modüllerin birleşimini sağlayan ana uygulamadır.
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import xgboost as xgb # XGBoost sürümünü kontrol etmek için eklendi

# Diğer modülleri içe aktarma
from utils import add_technical_indicators, add_market_time_features, calculate_volatility, calculate_percentage_change
from data_fetcher import VARLIK_BILGILERI, get_yfinance_data, get_coinapi_data, init_db, fetch_exchange_rates
from ai_model import prepare_data_for_model, train_xgboost_model, evaluate_model, predict_next_day_price, save_model, load_model, MODEL_PATH, SCALER_PATH, FEATURES_PATH

# --- Loglama Yapılandırması (Ana Uygulama için) ---
logging.basicConfig(filename='crypto_app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logger = logging.getLogger(__name__)

# --- Streamlit sayfa yapılandırması ---
st.set_page_config(page_title="Sanal Yatırım Sepeti Simülasyonu", layout="wide")

# --- Veritabanını Başlat ---
init_db()

# --- XGBoost Sürüm Kontrolü ---
try:
    xgboost_version = xgb.__version__
    st.sidebar.info(f"Yüklü XGBoost Sürümü: **{xgboost_version}**")
    logger.info(f"Uygulama çalışıyor. Yüklü XGBoost Sürümü: {xgboost_version}")
except Exception as e:
    st.sidebar.error(f"XGBoost sürümü kontrol edilirken hata oluştu: {e}")
    logger.error(f"XGBoost sürümü kontrol edilirken hata: {e}")

# --- Kullanıcıdan Giriş Alma ---
st.header("Yatırım Sepetinizi Oluşturun")

baslangic_bakiyesi = st.number_input("Başlangıç Bakiyeniz (USD):", min_value=100.0, value=1000.0, step=10.0)
st.write(f"Mevcut Bakiyeniz: ${baslangic_bakiyesi:,.2f}")

selected_varliklar = st.multiselect(
    "Yatırım yapmak istediğiniz varlıkları seçin:",
    list(VARLIK_BILGILERI.keys())
)

yatirim_tutarlari = {}
kalan_bakiye = baslangic_bakiyesi
yatirim_gecerli = True

if selected_varliklar:
    st.subheader("Yatırım Tutarlarını Belirleyin:")
    for varlik in selected_varliklar:
        # Maksimum tutarı kalan bakiye ile sınırla, ancak sıfırın altına düşmesin.
        max_tutar = max(0.0, kalan_bakiye) 
        tutar = st.number_input(
            f"{varlik} için yatırım tutarı (USD):",
            min_value=0.0,
            max_value=max_tutar,
            value=min(10.0, max_tutar) if max_tutar > 0 else 0.0, # Başlangıç değeri de max_tutar'ı aşmamalı
            step=1.0,
            key=f"input_{varlik}"
        )
        yatirim_tutarlari[varlik] = tutar
        kalan_bakiye -= tutar
        
        st.write(f"Kalan Bakiye: ${kalan_bakiye:,.2f}")
    
    if kalan_bakiye < 0:
        st.error("Yatırım tutarı bakiyenizi aşıyor. Lütfen düzeltin.")
        yatirim_gecerli = False
    # Kullanıcı tüm tutarları girdikten sonra son kalan bakiyeyi kontrol et
    if sum(yatirim_tutarlari.values()) > baslangic_bakiyesi:
        st.error("Toplam yatırım tutarı başlangıç bakiyenizi aşıyor. Lütfen düzeltin.")
        yatirim_gecerli = False

else:
    st.info("Lütfen yatırım yapmak istediğiniz varlıkları seçin.")
    yatirim_gecerli = False


st.sidebar.title("📊 Analiz Ayarları")

# Tarih aralığı seçimi
end_date_default = datetime.now()
start_date_default = end_date_default - timedelta(days=365 * 5) # Varsayılan olarak son 5 yıl

date_range = st.date_input("Analiz için başlangıç ve bitiş tarihini seçin:", value=(start_date_default.date(), end_date_default.date()))

start_date = None
end_date = None

if len(date_range) == 2:
    start_date = datetime.combine(date_range[0], datetime.min.time())
    end_date = datetime.combine(date_range[1], datetime.max.time())
    if start_date >= end_date:
        st.error("Başlangıç tarihi bitiş tarihinden önce olmalıdır.")
        st.stop()
else:
    st.warning("Lütfen analiz için geçerli bir başlangıç ve bitiş tarihi aralığı seçin.")
    st.stop()


# Ortak API kurları gösterimi (data_fetcher'dan alıyoruz)
st.sidebar.subheader("Piyasa Kurları")
# @st.cache_data dekoratörünü doğrudan burada uyguluyoruz
@st.cache_data(ttl=3600)
def cached_fetch_exchange_rates(base_currency="USD"):
    return fetch_exchange_rates(base_currency)

exchange_rates = cached_fetch_exchange_rates("USD")
if exchange_rates:
    st.sidebar.write(f"USD/TRY: {exchange_rates.get('TRY', 'N/A'):,.2f}")
    st.sidebar.write(f"USD/EUR: {exchange_rates.get('EUR', 'N/A'):,.2f}")
else:
    st.sidebar.warning("Döviz kurları çekilemedi.")


st.title("📈 Gelişmiş Kripto Para Tahmin Aracı v2.2")
st.markdown("Bu araç, seçilen kripto para birimleri için geçmiş verileri analiz eder, teknik göstergeler üretir ve XGBoost makine öğrenimi modeli kullanarak bir sonraki gün için fiyat tahmini yapar.")


selected_coin_name = st.selectbox("Bir Kripto Para Seçin (Tahmin Modeli Bu Varlık Üzerinde Eğitilir):", list(VARLIK_BILGILERI.keys()))
symbol_info = VARLIK_BILGILERI[selected_coin_name]
symbol = symbol_info["sembol"]
source = symbol_info["kaynak"]

prediction_days = st.number_input(
    "Kaç iş günü sonraki fiyatı tahmin etmek istersiniz?",
    min_value=1,
    max_value=30,
    value=1,
    step=1
)

st.sidebar.subheader("Model Eğitim Seçenekleri")
use_grid_search = st.sidebar.checkbox("Modeli GridSearchCV ile Eğit (Daha doğru ama yavaş)", value=False)

metrics_dict = None
y_test_abs_eval = pd.Series()
y_pred_abs_eval = pd.Series()
predicted_price = None


# Model eğitim ve tahmin butonu
if st.button("Verileri Çek, Modeli Eğit ve Tahmin Yap"):
    if not symbol:
        st.error("Lütfen bir varlık seçin.")
        st.stop()
    elif not start_date or not end_date:
        st.error("Lütfen geçerli bir tarih aralığı seçin.")
        st.stop()
    else:
        st.subheader(f"{selected_coin_name} ({symbol}) Analizi")
        with st.spinner(f"{selected_coin_name} için veriler çekiliyor, model eğitiliyor ve tahmin yapılıyor..."):
            
            data = pd.DataFrame()
            # @st.cache_data dekoratörlerini burada uygulayın
            @st.cache_data(ttl=3600)
            def _cached_get_yfinance_data(sym, start, end):
                return get_yfinance_data(sym, start, end)

            @st.cache_data(ttl=3600)
            def _cached_get_coinapi_data(base, quote, days):
                return get_coinapi_data(base, quote, days_back=days)

            if source == "yfinance":
                data = _cached_get_yfinance_data(symbol, start_date, end_date)
            elif source == "coinapi":
                days_back = (end_date - start_date).days + 1
                data = _cached_get_coinapi_data(symbol.split('-')[0], symbol.split('-')[1], days_back=days_back)
            
            if data.empty:
                st.error("Seçilen varlık için veri çekilemedi. Lütfen ayarları veya tarih aralığını kontrol edin.")
                logger.error(f"Veri çekme başarısız: {selected_coin_name} ({symbol})")
                st.stop()
            else:
                # --- Ham Veri Kontrolü ve Görüntüleme ---
                st.subheader("Ham Veri Detayları (İlk 5 Satır ve İstatistikler)")
                st.write(f"Çekilen veri boyutu: {data.shape}")
                st.dataframe(data.head())
                st.write(data.describe())
                
                # Kapanış fiyatı için basit bir çizgi grafik
                if 'Close' in data.columns and not data['Close'].isnull().all():
                    st.subheader("Ham Kapanış Fiyatı Grafiği")
                    fig_close = go.Figure(data=[go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Kapanış Fiyatı')])
                    fig_close.update_layout(title=f'{selected_coin_name} Ham Kapanış Fiyatı', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_close, use_container_width=True)
                    
                    # Fiyat aralığı kontrolü
                    min_price = data['Close'].min()
                    max_price = data['Close'].max()
                    st.info(f"Kapanış Fiyatı Aralığı: ${min_price:,.2f} - ${max_price:,.2f}")
                    if max_price > 1000000 and selected_coin_name != "Bitcoin": # BTC için normal olabilir
                        st.warning("Kapanış fiyatları anormal derecede yüksek görünüyor. Veri kaynağını kontrol edin!")
                else:
                    st.warning("Grafik çizilemedi: Kapanış fiyatı verileri eksik veya tamamen boş.")
                # --- Ham Veri Kontrolü Bitti ---


                st.subheader("Ham Veri Grafiği")
                if 'Close' in data.columns and not data['Close'].isnull().all():
                    fig_raw = go.Figure(data=[go.Candlestick(x=data.index,
                                                             open=data['Open'],
                                                             high=data['High'],
                                                             low=data['Low'],
                                                             close=data['Close'])])
                    fig_raw.update_layout(title=f'{selected_coin_name} Fiyat Grafiği', xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig_raw, use_container_width=True)
                else:
                    st.warning("Grafik çizilemedi: Kapanış fiyatı verileri eksik veya tamamen boş.")

                # Oynaklık Analizi
                st.subheader("Oynaklık Analizi")
                volatility_data = calculate_volatility(data.copy(), window=30)
                if not volatility_data.empty and 'Annualized_Volatility' in volatility_data.columns:
                    latest_volatility = volatility_data['Annualized_Volatility'].iloc[-1]
                    st.info(f"Son 30 Günlük Yıllıklandırılmış Oynaklık: **%{latest_volatility:.2f}**")
                    fig_vol = go.Figure(data=[go.Scatter(x=volatility_data.index, y=volatility_data['Annualized_Volatility'], mode='lines', name='Yıllıklandırılmış Oynaklık')])
                    fig_vol.update_layout(title=f'{selected_coin_name} Yıllıklandırılmış Oynaklık (30 Günlük)', xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig_vol, use_container_width=True)
                else:
                    st.warning("Oynaklık verileri hesaplanamadı veya boş.")


                # Veri hazırlığı
                X_train, X_test, y_train, y_test, scaler, X_test_raw, features_cols, last_known_close_prices_for_test, prep_status = \
                    prepare_data_for_model(data.copy(), prediction_days)

                if prep_status != "success":
                    st.error(f"Veri hazırlığı başarısız oldu: {prep_status}. Lütfen tarih aralığını veya veri kaynağını kontrol edin.")
                    logger.error(f"Veri hazırlığı başarısız, eğitim atlandı. Durum: {prep_status}")
                    st.stop() # Hata durumunda uygulamayı durdur

                # Modeli yüklemeyi veya eğitme kararını burada ver
                model, loaded_scaler, loaded_features_cols, load_status = load_model()
                
                current_model = None
                current_scaler = None
                current_features_cols = None

                # Modelin ve özelliklerin uyumlu olup olmadığını kontrol et
                if load_status == "success_load" and loaded_features_cols == features_cols:
                    st.info("Kaydedilmiş model bulundu ve yüklendi. Yeniden eğitim atlandı.")
                    current_model = model
                    current_scaler = loaded_scaler
                    current_features_cols = loaded_features_cols
                else:
                    if load_status == "warning_not_found":
                        st.warning("Kaydedilmiş model bulunamadı.")
                    elif load_status.startswith("error_load_failed"):
                        st.error(f"Model yüklenirken hata oluştu: {load_status}. Model yeniden eğitilecek.")
                    elif loaded_features_cols != features_cols:
                        st.warning("Kaydedilmiş modelin özellikleri mevcut özelliklerle uyuşmuyor. Model yeniden eğitiliyor.")
                        logger.warning("Kaydedilmiş modelin özellikleri mevcut özelliklerle uyuşmuyor.")
                    
                    st.info("Model yeniden eğitiliyor...")
                    current_model, train_status = train_xgboost_model(
                        X_train, y_train, 
                        use_grid_search=use_grid_search
                    )
                    current_scaler = scaler # Yeni eğitildiği için yeni scaler'ı kullan
                    current_features_cols = features_cols # Yeni eğitildiği için yeni feature_cols'ı kullan
                    
                    if current_model:
                        st.success("XGBoost modeli başarıyla eğitildi!")
                        save_status = save_model(current_model, current_scaler, current_features_cols)
                        if save_status.startswith("error_save_failed"):
                            st.error(f"Model kaydedilirken hata oluştu: {save_status}")
                    else: # Model eğitimi başarısız oldu
                        st.error(f"Model eğitilemedi. Hata: {train_status}. Lütfen log dosyasına bakın.")
                        logger.error(f"Model eğitimi başarısız. Durum: {train_status}")
                        st.stop() # Model yoksa devam etme

                if current_model:
                    # Test setinde tahmin yap ve performansı değerlendir
                    if not X_test_raw.empty:
                        X_test_scaled_for_eval = current_scaler.transform(X_test_raw[current_features_cols])
                        y_pred_for_eval = current_model.predict(X_test_scaled_for_eval)
                        
                        metrics_dict, y_test_abs_eval, y_pred_abs_eval = evaluate_model(current_model, X_test_scaled_for_eval, y_test, y_pred_for_eval, y_test_original_prices=last_known_close_prices_for_test)
                        
                        st.subheader(f"📊 Model Performansı (Test Verisi - Mutlak Fiyatlar Üzerinden)")
                        if metrics_dict:
                            st.write(f"**Ortalama Mutlak Hata (MAE):** {metrics_dict['MAE']:.2f}")
                            st.write(f"**Kök Ortalama Kare Hata (RMSE):** {metrics_dict['RMSE']:.2f}")
                            st.write(f"**R-kare (R²):** {metrics_dict['R2']:.2f}")
                            st.write(f"**Ortalama Mutlak Yüzde Hata (MAPE):** {metrics_dict['MAPE']:.2f}%")
                        else:
                            st.warning("Model performans metrikleri hesaplanamadı.")

                    else:
                        st.warning("Test seti boş olduğu için model performansı değerlendirilemedi.")
                        logger.warning("Test seti boş olduğu için model performansı değerlendirilemedi.")

                    # Gelecek gün tahmini
                    if not X_test_raw.empty:
                        last_processed_data_point = X_test_raw.iloc[[-1]]
                        last_known_close_price = data['Close'].iloc[-1]

                        predicted_price = predict_next_day_price(current_model, current_scaler, last_processed_data_point, current_features_cols)
                        
                        if last_known_close_price is not None and predicted_price is not None and not pd.isna(predicted_price):
                            predicted_price_change_pct = calculate_percentage_change(last_known_close_price, predicted_price)
                            change_icon = "⬆️" if predicted_price_change_pct >= 0 else "⬇️"
                            st.subheader(f"🚀 {prediction_days} İş Günü Sonraki Tahmini Kapanış Fiyatı:")
                            st.markdown(f"### ${predicted_price:,.2f} ({predicted_price_change_pct:+.2f}%) {change_icon}")
                        else:
                            st.subheader(f"🚀 {prediction_days} İş Günü Sonraki Tahmini Kapanış Fiyatı:")
                            st.success(f"### ${predicted_price:,.2f}")

                        # Tahmini fiyatı ve Geçmiş Tahmin vs Gerçek Grafiği
                        fig_pred = make_subplots(rows=1, cols=1, shared_xaxes=True)
                        
                        if 'Close' in data.columns and not data['Close'].isnull().all():
                            fig_pred.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Geçmiş Kapanış'), row=1, col=1)
                            
                            last_known_date = data.index[-1].date()
                            predicted_date = last_known_date
                            for _ in range(prediction_days):
                                predicted_date += timedelta(days=1)
                                while predicted_date.weekday() >= 5: # Hafta sonlarını atla (Cumartesi=5, Pazar=6)
                                    predicted_date += timedelta(days=1)
                            
                            fig_pred.add_trace(go.Scatter(x=[data.index[-1], predicted_date], y=[data['Close'].iloc[-1], predicted_price],
                                                            mode='lines+markers', name=f'{prediction_days} Gün Sonra Tahmin',
                                                            line=dict(color='red', dash='dash'), marker=dict(size=8, symbol='star')), row=1, col=1)
                            
                            fig_pred.update_layout(title=f'{selected_coin_name} Geçmiş Fiyat ve {prediction_days} İş Günü Sonraki Tahmin',
                                                    xaxis_rangeslider_visible=False,
                                                    height=500)
                            st.plotly_chart(fig_pred, use_container_width=True)
                        else:
                            st.warning("Tahmin grafiği çizilemedi: Kapanış fiyatı verileri eksik veya tamamen boş.")
                        
                        # Yeni: Test Seti Gerçek vs Tahmin Edilen Fiyatlar Grafiği
                        if not y_test_abs_eval.empty and not y_pred_abs_eval.empty:
                            fig_test_comparison = go.Figure()
                            fig_test_comparison.add_trace(go.Scatter(x=y_test_abs_eval.index, y=y_test_abs_eval, mode='lines', name='Gerçek Fiyatlar (Test Seti)', line=dict(color='blue')))
                            fig_test_comparison.add_trace(go.Scatter(x=y_pred_abs_eval.index, y=y_pred_abs_eval, mode='lines', name='Tahmin Edilen Fiyatlar (Test Seti)', line=dict(color='green', dash='dot')))
                            fig_test_comparison.update_layout(title=f'{selected_coin_name} Test Seti: Gerçek vs Tahmin Edilen Fiyatlar', xaxis_rangeslider_visible=False)
                            st.plotly_chart(fig_test_comparison, use_container_width=True)
                        else:
                            st.warning("Test seti karşılaştırma grafiği çizilemedi: Değerlendirme verileri boş.")


                    else:
                        st.warning("Model tahmini için yeterli geçmiş işlenmiş veri bulunamadı. Grafik ve tahmin gösterilemiyor.")
                        logger.warning("predict_next_day_price için X_test_raw boş, tahmin yapılamıyor.")
                        predicted_price = None

                    st.markdown("---")
                    st.subheader("Sanal Yatırım Sepeti Simülasyonu Sonuçları")
                    if yatirim_gecerli and yatirim_tutarlari:
                        st.write(f"Başlangıç Bakiyeniz: ${baslangic_bakiyesi:,.2f}")
                        st.write(f"Toplam Yatırılan Tutar: ${sum(yatirim_tutarlari.values()):,.2f}")
                        st.write(f"Kalan Nakit: ${kalan_bakiye:,.2f}")

                        st.markdown("##### Yatırım Sepetiniz:")
                        toplam_yeni_deger = float(kalan_bakiye)

                        # Portföy simülasyonu için seçili tüm varlıkların güncel fiyatlarını bir kerede çek
                        asset_current_prices = {}
                        for varlik_name in selected_varliklar:
                            if yatirim_tutarlari.get(varlik_name, 0) > 0: # Sadece yatırım yapılan varlıklar için çek
                                varlik_info = VARLIK_BILGILERI[varlik_name]
                                varlik_symbol = varlik_info["sembol"]
                                varlik_source = varlik_info["kaynak"]

                                current_asset_data = pd.DataFrame()
                                # Son 7 günlük veriyi çek, en son kapanış fiyatını almak için
                                if varlik_source == "yfinance":
                                    current_asset_data = _cached_get_yfinance_data(varlik_symbol, datetime.now() - timedelta(days=7), datetime.now())
                                elif varlik_source == "coinapi":
                                    current_asset_data = _cached_get_coinapi_data(varlik_symbol.split('-')[0], varlik_symbol.split('-')[1], days=7)

                                if not current_asset_data.empty and 'Close' in current_asset_data.columns and not current_asset_data['Close'].empty:
                                    asset_current_prices[varlik_name] = float(current_asset_data['Close'].iloc[-1].item())
                                else:
                                    asset_current_prices[varlik_name] = None
                                    logger.warning(f"Simülasyon için '{varlik_name}' varlığının güncel 'Close' fiyatı çekilemedi.")
                                    st.warning(f"Simülasyon için '{varlik_name}' varlığının güncel fiyatı çekilemedi.")

                        # Her varlık için portföy değerini hesapla
                        for varlik, yatirilan_val in yatirim_tutarlari.items():
                            if yatirilan_val > 0:
                                alis_fiyati = asset_current_prices.get(varlik)
                                if alis_fiyati is None:
                                    continue 

                                adet = float(yatirilan_val) / alis_fiyati
                                
                                if varlik == selected_coin_name and predicted_price is not None and not pd.isna(predicted_price):
                                    tahmini_satis_fiyati = float(predicted_price)
                                else:
                                    tahmini_satis_fiyati = alis_fiyati
                                    if varlik != selected_coin_name:
                                        st.info(f"'{varlik}' için tahmin modeli kullanılmadı, güncel fiyat esas alındı.")
                                    elif pd.isna(predicted_price):
                                        st.info(f"'{varlik}' için tahmin yapılamadığından güncel fiyat esas alındı.")

                                tahmini_varlik_degeri = float(adet * tahmini_satis_fiyati)
                                kar_zarar_yuzde = calculate_percentage_change(yatirilan_val, tahmini_varlik_degeri)
                                
                                change_icon_portf = "⬆️" if kar_zarar_yuzde >= 0 else "⬇️"
                                st.write(f"- **{varlik}:** Yatırılan ${yatirilan_val:,.2f} -> Tahmini Değer ${tahmini_varlik_degeri:,.2f} (Değişim: {kar_zarar_yuzde:+.2f}%) {change_icon_portf}")
                                toplam_yeni_deger += tahmini_varlik_degeri
                        
                        toplam_degisim_yuzde = calculate_percentage_change(baslangic_bakiyesi, toplam_yeni_deger)
                        change_icon_total = "⬆️" if toplam_degisim_yuzde >= 0 else "⬇️"

                        st.markdown(f"#### Toplam Portföy Tahmini Değeri: **${toplam_yeni_deger:,.2f}**")
                        st.markdown(f"#### Toplam Portföy Değişimi: **{toplam_degisim_yuzde:+.2f}%** {change_icon_total}")
                    else:
                        st.info("Yatırım sepeti simülasyonu için lütfen geçerli varlık ve tutar girin.")
                else:
                    st.error("Model eğitilemedi veya yüklenemediği için tahmin ve portföy simülasyonu yapılamadı.")
                    logger.error("Model eğitimi veya yüklemesi başarısız, tahmin yapılamadı.")
                st.markdown("---")
