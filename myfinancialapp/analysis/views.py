# myfinancialapp/analysis/views.py

from django.shortcuts import render
from datetime import datetime, timedelta
from django.utils import timezone # Zaman dilimi bilgisine sahip datetime objeleri için
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import plotly # PlotlyJSONEncoder için ana plotly modülü eklendi
from plotly.offline import plot
import logging

# Makine öğrenimi için gerekli kütüphaneler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

# data_fetcher.py dosyasından gerekli fonksiyonları ve VARLIK_BILGILERI'ni import edin
from .data_fetcher import fetch_all_popular_assets_and_save, get_historical_data_from_db_or_fetch, VARLIK_BILGILERI
from .models import HistoricalData, PopularAssetCache

logger = logging.getLogger(__name__)

# --- Makine Öğrenimi Fonksiyonları ---

def train_and_predict_model(df_historical: pd.DataFrame, prediction_days: int = 30) -> tuple[pd.DataFrame, list]:
    """
    Geçmiş verilerle bir LSTM modeli eğitir ve gelecekteki fiyatları tahmin eder.
    """
    if df_historical.empty:
        logger.warning("Eğitim için geçmiş veri boş. Tahmin yapılamıyor.")
        return pd.DataFrame(), []

    # 'Date' sütununu DataFrame'in bir sütunu olarak garantile
    if df_historical.index.name == 'Date':
        df_historical = df_historical.reset_index()
    elif 'Date' not in df_historical.columns:
        df_historical['Date'] = df_historical.index
        df_historical = df_historical.reset_index(drop=True)
    
    df_historical['Date'] = pd.to_datetime(df_historical['Date'])
    df_historical = df_historical.sort_values(by='Date').drop_duplicates(subset=['Date']).reset_index(drop=True)


    # Sadece 'Close' fiyatını kullan
    data = df_historical['Close'].values.reshape(-1, 1)

    # Veriyi ölçeklendir
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Eğitim verisi oluştur
    # Son 60 günü kullanarak bir sonraki günü tahmin et
    training_data_len = int(len(scaled_data) * 0.8) # Verinin %80'i eğitim için
    train_data = scaled_data[0:training_data_len, :]

    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Modeli oluştur ve eğit (eğer henüz eğitilmediyse)
    # Basit bir LSTM modeli
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2)) # Aşırı uydurmayı azaltmak için Dropout eklendi
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1) # Epoch sayısı 1 olarak ayarlandı, hızlı test için

    # Test verisi oluştur
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Tahminleri al
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions) # Ölçeklendirmeyi geri al

    # Gelecek tahminleri
    last_60_days = scaled_data[len(scaled_data) - 60:].reshape(1, 60, 1)
    future_predictions = []
    current_input = last_60_days

    for _ in range(prediction_days):
        next_prediction = model.predict(current_input)[0][0]
        # Hata düzeltme: next_prediction'ı 3 boyutlu hale getirerek append et
        current_input = np.append(current_input[:, 1:, :], np.array([[[next_prediction]]]), axis=1)
        future_predictions.append(next_prediction) # next_prediction'ı ekledikten sonra future_predictions'a ekle

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # Tahmin DataFrame'i oluştur
    last_date = df_historical['Date'].iloc[-1]
    prediction_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
    
    df_predictions = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted_Close': future_predictions
    })
    
    logger.info(f"{prediction_days} günlük tahmin başarıyla oluşturuldu.")
    return df_predictions, predictions.flatten().tolist() # predictions.flatten().tolist() olarak döndür

def create_prediction_plot(df_historical: pd.DataFrame, df_predictions: pd.DataFrame, actual_predictions: list, asset_display_name: str) -> dict:
    """
    Geçmiş verileri, gerçek tahminleri ve gelecek tahminleri içeren bir Plotly grafiği oluşturur.
    """
    # 'Date' sütununu DataFrame'in bir sütunu olarak garantile
    if df_historical.index.name == 'Date':
        df_historical = df_historical.reset_index()
    elif 'Date' not in df_historical.columns:
        df_historical['Date'] = df_historical.index
        df_historical = df_historical.reset_index(drop=True)
    
    df_historical['Date'] = pd.to_datetime(df_historical['Date'])
    df_historical = df_historical.sort_values(by='Date').drop_duplicates(subset=['Date']).reset_index(drop=True)

    fig = go.Figure()

    # Geçmiş veriler (OHLC)
    fig.add_trace(go.Candlestick(x=df_historical['Date'],
                    open=df_historical['Open'],
                    high=df_historical['High'],
                    low=df_historical['Low'],
                    close=df_historical['Close'],
                    name='Geçmiş Fiyat',
                    increasing_line_color='green', decreasing_line_color='red'))

    # Modelin geçmiş veriler üzerindeki tahminleri (eğitim ve test)
    # df_historical'ın son kısmı actual_predictions ile eşleşir
    # Tarihleri df_historical'dan almalıyız
    if actual_predictions:
        # actual_predictions'ın uzunluğuna göre df_historical'dan ilgili tarihleri al
        actual_pred_dates = df_historical['Date'].iloc[-len(actual_predictions):]
        fig.add_trace(go.Scatter(
            x=actual_pred_dates,
            y=actual_predictions,
            mode='lines',
            name='Model Tahmini (Geçmiş)',
            line=dict(color='orange', width=2),
            marker=dict(size=4)
        ))

    # Gelecek tahminleri
    if not df_predictions.empty:
        fig.add_trace(go.Scatter(
            x=df_predictions['Date'],
            y=df_predictions['Predicted_Close'],
            mode='lines+markers',
            name='Gelecek Tahmin',
            line=dict(color='blue', width=2, dash='dash'),
            marker=dict(symbol='circle', size=5)
        ))

    fig.update_layout(
        title={
            'text': f"{asset_display_name} Fiyat Geçmişi ve Tahmini", # Dinamik başlık
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Tarih",
        yaxis_title="Fiyat",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='#F8F9FA',
        plot_bgcolor='#FFFFFF',
        font=dict(color='#343A40'),
        hovermode="x unified", # Hover efektini geliştir
        height=500 # Grafiğin yüksekliğini ayarlayın
    )
    
    logger.info("Tahmin grafiği fig objesi başarıyla oluşturuldu.")
    return fig # fig objesini döndür

# Ana sayfa görünümü
def home_view(request):
    logger.info("home_view fonksiyonu çağrıldı.")

    # Varsayılan başlangıç ve bitiş tarihleri (örn: son 1 yıl)
    end_date = timezone.now()
    start_date = end_date - timedelta(days=365) # Son 1 yıl

    # Popüler varlıkları çek ve kaydet
    popular_assets_df = fetch_all_popular_assets_and_save()
    logger.info(f"fetch_all_popular_assets_and_save() sonrası popular_assets_df boş mu: {popular_assets_df.empty}")
    logger.info(f"popular_assets_df boyutu: {popular_assets_df.shape}")
    logger.info(f"popular_assets_df ilk 5 satır:\n{popular_assets_df.head()}")


    # "Değişim (%)" sütun adını "Değişim_yuzdesi" olarak yeniden adlandır
    if 'Değişim (%)' in popular_assets_df.columns:
        popular_assets_df = popular_assets_df.rename(columns={'Değişim (%)': 'Değişim_yuzdesi'})
        logger.info("Sütun adı 'Değişim (%)' -> 'Değişim_yuzdesi' olarak yeniden adlandırıldı.")

    popular_assets_data = popular_assets_df.to_dict('records')
    logger.info(f"popular_assets_data listesi boyutu: {len(popular_assets_data)}")
    logger.info(f"popular_assets_data ilk elemanı: {popular_assets_data[0] if popular_assets_data else 'Boş'}")
    logger.info(f"popular_assets_data tipi: {type(popular_assets_data)}") # Tipi kontrol et


    # Varlık seçimleri için veritabanından mevcut varlıkları al
    asset_choices = []
    # VARLIK_BILGILERI'ni doğrudan kullan, çünkü PopularAssetCache'te semboller var
    for varlik_adi, bilgi in VARLIK_BILGILERI.items():
        asset_choices.append((bilgi["sembol"], varlik_adi))
    asset_choices = sorted(list(set(asset_choices)), key=lambda x: x[1]) # Yinelenenleri kaldır ve alfabetik sıraya göre sırala
    logger.info(f"Oluşturulan asset_choices boyutu: {len(asset_choices)}")


    selected_asset_symbol = "BTC" # Varsayılan Bitcoin
    prediction_days = 30 # Varsayılan tahmin gün sayısı
    popular_assets_data_json = json.dumps(popular_assets_data)

    # POST isteği geldiğinde form verilerini işle
    if request.method == 'POST':
        selected_asset_symbol = request.POST.get('asset_symbol', "BTC")
        try:
            prediction_days = int(request.POST.get('prediction_days', 30))
        except ValueError:
            prediction_days = 30
        logger.info(f"POST isteği alındı: selected_asset_symbol={selected_asset_symbol}, prediction_days={prediction_days}")

    # Seçilen varlığın tam adını al
    selected_asset_display_name = next((name for symbol, name in asset_choices if symbol == selected_asset_symbol), selected_asset_symbol)


    # Seçilen varlığın geçmiş verisini çek (DB'den veya dummy veri)
    df_historical = get_historical_data_from_db_or_fetch(selected_asset_symbol, start_date, end_date)
    logger.info(f"get_historical_data_from_db_or_fetch() sonrası df_historical boş mu: {df_historical.empty}")
    logger.info(f"df_historical boyutu: {df_historical.shape}")
    logger.info(f"df_historical ilk 5 satır:\n{df_historical.head()}")

    # Plotly grafik verilerini ve düzenini JSON olarak hazırlayın
    historical_chart_data_json = "[]"
    historical_chart_layout_json = "{}"
    prediction_chart_data_json = "[]"
    prediction_chart_layout_json = "{}"
    
    if not df_historical.empty:
        # df_historical'da 'Date' sütununu DataFrame'in bir sütunu olarak garantile ve temizle
        # Bu kısım, DataFrame'in indeksinden 'Date' sütununu oluşturur ve yinelenenleri kaldırır.
        if df_historical.index.name == 'Date':
            df_historical = df_historical.reset_index()
        elif 'Date' not in df_historical.columns:
            # Eğer 'Date' sütunu yoksa ve indeksin adı yoksa varsayılan 'index' adını kullan
            if df_historical.index.name is None:
                df_historical = df_historical.reset_index().rename(columns={'index': 'Date'})
            else:
                df_historical = df_historical.reset_index().rename(columns={df_historical.index.name: 'Date'})
        
        df_historical['Date'] = pd.to_datetime(df_historical['Date'])
        # Tarihe göre sırala ve yinelenen tarihleri kaldır (ilkini koru)
        df_historical = df_historical.sort_values(by='Date').drop_duplicates(subset=['Date'], keep='first').reset_index(drop=True)


        # Geçmiş fiyat grafiği için fig objesi oluştur
        historical_fig = go.Figure()
        historical_fig.add_trace(go.Candlestick(x=df_historical['Date'],
                        open=df_historical['Open'],
                        high=df_historical['High'],
                        low=df_historical['Low'],
                        close=df_historical['Close'],
                        name='OHLC'))

        historical_fig.update_layout(
            title={
                'text': f"{selected_asset_display_name} Geçmiş Fiyat Grafiği", # Dinamik başlık
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='#F8F9FA',
            plot_bgcolor='#FFFFFF',
            font=dict(color='#343A40'),
            height=400 # Grafiğin yüksekliğini ayarlayın
        )
        
        historical_chart_data_json = json.dumps(historical_fig.data, cls=plotly.utils.PlotlyJSONEncoder)
        historical_chart_layout_json = json.dumps(historical_fig.layout, cls=plotly.utils.PlotlyJSONEncoder)
        logger.info("Geçmiş grafik verileri (data ve layout) JSON olarak başarıyla oluşturuldu.")

        # Tahmin modelini eğit ve tahminleri al
        df_predictions, actual_predictions = train_and_predict_model(df_historical, prediction_days)
        logger.info(f"train_and_predict_model() sonrası df_predictions boş mu: {df_predictions.empty}")
        logger.info(f"df_predictions boyutu: {df_predictions.shape}")
        logger.info(f"actual_predictions boyutu: {len(actual_predictions)}")

        if not df_predictions.empty:
            # Tahmin grafiği için fig objesi oluştur
            prediction_fig = create_prediction_plot(df_historical, df_predictions, actual_predictions, selected_asset_display_name) # Dinamik başlık için yeni parametre
            prediction_chart_data_json = json.dumps(prediction_fig.data, cls=plotly.utils.PlotlyJSONEncoder)
            prediction_chart_layout_json = json.dumps(prediction_fig.layout, cls=plotly.utils.PlotlyJSONEncoder)
            logger.info("Tahmin grafik verileri (data ve layout) JSON olarak başarıyla oluşturuldu.")
        else:
            logger.warning(f"{selected_asset_display_name} için tahmin verisi boş veya alınamadı. Tahmin grafiği oluşturulamadı.")

    else:
        logger.warning(f"{selected_asset_display_name} için geçmiş veri boş veya alınamadı. Grafik oluşturulamadı.")

    context = {
        'asset_choices': asset_choices,
        'selected_asset': selected_asset_symbol,
        'prediction_days': prediction_days,
        'historical_chart_data_json': historical_chart_data_json, 
        'historical_chart_layout_json': historical_chart_layout_json,
        'prediction_chart_data_json': prediction_chart_data_json, 
        'prediction_chart_layout_json': prediction_chart_layout_json, 
        'popular_assets_data_json': popular_assets_data_json, 
        'popular_assets_data': popular_assets_data,
    }
    logger.info("Context verileri template'e gönderiliyor.")
    print(f"DEBUG: popular_assets_data_json içeriği: {popular_assets_data_json[:500]}...")
    print(f"DEBUG: historical_chart_data_json boş mu? {historical_chart_data_json == '[]'}")
    print(f"DEBUG: historical_chart_layout_json boş mu? {historical_chart_layout_json == '{}'}")
    print(f"DEBUG: prediction_chart_data_json boş mu? {prediction_chart_data_json == '[]'}")
    print(f"DEBUG: prediction_chart_layout_json boş mu? {prediction_chart_layout_json == '{}'}")
    print(f"DEBUG: Context'teki popular_assets_data tipi: {type(context['popular_assets_data'])}")
    print(f"DEBUG: Context'teki popular_assets_data boyutu: {len(context['popular_assets_data'])}")

    return render(request, 'analysis/dashboard.html', context)
