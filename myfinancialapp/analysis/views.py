# myfinancialapp/analysis/views.py

import json
import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponseBadRequest
from .data_fetcher import fetch_all_popular_assets_and_save, get_historical_data_from_db_or_fetch, VARLIK_BILGILERI
from .models import PopularAssetCache, HistoricalData
import asyncio
import logging
from datetime import datetime, timedelta

# Django'nun varsayılan kimlik doğrulama sistemi için gerekli import'lar
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.urls import reverse_lazy
from django.views.generic.edit import CreateView
from django.contrib.auth.models import User
from .forms import CustomUserCreationForm

import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

logger = logging.getLogger(__name__)

# LSTM modelini yükleyen veya oluşturan fonksiyon
def load_or_create_model(symbol, input_shape):
    model_dir = 'analysis/models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{symbol}_lstm_model.h5')

    if os.path.exists(model_path):
        logger.info(f"Yüklü model bulundu: {model_path}")
        try:
            model = load_model(model_path)
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model
        except Exception as e:
            logger.error(f"Model yüklenirken hata oluştu ({model_path}): {e}", exc_info=True)
    
    logger.info(f"Model oluşturuluyor/yeniden eğitiliyor: {model_path}")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Veri setini oluşturan fonksiyon
def create_dataset(data, time_step=1):
    X, Y = [], []
    if len(data) < time_step + 1:
        logger.warning(f"create_dataset: Veri uzunluğu ({len(data)}) time_step + 1 ({time_step + 1})'den küçük. Boş arrayler döndürülecek.")
        return np.array([]), np.array([]) 
        
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)


# Kayıt görünümü (Django'nun varsayılan kullanıcı modelini kullanır)
class RegisterView(CreateView):
    form_class = CustomUserCreationForm
    success_url = reverse_lazy('login') # Kayıt başarılı olursa giriş sayfasına yönlendir
    template_name = 'registration/register.html' # Kayıt şablonu

    def form_valid(self, form):
        response = super().form_valid(form)
        # Kullanıcıyı otomatik olarak giriş yapmasını istiyorsanız bu satırı aktif edebilirsiniz.
        # login(self.request, self.object)
        return response

# Ana dashboard görünümü
@login_required # Bu dekoratör, sadece giriş yapmış kullanıcıların erişmesini sağlar
def home_view(request): # async def yerine def olarak değiştirildi
    # Varsayılan değerler
    popular_assets_data_json = json.dumps([])
    historical_chart_json = json.dumps({}) 
    prediction_chart_json = json.dumps({}) 
    
    selected_asset_name = 'Bitcoin'
    prediction_days = 30
    currency_type = 'USD'
    investment_amount = 1000.0

    # Simülasyon sonuçları için varsayılan değerler
    simulation_message = "Simülasyon sonuçları burada gösterilecektir."
    predicted_gain_loss = None
    predicted_change_percent = None
    risk_level = None
    advice = None
    currency_symbol = '$'

    # >>> BURADA 'context' SÖZLÜĞÜNÜ BAŞLATTIK <<<
    context = {
        'asset_options': list(VARLIK_BILGILERI.keys()),
        'popular_assets_data': popular_assets_data_json,
        'historical_chart_json': historical_chart_json,
        'prediction_chart_json': prediction_chart_json,
        
        # Form için seçili değerler
        'selected_asset': selected_asset_name,
        'prediction_days': prediction_days,
        'currency_type': currency_type,
        'investment_amount': investment_amount,

        # Varlık Bilgisi ve Tahmin için
        'current_asset_price': None, # Varsayılan olarak None
        'current_day_change_percent': None, # Varsayılan olarak None
        'predicted_future_price': None, # Varsayılan olarak None

        # Simülasyon sonuçları için
        'simulation_message': simulation_message,
        'predicted_gain_loss': predicted_gain_loss,
        'predicted_change_percent': predicted_change_percent,
        'risk_level': risk_level,
        'advice': advice,
        'currency_symbol': currency_symbol,
    }
    # <<< BURADA 'context' SÖZLÜĞÜNÜ BAŞLATTIK >>>

    # Form gönderimi varsa değerleri al
    if request.method == 'POST':
        context['selected_asset'] = request.POST.get('asset_select', 'Bitcoin')
        context['prediction_days'] = int(request.POST.get('prediction_days', 30))
        context['currency_type'] = request.POST.get('currency_type', 'USD')
        try:
            context['investment_amount'] = float(request.POST.get('investment_amount', 1000.0))
        except ValueError:
            context['investment_amount'] = 1000.0 # Hatalı giriş durumunda varsayılan değer

    selected_asset_symbol = next((info["sembol"] for name, info in VARLIK_BILGILERI.items() if name == context['selected_asset']), 'BTC-USD')

    try:
        # Asenkron fonksiyonları senkron bağlamda çağırmak için asyncio.run() kullanın
        popular_assets_df = asyncio.run(fetch_all_popular_assets_and_save())
        if not popular_assets_df.empty:
            popular_assets_data = popular_assets_df.where(pd.notnull(popular_assets_df), None).to_dict(orient='records')
            context['popular_assets_data'] = json.dumps(popular_assets_data)
        else:
            logger.warning("Popüler varlıklar DataFrame boş döndü.")

    except Exception as e:
        logger.error(f"Popüler varlıkları çekerken hata: {e}", exc_info=True)
        context['popular_assets_data'] = json.dumps([{"Varlık": "Veri bulunamadı.", "Fiyat": None, "Değişim (%)": None}])
    
    if selected_asset_symbol:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5) # 5 yıllık veri çek

        # Asenkron fonksiyonu senkron bağlamda çağırmak için asyncio.run() kullanın
        df_historical = asyncio.run(get_historical_data_from_db_or_fetch(selected_asset_symbol, start_date, end_date))
        
        if not df_historical.empty and 'Close' in df_historical.columns:
            # Güncel fiyat ve değişim hesaplama
            if len(df_historical['Close']) >= 2:
                context['current_asset_price'] = df_historical['Close'].iloc[-1]
                previous_day_price = df_historical['Close'].iloc[-2]
                if previous_day_price != 0: # Sıfıra bölme hatasını önle
                    context['current_day_change_percent'] = ((context['current_asset_price'] - previous_day_price) / previous_day_price) * 100
            elif len(df_historical['Close']) == 1:
                context['current_asset_price'] = df_historical['Close'].iloc[-1]
                context['current_day_change_percent'] = 0.0 # Tek gün varsa değişim 0 kabul edilebilir
            
            # Geçmiş Fiyat Grafiği Oluşturma
            fig_historical = go.Figure(data=[go.Candlestick(
                x=df_historical.index,
                open=df_historical['Open'],
                high=df_historical['High'],
                low=df_historical['Low'], 
                close=df_historical['Close'],
                name='Geçmiş Fiyat'
            )])
            fig_historical.update_layout(
                title=f'{context["selected_asset"]} Geçmiş Fiyat Grafiği',
                xaxis_title='Tarih',
                yaxis_title='Fiyat',
                xaxis_rangeslider_visible=False,
                template='plotly_white',
                height=400
            )
            context['historical_chart_json'] = fig_historical.to_json()

            try:
                scaler = MinMaxScaler(feature_range=(0, 1))
                
                close_prices = df_historical['Close'].values.reshape(-1, 1)
                scaled_data = scaler.fit_transform(close_prices)

                time_step = 40
                
                MIN_REQUIRED_TOTAL_DATA = int((time_step + 1) / 0.20) + 1 
                
                if len(scaled_data) < MIN_REQUIRED_TOTAL_DATA: 
                    logger.warning(f"'{context['selected_asset']}' için yetersiz geçmiş veri: {len(scaled_data)} gün. Tahmin için en az {MIN_REQUIRED_TOTAL_DATA} gün gereklidir (time_step {time_step} için).")
                    context['prediction_chart_json'] = json.dumps({ 
                        'data': [],
                        'layout': {
                            'title': f'Tahmin Grafiği (En Az {MIN_REQUIRED_TOTAL_DATA} Günlük Veri Gerekli)',
                            'height': 500
                        }
                    })
                    # Yetersiz veri durumunda simülasyon hatası
                    context['simulation_message'] = f"Simülasyon için yeterli geçmiş veri yok ({len(scaled_data)} gün, en az {MIN_REQUIRED_TOTAL_DATA} gün gerekli)."
                    context['predicted_gain_loss'] = None
                    context['predicted_change_percent'] = None
                    context['risk_level'] = "Veri Yetersiz"
                    context['advice'] = "Daha fazla geçmiş veri olduğunda simülasyonu tekrar deneyin."
                else:
                    training_size = int(len(scaled_data) * 0.80)
                    train_data = scaled_data[0:training_size, :]
                    test_data = scaled_data[training_size:len(scaled_data), :]

                    if len(train_data) < time_step + 1 or len(test_data) < time_step + 1:
                        logger.warning(f"'{context['selected_asset']}' için eğitim veya test seti time_step ({time_step}) kadar veri içermiyor. Eğitim seti: {len(train_data)}, Test seti: {len(test_data)}. Tahmin grafiği çizilemiyor.")
                        context['prediction_chart_json'] = json.dumps({
                            'data': [],
                            'layout': {
                                'title': 'Tahmin Grafiği (Yetersiz Eğitim/Test Verisi)',
                                'height': 500
                            }
                        })
                        context['simulation_message'] = "Simülasyon için eğitim veya test seti yetersiz."
                        context['predicted_gain_loss'] = None
                        context['predicted_change_percent'] = None
                        context['risk_level'] = "Veri Yetersiz"
                        context['advice'] = "Daha uzun dönemli veri çekmeye çalışın."
                    else:
                        X_train, y_train = create_dataset(train_data, time_step)
                        X_test, y_test = create_dataset(test_data, time_step)

                        if X_train.size == 0 or X_test.size == 0:
                            logger.warning(f"'{context['selected_asset']}' için eğitim veya test seti boş döndü (create_dataset sonrası). Tahmin grafiği çizilemiyor.")
                            context['prediction_chart_json'] = json.dumps({
                                'data': [],
                                'layout': {
                                    'title': 'Tahmin Grafiği (Eğitim/Test Seti Boş)',
                                    'height': 500
                                }
                            })
                            context['simulation_message'] = "Simülasyon için eğitim veya test seti boş."
                            context['predicted_gain_loss'] = None
                            context['predicted_change_percent'] = None
                            context['risk_level'] = "Veri Yetersiz"
                            context['advice'] = "Farklı bir varlık deneyin veya daha fazla geçmiş veri ile deneyin."
                        else:
                            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                            # Model dosyasını silip yeniden eğitime zorlama mantığı burada
                            model_path_to_delete = os.path.join('analysis/models', f'{selected_asset_symbol}_lstm_model.h5')
                            if os.path.exists(model_path_to_delete):
                                os.remove(model_path_to_delete)
                                logger.info(f"Mevcut model dosyası silindi: {model_path_to_delete}. Yeni eğitim zorlanacak.")
                            
                            model = load_or_create_model(selected_asset_symbol, (time_step, 1))

                            if X_train.shape[0] > 0: 
                                logger.info(f"'{context['selected_asset']}' için model eğitiliyor (epochs: 100).")
                                model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=0) 
                                os.makedirs('analysis/models', exist_ok=True)
                                model.save(os.path.join('analysis/models', f'{selected_asset_symbol}_lstm_model.h5'))
                                logger.info(f"'{context['selected_asset']}' için model kaydedildi.")
                            else:
                                logger.warning(f"'{context['selected_asset']}' için boş X_train nedeniyle model eğitilemedi.")
                                context['simulation_message'] = "Model eğitilemedi, tahmin yapılamıyor."
                                context['predicted_gain_loss'] = None
                                context['predicted_change_percent'] = None
                                context['risk_level'] = "Model Hatası"
                                context['advice'] = "Daha fazla veri veya farklı bir varlık deneyin."
                            
                            # --- Tahmin ve Simülasyon Hesaplamaları ---
                            current_input_sequence = scaled_data[len(scaled_data) - time_step:].reshape(1, time_step, 1)
                            lst_output = []
                            for _ in range(context['prediction_days']):
                                predicted_scaled_value = model.predict(current_input_sequence, verbose=0)[0]
                                
                                temp_sequence = current_input_sequence[0, :, 0].tolist() 
                                temp_sequence.append(predicted_scaled_value[0]) 
                                
                                current_input_sequence = np.array(temp_sequence[-time_step:]).reshape(1, time_step, 1)
                                
                                lst_output.append(predicted_scaled_value.tolist())
                            
                            predicted_prices = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))
                            context['predicted_future_price'] = predicted_prices[-1][0] # Grafikte ve simülasyonda kullanılacak nihai tahmin

                            last_date = df_historical.index[-1]
                            prediction_dates = [last_date + timedelta(days=x) for x in range(1, context['prediction_days'] + 1)]

                            # Tahmin Grafiği Oluşturma
                            fig_prediction = go.Figure(data=[
                                go.Candlestick(
                                    x=df_historical.index,
                                    open=df_historical['Open'],
                                    high=df_historical['High'],
                                    low=df_historical['Low'],
                                    close=df_historical['Close'],
                                    name='Geçmiş Fiyat'
                                ),
                                go.Scatter(
                                    x=prediction_dates,
                                    y=predicted_prices.flatten(),
                                    mode='lines',
                                    name='Tahmin Edilen Fiyat',
                                    line=dict(color='orange')
                                )
                            ])
                            fig_prediction.update_layout(
                                title=f'{context["selected_asset"]} Fiyat Tahmini ({context["prediction_days"]} Gün)',
                                xaxis_title='Tarih',
                                yaxis_title='Fiyat',
                                xaxis_rangeslider_visible=False,
                                template='plotly_white',
                                height=400
                            )
                            context['prediction_chart_json'] = fig_prediction.to_json()

                            # --- Simülasyon Sonuçlarını Hazırlama ---
                            if context['current_asset_price'] is not None and context['predicted_future_price'] is not None:
                                context['predicted_change_percent'] = ((context['predicted_future_price'] - context['current_asset_price']) / context['current_asset_price']) * 100
                                context['predicted_gain_loss'] = (context['investment_amount'] / context['current_asset_price']) * (context['predicted_future_price'] - context['current_asset_price'])

                                # Risk seviyesi hesaplama (geçmiş günlük getirilerin standart sapması)
                                if len(df_historical['Close']) > 1:
                                    daily_returns = df_historical['Close'].pct_change().dropna()
                                    if not daily_returns.empty:
                                        volatility = daily_returns.std() * np.sqrt(252) # Yıllık volatilite
                                        if volatility > 0.05: # %5'ten büyükse yüksek riskli kabul edelim
                                            context['risk_level'] = "Yüksek"
                                        elif volatility > 0.02: # %2 ile %5 arası orta riskli
                                            context['risk_level'] = "Orta"
                                        else:
                                            context['risk_level'] = "Düşük"
                                    else:
                                        context['risk_level'] = "Veri Yetersiz (Risk)"
                                else:
                                    context['risk_level'] = "Veri Yetersiz (Risk)"

                                if context['predicted_change_percent'] > 0:
                                    context['simulation_message'] = f"Yatırımınızın {context['prediction_days']} gün içinde değer kazanması bekleniyor."
                                    if context['risk_level'] == "Yüksek":
                                        context['advice'] = "Yüksek kazanç potansiyeli var ancak yüksek risk içeriyor, dikkatli değerlendirin. Diversifikasyon düşünebilirsiniz."
                                    elif context['risk_level'] == "Orta":
                                        context['advice'] = "Orta riskli bir kazanç potansiyeli mevcut. Portföyünüzü çeşitlendirmeyi düşünebilirsiniz."
                                    else: 
                                        context['advice'] = "Düşük riskle kazanç potansiyeli yüksek görünüyor. Ancak her zaman piyasa koşullarını takip edin."
                                else:
                                    context['simulation_message'] = f"Yatırımınızın {context['prediction_days']} gün içinde değer kaybetmesi bekleniyor."
                                    if context['risk_level'] == "Yüksek":
                                        context['advice'] = "Yüksek kayıp riski taşıyor. Bu yatırımdan kaçınmanız veya çok dikkatli olmanız önerilir."
                                    elif context['risk_level'] == "Orta":
                                        context['advice'] = "Orta seviyede kayıp riski mevcut. Detaylı araştırma yapmadan yatırım yapmaktan kaçının."
                                    else: 
                                        context['advice'] = "Düşük riskli olsa da, beklenen düşüş nedeniyle bu yatırım riskli olabilir. Daha fazla araştırma yapmanız önerilir."
                                    
                                    if context['currency_type'] == "EUR":
                                        context['currency_symbol'] = "€"
                                    elif context['currency_type'] == "TRY":
                                        context['currency_symbol'] = "₺"
                            else:
                                context['simulation_message'] = "Simülasyon için gerekli fiyat bilgileri bulunamadı."
                                context['risk_level'] = "Hesaplanamadı"
                                context['advice'] = "Lütfen farklı bir varlık veya zaman aralığı deneyin."


            except Exception as e:
                logger.error(f"Tahmin oluşturulurken genel hata: {e}", exc_info=True)
                context['prediction_chart_json'] = json.dumps({ 
                    'data': [],
                    'layout': {
                        'title': 'Tahmin Grafiği (Hata Oluştu)',
                        'height': 500
                    }
                })
                context['simulation_message'] = f"Tahmin ve simülasyon sırasında bir hata oluştu: {e}"
                context['predicted_gain_loss'] = None
                context['predicted_change_percent'] = None
                context['risk_level'] = "Hata"
                context['advice'] = "Lütfen tekrar deneyin veya sistem yöneticisiyle iletişime geçin."

            else: # Bu else bloğu if not df_historical.empty and 'Close' in df_historical.columns: bloğuna ait
                logger.warning(f"'{context['selected_asset']}' için geçmiş veri boş veya 'Close' sütunu yok.")
                context['simulation_message'] = f"'{context['selected_asset']}' için geçmiş veri bulunamadı. Simülasyon yapılamıyor."
                context['predicted_gain_loss'] = None
                context['predicted_change_percent'] = None
                context['risk_level'] = "Veri Yok"
                context['advice'] = "Farklı bir varlık seçmeyi deneyin."
        else: # Bu else bloğu if selected_asset_symbol: bloğuna ait
            logger.warning(f"'{context['selected_asset']}' için sembol bulunamadı.")
            context['simulation_message'] = f"'{context['selected_asset']}' için geçerli sembol bulunamadı. Simülasyon yapılamıyor."
            context['predicted_gain_loss'] = None
            context['predicted_change_percent'] = None
            context['risk_level'] = "Geçersiz Varlık"
            context['advice'] = "Lütfen listeden geçerli bir varlık seçin."
        
    return render(request, 'analysis/dashboard.html', context)

# Haber Duyarlılığı Görünümü (news_sentiment_view)
# Bu fonksiyon, haber duyarlılığı analizini yapar ve sonuçları bir şablona aktarır.
def news_sentiment_view(request): # async def yerine def olarak değiştirildi
    # Bu fonksiyonun içeriği, haber API'lerinden veri çekme,
    # duyarlılık analizi yapma ve sonuçları bir şablona render etme adımlarını içerecektir.
    # Şimdilik basit bir placeholder döndürelim.
    
    # Örnek olarak, burada bir haber duyarlılığı analizi yapılabilir.
    # news_data = asyncio.run(fetch_news_data()) # data_fetcher.py'den gelen bir fonksiyon olabilir
    # sentiment_scores = analyze_sentiment(news_data) # Kendi duyarlılık analizi fonksiyonunuz
    
    context = {
        'sentiment_result': 'Haber duyarlılığı analizi sonuçları burada gösterilecektir.',
        # 'news_articles': news_articles,
        # 'sentiment_chart_json': sentiment_chart_json,
    }
    return render(request, 'analysis/news_sentiment.html', context)
