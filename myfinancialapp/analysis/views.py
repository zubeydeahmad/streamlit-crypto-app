# myfinancialapp/analysis/views.py

from django.shortcuts import render
from datetime import datetime, timedelta
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import logging

# data_fetcher.py dosyasından gerekli fonksiyonları ve VARLIK_BILGILERI'ni import edin
from .data_fetcher import fetch_all_popular_assets_and_save, get_historical_data_from_db_or_fetch, VARLIK_BILGILERI
from .models import HistoricalData, PopularAssetCache

logger = logging.getLogger(__name__)

# Ana sayfa görünümü
def home_view(request):
    # Varsayılan başlangıç ve bitiş tarihleri (örn: son 1 yıl)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365) # Son 1 yıl

    # Popüler varlıkları çek ve kaydet
    # Bu fonksiyon, cache mekanizması ile fiyatları güncel tutar
    popular_assets_df = fetch_all_popular_assets_and_save()
    popular_assets_data = popular_assets_df.to_dict('records')

    # Varlık seçimleri için veritabanından mevcut varlıkları al
    # Eğer henüz veritabanında varlık yoksa, boş bir liste ile devam et
    asset_choices = []
    try:
        # PopularAssetCache'teki tüm asset_name'leri alıp uniq sembollerle eşleştir
        all_popular_assets = PopularAssetCache.objects.all()
        for asset in all_popular_assets:
            # VARLIK_BILGILERI sabitinde sembole göre ismi bul
            # asset.asset_name aslında sembol (BTC, GC=F gibi) olmalı, değilse düzeltme gerekebilir.
            # Şu anki mantıkta PopularAssetCache'te asset_name sembolü tuttuğunuz varsayılıyor.
            for varlik_adi, bilgi in VARLIK_BILGILERI.items():
                if bilgi["sembol"] == asset.asset_name: # Bu satır doğru çalışıyorsa sorun yok, aksi takdirde asset.asset_name'i kontrol edin
                    asset_choices.append((asset.asset_name, varlik_adi))
                    break # Bulunca döngüden çık

        # Eğer PopularAssetCache boşsa veya asset_choices hala boşsa, varsayılan VARLIK_BILGILERI'ni kullan
        if not asset_choices and VARLIK_BILGILERI:
            for varlik_adi, bilgi in VARLIK_BILGILERI.items():
                asset_choices.append((bilgi["sembol"], varlik_adi))
        
        # Seçimler alfabetik sıraya göre sıralanabilir
        asset_choices = sorted(list(set(asset_choices)), key=lambda x: x[1]) # Tekrarlananları sil ve ada göre sırala

    except Exception as e:
        logger.error(f"Varlık seçimleri oluşturulurken hata: {e}", exc_info=True)
        # Hata durumunda varsayılan olarak VARLIK_BILGILERI'ni kullan
        asset_choices = [(info["sembol"], name) for name, info in VARLIK_BILGILERI.items()]
        asset_choices = sorted(asset_choices, key=lambda x: x[1])


    selected_asset_symbol = "BTC" # Varsayılan Bitcoin
    prediction_days = 30 # Varsayılan tahmin gün sayısı
    historical_data_json = None
    popular_assets_data_json = json.dumps(popular_assets_data) # Popüler varlıklar JSON'a dönüştürüldü

    # POST isteği geldiğinde form verilerini işle
    if request.method == 'POST':
        selected_asset_symbol = request.POST.get('asset_symbol', "BTC")
        try:
            prediction_days = int(request.POST.get('prediction_days', 30))
        except ValueError:
            prediction_days = 30 # Geçersiz değer durumunda varsayılan

    # Seçilen varlığın geçmiş verisini çek (DB'den veya API'den)
    df_historical = get_historical_data_from_db_or_fetch(selected_asset_symbol, start_date, end_date)

    historical_chart_div = ""
    if not df_historical.empty:
        df_historical = df_historical.reset_index().rename(columns={'index': 'Date'})
        
        # Plotly ile etkileşimli grafik oluştur
        fig = go.Figure()

        # Open, High, Low, Close (OHLC) grafiği ekle
        fig.add_trace(go.Candlestick(x=df_historical['Date'],
                        open=df_historical['Open'],
                        high=df_historical['High'],
                        low=df_historical['Low'],
                        close=df_historical['Close'],
                        name='OHLC'))

        fig.update_layout(
            title={
                'text': f"{selected_asset_symbol} Geçmiş Fiyat Grafiği",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=50, b=20),
            # Arka plan renklerini tema ile uyumlu hale getir
            paper_bgcolor='#F8F9FA', # Açık gri tonu
            plot_bgcolor='#FFFFFF', # Beyaz grafik alanı
            font=dict(color='#343A40') # Koyu gri yazı rengi
        )
        
        # Grafiği HTML div'e dönüştür
        historical_chart_div = plot(fig, output_type='div', include_plotlyjs=False)
    else:
        logger.warning(f"{selected_asset_symbol} için geçmiş veri boş veya alınamadı.")

    context = {
        'asset_choices': asset_choices,
        'selected_asset': selected_asset_symbol,
        'prediction_days': prediction_days,
        'historical_chart_div': historical_chart_div,
        'popular_assets_data_json': popular_assets_data_json, # JSON formatında gönder
    }
    return render(request, 'analysis/index.html', context)
