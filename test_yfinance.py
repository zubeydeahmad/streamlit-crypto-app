# test_yfinance.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

try:
    print("yfinance kütüphanesinin temel veri çekme fonksiyonelliği test ediliyor...")
    
    # Test için yaygın kullanılan semboller:
    # AAPL (Apple Hissesi), GC=F (Altın Vadelileri), BTC-USD (Bitcoin/USD)
    test_symbols = ["AAPL", "GC=F", "BTC-USD"]
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5) # Son 5 günün verisi
    
    for symbol in test_symbols:
        print(f"\nSembol: {symbol} için veri çekiliyor ({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})...")
        # auto_adjust=True ile FutureWarning'ı bastırıyoruz ve düzeltilmiş fiyatları alıyoruz.
        data = yf.download(symbol, start=start_date, end=end_date, interval="1d", progress=False, actions=False, auto_adjust=True)
        
        if not data.empty:
            print(f"Başarılı! {symbol} için {len(data)} satır veri çekildi. İlk 5 satır:")
            print(data.head())
        else:
            print(f"HATA: {symbol} için veri çekilemedi. DataFrame boş döndü.")
            
except Exception as e:
    print(f"\nGenel bir hata oluştu: {e}")
    print("Lütfen yfinance kütüphanesinin doğru yüklendiğinden ve internet bağlantınızın olduğundan emin olun.")
    print("Ayrıca, proxy veya güvenlik duvarı ayarlarınızı kontrol edin.")

print("\nTest tamamlandı.")
