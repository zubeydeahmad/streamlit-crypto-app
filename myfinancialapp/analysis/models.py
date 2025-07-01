# myfinancialapp/analysis/models.py

from django.db import models

class HistoricalData(models.Model):
    # Varlık sembolü (örn. "GC=F" - altın vadelileri, "BTC" - Bitcoin)
    # unique=True: Her sembol için sadece bir kayıt olmasını sağlar.
    asset_symbol = models.CharField(max_length=50, unique=True)
    
    # Geçmiş veriyi (Pandas DataFrame) JSON formatında saklarız.
    # JSONField, Django'nun JSON verilerini yerel Python nesneleri olarak yönetmesini sağlar.
    data_json = models.JSONField() 
    
    # Kayıtın ne zaman güncellendiğini otomatik olarak ayarlar.
    last_updated = models.DateTimeField(auto_now=True)

    # Admin panelinde objeleri daha anlamlı göstermek için
    def __str__(self):
        return f"{self.asset_symbol} - {self.last_updated.strftime('%Y-%m-%d %H:%M')}"

    class Meta:
        # Admin panelinde modelin isminin çoğul halini daha okunaklı yapar.
        verbose_name_plural = "Historical Data" 

class PopularAssetCache(models.Model):
    # Popüler varlığın adı (örn. "Altın", "Bitcoin", "Dolar/Türk Lirası")
    asset_name = models.CharField(max_length=100, unique=True)
    
    # Varlığın anlık fiyatı
    price = models.FloatField(null=True, blank=True)
    
    # Varlığın değişim yüzdesi
    change_percent = models.FloatField(null=True, blank=True)
    
    # Önbellek kaydının ne zaman güncellendiğini otomatik olarak ayarlar.
    last_updated = models.DateTimeField(auto_now=True)

    # Admin panelinde objeleri daha anlamlı göstermek için
    def __str__(self):
        return self.asset_name

    class Meta:
        verbose_name_plural = "Popular Asset Cache"