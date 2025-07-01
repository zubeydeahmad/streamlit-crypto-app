from django.db import models

#Create your models here.
class HistoricalData(models.Model):
    # Varlık sembolü (örn. "GC=F", "BTC")
    asset_symbol = models.CharField(max_length=50, unique=True)
    
    # DataFrame'i JSON string olarak saklayacağız.
    # Django 3.1+ ile JSONField daha uygun olabilir, ancak SQLite'da JSON desteği sınırlı olabilir.
    # Alternatif olarak, eğer JSONField hata verirse, CharField kullanıp manuel olarak JSON.dumps/loads yapabilirsiniz.
    data_json = models.JSONField() 
    
    # Verinin son güncellenme zamanı
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.asset_symbol} - {self.last_updated.strftime('%Y-%m-%d %H:%M')}"

    class Meta:
        verbose_name_plural = "Historical Data" # Admin arayüzünde daha okunaklı isim

class PopularAssetCache(models.Model):
    # Varlık adı (örn. "Altın", "Bitcoin")
    asset_name = models.CharField(max_length=100, unique=True)
    
    # Fiyat ve değişim yüzdesi
    price = models.FloatField(null=True, blank=True)
    change_percent = models.FloatField(null=True, blank=True)
    
    # Önbelleğin son güncellenme zamanı
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.asset_name

    class Meta:
        verbose_name_plural = "Popular Asset Cache" # Admin arayüzünde daha okunaklı isim