# myfinancialapp/analysis/models.py

from django.db import models

class HistoricalData(models.Model):
    asset_symbol = models.CharField(max_length=50, unique=True)
    data_json = models.JSONField() 
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.asset_symbol} - {self.last_updated.strftime('%Y-%m-%d %H:%M')}"

    class Meta:
        verbose_name_plural = "Historical Data" 

class PopularAssetCache(models.Model):
    asset_name = models.CharField(max_length=100, unique=True)
    price = models.FloatField(null=True, blank=True)
    change_percent = models.FloatField(null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.asset_name

    class Meta:
        verbose_name_plural = "Popular Asset Cache"
