# analysis/urls.py
from django.urls import path
from . import views

# Bu, Django'nun URL'lerinizi tersine çevirirken (örneğin template içinde) kullanacağı isim alanıdır.
# Aynı isimde iki url pattern olmaması için önemlidir.
app_name = 'analysis' 

urlpatterns = [
    # Boş bir yol ('') tanımlıyoruz, bu da 'analysis/' adresine gidildiğinde
    # views.py dosyasındaki home_view fonksiyonunun çalışacağını belirtir.
    # name='home' bu URL modeline bir isim verir, böylece template veya başka bir view'den
    # bu URL'ye kolayca referans verebiliriz (örneğin {% url 'analysis:home' %}).
    path('', views.home_view, name='home'), 
    # Buraya daha sonra diğer analiz ilgili URL'leriniz gelecek.
]