"""
URL configuration for myfinancialapp project.

The `urlpatterns` list routes URLs to views. For more information please see:
https://docs.djangoproject.com/en/5.2/topics/http-urls/

Examples:
Function views
  1. Add an import:  from my_app import views
  2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
  1. Add an import:  from other_app.views import Home
  2. Add a URL to urlpatterns:  path('blog/', Home.as_view(), name='home')
Including another URLconf
  1. Import the include() function: from django.urls import include, path
  2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include # <-- 'include' fonksiyonu buraya eklendi

urlpatterns = [
    path('admin/', admin.site.urls),
    # analysis uygulamasına ait URL'leri dahil ediyoruz
    # Kullanıcılar artık http://127.0.0.1:8000/analysis/ adresinden erişebilecekler.
    path('analysis/', include('analysis.urls')), # <-- Bu satır eklendi
    
    # news_sentiment uygulamasına ait URL'leri dahil ediyoruz
    # Eğer news_sentiment uygulamasını kullanacaksanız bu satırı da ekleyin.
    # Ancak önce news_sentiment/urls.py dosyasını oluşturmanız gerektiğini unutmayın.
    #path('news_sentiment/', include('news_sentiment.urls')), # <-- Bu satır eklendi
]

