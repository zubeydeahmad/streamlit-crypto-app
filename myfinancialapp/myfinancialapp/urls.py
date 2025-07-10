# myfinancialapp/myfinancialapp/urls.py

from django.contrib import admin
from django.urls import path, include
#from two_factor.urls import urlpatterns as tf_urls 
#from two_factor.views import LoginView # SADECE LoginView'i import edin


urlpatterns = [
    path('admin/', admin.site.urls),

    # analysis uygulamasının tüm URL'lerini 'analysis_app' namespace'i altında dahil ediyoruz
    path('analysis/', include(('analysis.urls', 'analysis'), namespace='analysis_app')), 

    # Django'nun dahili kimlik doğrulama URL'lerini ekliyoruz
    path('accounts/', include('django.contrib.auth.urls')), 
]