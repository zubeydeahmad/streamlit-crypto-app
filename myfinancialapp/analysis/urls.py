# analysis/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='dashboard'), 
    path('news_sentiment/', views.news_sentiment_view, name='news_sentiment'), # Bu satır aktif olmalı
    path('register/', views.RegisterView.as_view(), name='register'),
]