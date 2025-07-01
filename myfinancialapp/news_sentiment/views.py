# analysis/views.py
from django.shortcuts import render # render fonksiyonunu import ediyoruz

def home_view(request):
    """
    Finansal analiz uygulamasının ana sayfası görünümü.
    Bu fonksiyon, web isteğini alır ve bir HTML şablonu döndürür.
    """
    context = {
        # Şablona gönderilecek verileri bir sözlük içinde tanımlıyoruz.
        # dashboard.html içinde {{ page_title }} olarak erişilecek.
        'page_title': "Finansal Uygulamama Hoş Geldiniz!"
    }
    # 'analysis/dashboard.html' şablonunu render ediyoruz.
    # Django, settings.py'deki TEMPLATES ayarları sayesinde bu şablonu bulacaktır.
    return render(request, 'analysis/dashboard.html', context)

# İleride buraya başka görünümler (views) ekleyeceksiniz, örneğin:
# def asset_detail_view(request, asset_symbol):
#     # ... varlık detay sayfasının mantığı
#     return render(request, 'analysis/asset_detail.html', {'symbol': asset_symbol})