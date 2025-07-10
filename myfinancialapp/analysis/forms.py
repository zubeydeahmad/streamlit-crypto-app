# myfinancialapp/analysis/forms.py

from django import forms
from django.contrib.auth.forms import UserCreationForm # Yeni eklendi
from django.contrib.auth.models import User # Yeni eklendi
from .models import PopularAssetCache # Popüler varlıkları seçim listesi için kullanacağız

class AssetSelectionForm(forms.Form):
    """
    Kullanıcının finansal varlık seçimi yapmasını sağlayan Django formu.
    """
    asset = forms.ChoiceField(
        label="Varlık Seçin", # Formda görünecek etiket
        choices=[],          # Seçenekler dinamik olarak views.py'de doldurulacak
        widget=forms.Select(attrs={'class': 'form-select rounded-md shadow-sm border-gray-300 focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50 p-2 w-full'})
    )
    
    prediction_days = forms.IntegerField(
        label="Tahmin Gün Sayısı (1-3)",
        min_value=1,
        max_value=3,
        initial=1, # Varsayılan değer
        widget=forms.NumberInput(attrs={'class': 'form-input rounded-md shadow-sm border-gray-300 focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50 p-2 w-full', 'min': '1', 'max': '3'})
    )

    def __init__(self, *args, **kwargs):
        # choices dinamik olarak dışarıdan alınacak
        choices = kwargs.pop('choices', []) 
        super().__init__(*args, **kwargs)
        if choices:
            self.fields['asset'].choices = choices

# Yeni eklenecek kısım: Kullanıcı kayıt formu
class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = User
        fields = UserCreationForm.Meta.fields + ('email',) # E-posta alanını da ekleyelim
