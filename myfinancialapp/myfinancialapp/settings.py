# myfinancialapp/myfinancialapp/settings.py

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-%c*13x%2kkhj)drg+(&!f#tv5algx=^tmema%4i(p&ueecuayg' # Güvenli bir anahtar kullanın

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.humanize', # Genellikle sayı formatlama için kullanılır

    # 2FA (Two-Factor Authentication) için gerekli uygulamalar
    #'django_otp',
    #'django_otp.plugins.otp_totp', # Google Authenticator gibi TOTP cihazları için
    #'two_factor', # Ana iki faktörlü kimlik doğrulama uygulaması
    #'qr_code', # QR kodları oluşturmak için (2FA kurulumunda kullanılır)
    #'two_factor.plugins.phonenumber', # Telefon numarası tabanlı 2FA için (SMS vs.)

    'analysis', # Kendi özel uygulamanız
    # 'news_sentiment', # Eğer ayrı bir uygulama olarak news_sentiment varsa ve kullanılacaksa aktif edin
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',

    # 2FA için OTP middleware'i (kimlik doğrulama middleware'inden sonra gelmeli)
    #'django_otp.middleware.OTPMiddleware', 
]


ROOT_URLCONF = 'myfinancialapp.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'], # Proje seviyesi şablonlar için
        'APP_DIRS': True, # Uygulama seviyesi şablonlar için (analysis/templates gibi)
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'myfinancialapp.wsgi.application'


# Database
# https://docs.djangoproject.com/en/5.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.0/topics/i18n/

LANGUAGE_CODE = 'tr-tr' 

TIME_ZONE = 'Europe/Istanbul' 

USE_I18N = True 

USE_TZ = True 


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.0/howto/static-files/

STATIC_URL = 'static/'

# Django'nun statik dosyaları nerede arayacağını belirtir.
# Projenizin kök dizininde (manage.py ile aynı yerde) 'static' adında bir klasör oluşturun.
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

# Canlı sunucuya dağıtım yaparken, tüm statik dosyaların toplanacağı yer.
# Geliştirme aşamasında zorunlu değil, ancak gelecekte dağıtım için önemlidir.
# STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')


# API Anahtarları ve Diğer Uygulama Sabitleri
# decouple kaldırıldığı için doğrudan atama yapıyoruz (güvenlik için üretimde farklı bir yöntem düşünülmeli)
COINAPI_API_KEY = "f970d607-417d-4767-a532-39c637b4edaa" 
FIXER_API_KEY = "7e0984ecf4a9866e00e32f9cb4fef74b"
NEWS_API_KEY = "151bb830586e47879edc74544b713f88"

# Modelleriniz ve özellik mühendisliğiniz için kullandığınız sabitler
FEATURE_LAG = 7
TARGET_LAG = 1
VOLATILITY_THRESHOLD = 0.005 
TREND_FLATNESS_THRESHOLD = 0.0001
STABILITY_WINDOW = 20


# Default primary key field type
# https://docs.djangoproject.com/en/5.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Kullanıcı giriş yaptıktan sonra yönlendirilecek URL
# 2FA ile giriş yapıldıktan sonra iki faktörlü doğrulama sayfasına yönlendirilecektir.
LOGIN_REDIRECT_URL = '/analysis/' 

# Kullanıcı giriş yapması gerektiğinde yönlendirilecek URL
# 2FA için varsayılan giriş URL'sini kullanıyoruz
LOGIN_URL = '/accounts/login/'

# Kullanıcı çıkış yaptıktan sonra yönlendirilecek URL
LOGOUT_REDIRECT_URL = '/' 

# 2FA için ek ayarlar
#TWO_FACTOR_FORMS = {
#    'login': 'two_factor.forms.AuthDeviceForm',
#    'setup': 'two_factor.forms.TOTPDeviceForm',
#    'setup_complete': 'two_factor.forms.TOTPDeviceForm',
#}
# İsteğe bağlı: 2FA'yı tüm kullanıcılar için zorunlu kılmak isterseniz True yapın.
TWO_FACTOR_REQUIRED = False 

# E-posta ayarları (Şifre sıfırlama ve 2FA kurtarma kodları için gereklidir)
# Geliştirme ortamında e-postaları konsola yazdırmak için:
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
# Gerçek bir projede SMTP ayarları buraya gelecektir:
# EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
# EMAIL_HOST = 'smtp.example.com'
# EMAIL_PORT = 587
# EMAIL_USE_TLS = True
# EMAIL_HOST_USER = 'your_email@example.com'
# EMAIL_HOST_PASSWORD = 'your_email_password'