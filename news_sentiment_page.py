# 03_news_sentiment_page.py
# (Haberler ve Duyarlılık Analizi)
# Bu dosya, haberleri web'den çekip duygu analizi yapan ayrı bir Streamlit sayfası olarak tasarlanmıştır. Doğrudan streamlit run 03_news_sentiment_page.py komutuyla çalıştırılabilir.
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import logging

# Loglama yapılandırması
logging.basicConfig(filename='crypto_app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Haber ve Duyarlılık Analizi", layout="wide")

# NLTK'nın VADER sözlüğünü ve Punkt tokenizer'ı indirin (ilk çalıştırmada bir kere yapılır)
# Ayrıca, indirme işlemi sırasında oluşabilecek diğer hataları da yakalıyoruz.
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    st.info("VADER lexicon bulunamadı, indiriliyor...")
    try:
        nltk.download('vader_lexicon')
    except Exception as e:
        st.error(f"VADER lexicon indirilirken bir hata oluştu: {e}. Lütfen internet bağlantınızı kontrol edin.")
        logger.error(f"VADER lexicon indirme hatası: {e}")
except Exception as e:
    st.error(f"VADER lexicon kontrol edilirken beklenmeyen bir hata oluştu: {e}")
    logger.error(f"VADER lexicon kontrol hatası: {e}")

try:
    nltk.data.find('tokenizers/punkt.zip')
except LookupError:
    st.info("Punkt tokenizer bulunamadı, indiriliyor...")
    try:
        nltk.download('punkt')
    except Exception as e:
        st.error(f"Punkt tokenizer indirilirken bir hata oluştu: {e}. Lütfen internet bağlantınızı kontrol edin.")
        logger.error(f"Punkt tokenizer indirme hatası: {e}")
except Exception as e:
    st.error(f"Punkt tokenizer kontrol edilirken beklenmeyen bir hata oluştu: {e}")
    logger.error(f"Punkt tokenizer kontrol hatası: {e}")


def get_news_headlines(url):
    """Belirli bir URL'den haber başlıklarını çekmeye çalışır."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = []
        
        # Bu kısım her web sitesi için özelleştirilmelidir!
        # CoinDesk'in HTML yapısı sıkça değişebilir, bu seçiciler güncel olmayabilir.
        # Genellikle başlıklar h2, h3 etiketleri içinde veya belirli class'lara sahip div'ler içinde yer alır.
        
        # Olası CoinDesk başlık seçicileri (güncel olanı kontrol etmeniz gerekebilir):
        for h2_tag in soup.find_all('h2', class_='css-1a6v75g'):
            a_tag = h2_tag.find('a')
            if a_tag and a_tag.text:
                headlines.append(a_tag.text.strip())
        
        if not headlines:
            for div_tag in soup.find_all('div', class_='text-xl'):
                a_tag = div_tag.find('a')
                if a_tag and a_tag.text:
                    headlines.append(a_tag.text.strip())
        
        if not headlines:
            for title_tag in soup.find_all(['h1', 'h2', 'h3', 'h4']):
                if title_tag.find('a') and title_tag.find('a').text:
                    headlines.append(title_tag.find('a').text.strip())
                elif title_tag.text and len(title_tag.text.strip()) > 10:
                    headlines.append(title_tag.text.strip())

        logger.info(f"{len(headlines)} haber başlığı çekildi.")
        return headlines
    except requests.exceptions.RequestException as e:
        st.error(f"Haber çekilirken ağ hatası oluştu: {e}. Lütfen URL'yi ve internet bağlantınızı kontrol edin.")
        logger.error(f"Haber çekilirken ağ hatası: {e}")
        return []
    except Exception as e:
        st.error(f"Haber çekilirken beklenmeyen bir hata oluştu: {e}")
        logger.error(f"Haber çekilirken beklenmeyen hata: {e}")
        return []

def analyze_sentiment(text_list):
    """Metin listesi için duygu analizi yapar ve ortalama bileşik skoru döndürür."""
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for text in text_list:
        vs = analyzer.polarity_scores(text)
        sentiments.append(vs['compound']) # Bileşik skor (-1.0 ile +1.0 arası)
    
    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
        logger.info(f"Duygu analizi tamamlandı. Ortalama skor: {avg_sentiment:.2f}")
        return avg_sentiment
    logger.info("Duygu analizi için metin bulunamadı.")
    return 0.0 # Haber yoksa nötr


# --- Streamlit Uygulaması ---
st.title("📰 Kripto Para Haberleri ve Duyarlılık Analizi")
st.markdown("Bu bölüm, belirlediğiniz bir kaynaktan (varsayılan: CoinDesk) haber başlıklarını çekerek piyasa duyarlılığını analiz eder.")

coindesk_url = st.text_input("Haber Kaynağı URL'si (örneğin CoinDesk):", "https://www.coindesk.com/")

if st.button("Haberleri Çek ve Analiz Et"):
    with st.spinner("Haberler çekiliyor ve analiz ediliyor..."):
        news_headlines = get_news_headlines(coindesk_url)
        if news_headlines:
            st.subheader("Çekilen Haber Başlıkları:")
            for i, headline in enumerate(news_headlines[:10]): # İlk 10 başlığı göster
                st.write(f"- {headline}")
            
            avg_sentiment = analyze_sentiment(news_headlines)
            
            st.subheader("Duygu Analizi Sonucu:")
            st.write(f"**Ortalama Duygu Skoru (VADER):** {avg_sentiment:.2f} (1.0 = çok pozitif, -1.0 = çok negatif)")

            if avg_sentiment > 0.1:
                st.success("Genel haber duyarlılığı pozitif görünüyor. Bu durum piyasa için olumlu olabilir.")
            elif avg_sentiment < -0.1:
                st.error("Genel haber duyarlılığı negatif görünüyor. Bu durum piyasa için olumsuz olabilir.")
            else:
                st.info("Genel haber duyarlılığı nötr görünüyor.")
        else:
            st.warning("Haber başlıkları çekilemedi veya site yapısı değişmiş olabilir. Lütfen URL'yi ve sitenin HTML yapısını kontrol edin.")

st.markdown("---")
st.caption("Not: Web kazıma kodları, hedef sitenin HTML yapısı değiştiğinde çalışmayabilir. Profesyonel uygulamalar için genellikle haber API'leri tercih edilir.")