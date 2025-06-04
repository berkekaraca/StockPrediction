from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_mail import Mail, Message
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
import sqlite3
import os
import re
import shutil
from datetime import datetime, timedelta
from flask_bcrypt import Bcrypt
from flask_session import Session
import uuid
import requests
from cachetools import TTLCache, cached
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import nltk
from nltk.corpus import stopwords
from TurkishStemmer import TurkishStemmer
import feedparser
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, precision_score, recall_score
from dotenv import load_dotenv  # .env dosyasını okumak için

# .env dosyasını yükle
load_dotenv()

nltk.download('stopwords')
turkish_stopwords = set(stopwords.words('turkish'))
stemmer = TurkishStemmer()

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
Session(app)
bcrypt = Bcrypt(app)
mail = Mail(app)



classifier = pipeline('sentiment-analysis', model='savasy/bert-base-turkish-sentiment-cased')


bist100_list = [
    "AKBNK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS",
    "GUBRF.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KRDMD.IS", "PETKM.IS", "PGSUS.IS",
    "SAHOL.IS", "SISE.IS", "SASA.IS", "THYAO.IS", "TKFEN.IS", "TUPRS.IS", "VAKBN.IS", "YKBNK.IS"
]


end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=90)).strftime('%Y-%m-%d')


ticker_to_company = {
    "AKBNK.IS": "Akbank",
    "ASELS.IS": "Aselsan",
    "BIMAS.IS": "BİM Birleşik Mağazalar",
    "EKGYO.IS": "Emlak Konut GYO",
    "EREGL.IS": "Ereğli Demir Çelik",
    "FROTO.IS": "Ford Otosan",
    "GARAN.IS": "Garanti BBVA",
    "GUBRF.IS": "Gübre Fabrikaları",
    "HEKTS.IS": "Hektaş Ticaret",
    "ISCTR.IS": "İş Bankası",
    "KCHOL.IS": "Koç Holding",
    "KRDMD.IS": "Kardemir",
    "PETKM.IS": "Petkim",
    "PGSUS.IS": "Pegasus Hava Yolları",
    "SAHOL.IS": "Sabancı Holding",
    "SISE.IS": "Şişe Cam",
    "SASA.IS": "Sasa Polyester",
    "THYAO.IS": "Türk Hava Yolları",
    "TKFEN.IS": "Tekfen Holding",
    "TUPRS.IS": "Tüpraş",
    "VAKBN.IS": "Vakıfbank",
    "YKBNK.IS": "Yapı Kredi"
}


ticker_to_sectors = {
    "AKBNK.IS": ["bankacılık", "banka", "kredi", "faiz", "mevduat"],
    "ASELS.IS": ["savunma", "teknoloji", "elektronik"],
    "BIMAS.IS": ["perakende", "market", "toptan"],
    "EKGYO.IS": ["emlak", "gayrimenkul", "konut"],
    "EREGL.IS": ["çelik", "demir", "metal"],
    "FROTO.IS": ["otomotiv", "ford", "araç"],
    "GARAN.IS": ["bankacılık", "banka", "kredi", "faiz", "mevduat"],
    "GUBRF.IS": ["gübre", "tarım", "kimya"],
    "HEKTS.IS": ["tarım", "ilaç", "kimya"],
    "ISCTR.IS": ["bankacılık", "banka", "kredi", "faiz", "mevduat"],
    "KCHOL.IS": ["holding", "yatırım", "sanayi"],
    "KRDMD.IS": ["çelik", "demir", "metal"],
    "PETKM.IS": ["petrokimya", "kimya", "plastik"],
    "PGSUS.IS": ["havacılık", "uçak", "havayolu", "sefer", "bilet"],
    "SAHOL.IS": ["holding", "yatırım", "sanayi"],
    "SISE.IS": ["cam", "üretim", "sanayi"],
    "SASA.IS": ["polyester", "tekstil", "kimya"],
    "THYAO.IS": ["havacılık", "uçak", "havayolu", "sefer", "bilet"],
    "TKFEN.IS": ["inşaat", "mühendislik", "enerji"],
    "TUPRS.IS": ["petrol", "rafineri", "enerji"],
    "VAKBN.IS": ["bankacılık", "banka", "kredi", "faiz", "mevduat"],
    "YKBNK.IS": ["bankacılık", "banka", "kredi", "faiz", "mevduat"]
}


ticker_to_preference = {
    "AKBNK.IS": "uzun vadeli",
    "ASELS.IS": "kısa vadeli",
    "BIMAS.IS": "uzun vadeli",
    "EKGYO.IS": "kısa vadeli",
    "EREGL.IS": "uzun vadeli",
    "FROTO.IS": "uzun vadeli",
    "GARAN.IS": "hem kısa vadeli hem uzun vadeli tercih edilebilir",
    "GUBRF.IS": "kısa vadeli",
    "HEKTS.IS": "kısa vadeli",
    "ISCTR.IS": "hem kısa vadeli hem uzun vadeli tercih edilebilir",
    "KCHOL.IS": "hem kısa vadeli hem uzun vadeli tercih edilebilir",
    "KRDMD.IS": "kısa vadeli",
    "PETKM.IS": "kısa vadeli",
    "PGSUS.IS": "hem kısa vadeli hem uzun vadeli tercih edilebilir",
    "SAHOL.IS": "uzun vadeli",
    "SISE.IS": "hem kısa vadeli hem uzun vadeli tercih edilebilir",
    "SASA.IS": "kısa vadeli",
    "THYAO.IS": "uzun vadeli",
    "TKFEN.IS": "uzun vadeli",
    "TUPRS.IS": "hem kısa vadeli hem uzun vadeli tercih edilebilir",
    "VAKBN.IS": "hem kısa vadeli hem uzun vadeli tercih edilebilir",
    "YKBNK.IS": "kısa vadeli"
}


cache = TTLCache(maxsize=100, ttl=3600)
prediction_cache = TTLCache(maxsize=20, ttl=1800)



def get_preference_details(ticker):
    preference = ticker_to_preference.get(ticker, "")
    if "hem kısa vadeli hem uzun vadeli" in preference:
        return {"uzun_vadeli": "Evet", "kisa_vadeli": "Evet"}
    elif "uzun vadeli" in preference:
        return {"uzun_vadeli": "Evet", "kisa_vadeli": "Hayır"}
    elif "kısa vadeli" in preference:
        return {"uzun_vadeli": "Hayır", "kisa_vadeli": "Evet"}
    else:
        return {"uzun_vadeli": "Bilgi Yok", "kisa_vadeli": "Bilgi Yok"}



def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        is_verified INTEGER DEFAULT 0,
        verification_token TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        ticker TEXT,
        prediction_date TEXT,
        price_1_day REAL,
        price_5_days REAL,
        price_1_month REAL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS portfolios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        ticker TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    conn.commit()
    conn.close()


init_db()



def validate_password(password):
    if len(password) < 8:
        return False, "Şifre en az 8 karakter olmalıdır."
    if not re.search(r"[A-Z]", password):
        return False, "Şifre en az bir büyük harf içermelidir."
    if not re.search(r"[a-z]", password):
        return False, "Şifre en az bir küçük harf içermelidir."
    if not re.search(r"[0-9]", password):
        return False, "Şifre en az bir rakam içermelidir."
    return True, ""



def backup_database():
    try:
        shutil.copyfile('database.db', 'database_backup.db')
    except Exception as e:
        print(f"Veritabanı yedekleme hatası: {e}")



@cached(cache)
def load_data(ticker):
    ticker = ticker.replace('$', '')
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval='1d', auto_adjust=True)
        if data.empty:
            print(f"{ticker}: Günlük veri alınamadı, saatlik veriye geçiliyor.")
            temp_start = (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d')
            data = stock.history(start=temp_start, end=end_date, interval='1h', auto_adjust=True)
        if data.empty:
            print(f"{ticker}: Veri çekilemedi (boş veri).")
            return pd.DataFrame()
        print(
            f"{ticker}: Veri başarıyla çekildi, {len(data)} satır, interval: {'1d' if data.index[1] - data.index[0] >= timedelta(days=1) else '1h'}.")
        return data
    except Exception as e:
        print(f"{ticker}: Veri çekme hatası: {e}")
        return pd.DataFrame()


# Teknik göstergeleri hesaplama fonksiyonu
def compute_indicators(data, ticker):
    if data.empty:
        return data
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    data['MACD'] = macd
    data['Signal'] = signal

    bist100_data = load_data("XU100.IS")
    if not bist100_data.empty:
        bist100_data = bist100_data['Close'].reindex(data.index, method='ffill')
        data['BIST100_Close'] = bist100_data
    else:
        data['BIST100_Close'] = data['Close']

    return data


# Metin temizleme fonksiyonları
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def remove_stopwords(text):
    words = text.split()
    return ' '.join(word for word in words if word not in turkish_stopwords)


def lemmatize_text(text):
    words = text.split()
    return ' '.join(stemmer.stem(word) for word in words)



def analyze_sentiment_reason(text, company_name="Türk Hava Yolları"):
    positive_keywords = ['anlaşma', 'büyüme', 'ihracat', 'kâr', 'kar', 'kârlı', 'kârlılık', 'artış',
                         'başarı','yeni', 'özellik', 'aştı' 'yükseliş', 'bilanço', 'gelir', 'al', 'tavsiye', 'kazanç', 'başarılı',
                         'pozitif', 'iyileşme', 'yoğun ilgi', 'net alım', 'alım yaptı', 'indirim', 'kampanya',
                         'fırsat', 'rekor', 'lider', 'başlıyor', 'avantaj', 'özel fiyat', 'en', 'konfor', 'yüksek'
                                                                                                          'stratejik',
                         'imza', 'sıra', 'önemli', 'gelişim', 'teknoloji', 'inovasyon']
    negative_keywords = ['kayıp', 'mağlup', 'düşüş', 'zarar', 'kriz', 'iflas', 'sorun', 'pistten çıktı', 'kaza',
                         'iptal', 'olumsuz', 'ceza', 'yanlış', 'ters', 'dert', 'iptal edildi']
    special_negative_phrases = ['ters rüzgâr']

    company_variations = [company_name.lower(), "thy", "türk hava yolları"]

    sentences = text.split(',')
    relevant_text = text
    for sentence in sentences:
        sentence = sentence.strip()
        if any(variation in sentence.lower() for variation in company_variations):
            relevant_text = sentence
            break

    print(f"[BAĞLAM TESPİTİ] Analiz edilen metin: {relevant_text} (Orijinal: {text})")

    is_company_related = any(variation in relevant_text.lower() for variation in company_variations)

    if any(phrase in relevant_text.lower() for phrase in special_negative_phrases):
        sentiment = 'Negative'
        adjusted_score = -0.9
        reason = "Haber 'ters rüzgâr' gibi olumsuz bir ifade içeriyor, bu nedenle olumsuz olarak sınıflandırıldı."
        print(f"[ÖZEL KURAL - Ters Rüzgâr] {relevant_text} -> Sentiment: {sentiment}, Skor: {adjusted_score}")
        return sentiment, adjusted_score, reason

    cancel_keywords = ['iptal', 'iptal edildi']
    if any(keyword in relevant_text.lower() for keyword in cancel_keywords):
        sentiment = 'Negative'
        adjusted_score = -0.9
        reason = "Haber 'iptal' veya ilgili bir kelime içeriyor, bu nedenle olumsuz olarak sınıflandırıldı."
        print(f"[ÖZEL KURAL - İptal] {relevant_text} -> Sentiment: {sentiment}, Skor: {adjusted_score}")
        return sentiment, adjusted_score, reason

    penalty_keywords = ['ceza']
    if any(keyword in relevant_text.lower() for keyword in penalty_keywords):
        if is_company_related:
            sentiment = 'Negative'
            adjusted_score = -0.9
            reason = "Haber 'ceza' kelimesi içeriyor ve THY ile ilgili, bu nedenle olumsuz olarak sınıflandırıldı."
            print(
                f"[ÖZEL KURAL - Ceza (THY İle İlgili)] {relevant_text} -> Sentiment: {sentiment}, Skor: {adjusted_score}")
        else:
            sentiment = 'Neutral'
            adjusted_score = 0.0
            reason = "Haber 'ceza' kelimesi içeriyor ancak THY ile doğrudan ilgili değil, bu nedenle nötr olarak sınıflandırıldı."
            print(
                f"[ÖZEL KURAL - Ceza (THY İle İlgili Değil)] {relevant_text} -> Sentiment: {sentiment}, Skor: {adjusted_score}")
        return sentiment, adjusted_score, reason

    loss_keywords = ['zarar', 'kayıp']
    if any(keyword in relevant_text.lower() for keyword in loss_keywords):
        sentiment = 'Negative'
        adjusted_score = -0.9
        reason = "Haber 'zarar' veya 'kayıp' kelimesi içeriyor, bu nedenle olumsuz olarak sınıflandırıldı."
        print(f"[ÖZEL KURAL - Zarar] {relevant_text} -> Sentiment: {sentiment}, Skor: {adjusted_score}")
        return sentiment, adjusted_score, reason

    kar_keywords = ['kâr', 'kar', 'kârlı', 'kârlılık']
    if any(keyword in relevant_text.lower() for keyword in kar_keywords):
        sentiment = 'Positive'
        adjusted_score = 0.9
        reason = "Haber 'kâr' veya ilgili bir kelime içeriyor, bu nedenle olumlu olarak sınıflandırıldı."
        print(f"[ÖZEL KURAL - Kâr] {relevant_text} -> Sentiment: {sentiment}, Skor: {adjusted_score}")
        return sentiment, adjusted_score, reason

    investment_keywords = ['yoğun ilgi', 'net alım', 'alım yaptı']
    if any(keyword in relevant_text.lower() for keyword in investment_keywords):
        sentiment = 'Positive'
        adjusted_score = 0.9
        reason = "Haber 'yoğun ilgi', 'net alım' veya 'alım yaptı' gibi olumlu yatırım ifadeleri içeriyor."
        print(f"[ÖZEL KURAL - Yatırım] {relevant_text} -> Sentiment: {sentiment}, Skor: {adjusted_score}")
        return sentiment, adjusted_score, reason

    discount_keywords = ['indirim', 'kampanya', 'fırsat', 'özel fiyat', 'avantaj']
    if any(keyword in relevant_text.lower() for keyword in discount_keywords):
        sentiment = 'Positive'
        adjusted_score = 0.9
        reason = "Haber 'indirim', 'kampanya' veya 'fırsat' gibi olumlu ifadeler içeriyor."
        print(f"[ÖZEL KURAL - İndirim] {relevant_text} -> Sentiment: {sentiment}, Skor: {adjusted_score}")
        return sentiment, adjusted_score, reason

    result = classifier(relevant_text)[0]
    label = result['label']
    score = result['score']

    if label == 'positive':
        sentiment = 'Positive'
        adjusted_score = score * 0.3
    else:
        sentiment = 'Negative'
        adjusted_score = -score * 0.3

    print(f"[BERT Sonucu] {relevant_text} -> Label: {label}, Skor: {score}, İlk Adjusted Skor: {adjusted_score}")

    positive_weight = 0
    negative_weight = 0
    for keyword in positive_keywords:
        if keyword in relevant_text.lower():
            positive_weight += 0.4
            break
    for keyword in negative_keywords:
        if keyword in relevant_text.lower():
            negative_weight -= 0.6
            break

    adjusted_score += (positive_weight + negative_weight)
    adjusted_score = max(min(adjusted_score, 1.0), -1.0)

    if adjusted_score > 0.3:
        sentiment = 'Positive'
    elif adjusted_score < -0.3:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    reason = ""
    if sentiment == 'Positive':
        if positive_weight > 0:
            reason = "Haber içeriği olumlu anahtar kelimeler (örneğin, 'artış', 'kâr', 'indirim') içeriyor."
        else:
            reason = "Haber genel olarak olumlu bir tona sahip."
    elif sentiment == 'Negative':
        if negative_weight < 0:
            reason = "Haber içeriği olumsuz anahtar kelimeler (örneğin, 'düşüş', 'iptal', 'ceza', 'ters') içeriyor."
        else:
            reason = "Haber genel olarak olumsuz bir tona sahip."
    else:
        reason = "Haber ne güçlü bir olumlu ne de olumsuz ton içeriyor."

    print(f"[SONUÇ] {relevant_text} -> Sentiment: {sentiment}, Adjusted Skor: {adjusted_score}, Sebep: {reason}")

    return sentiment, adjusted_score, reason


# Duygu skoru hesaplama fonksiyonu
def get_sentiment_score(ticker):
    if ticker in cache:
        print(f"{ticker}: Önbellekten alındı, Önbellek içeriği: {cache[ticker]}")
        return cache[ticker]

    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    if not NEWS_API_KEY:
        print(f"{ticker}: NEWS_API_KEY eksik, haber çekimi yapılamadı.")
        return None, [], "API anahtarı eksik, haber çekimi yapılamadı."

    company_name = ticker_to_company.get(ticker, ticker.split('.')[0])
    ticker_short = ticker.split('.')[0]
    sectors = ticker_to_sectors.get(ticker, [])

    articles = []
    query = f'"{company_name}" OR "{ticker_short}"'
    url = f"https://newsapi.org/v2/everything?q={query}&language=tr&sortBy=publishedAt&apiKey={NEWS_API_KEY}&pageSize=50"
    headers = {
        'User-Agent': 'HisseTahminApp/1.0 (Server-Side Request)',
        'Accept': 'application/json'
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        print(f"{ticker}: News API isteği, Status: {response.status_code}, URL: {url}, Response: {response.text[:200]}")
        if response.status_code != 200:
            error_msg = f"News API hatası: Status {response.status_code}, Mesaj: {response.text}"
            print(error_msg)
        else:
            data = response.json()
            articles.extend(data.get('articles', []))
            print(f"{ticker}: NewsAPI'den {len(data.get('articles', []))} haber çekildi (sorgu: {query}).")
    except Exception as e:
        print(f"{ticker}: News API veri çekme hatası: {e}")

    rss_feeds = [
        "https://www.airporthaber2.com/rss",
        "https://www.airlinehaber.com/feed/",
        "https://www.kap.org.tr/tr/BildirimRss/Genel",
        "https://www.bloomberght.com/rss",
        "https://www.bloomberght.com/rss/borsa",
        "https://www.hurriyet.com.tr/rss/ekonomi",
        "https://www.milliyet.com.tr/rss/rssNew/ekonomiRss.xml",
        "https://www.sabah.com.tr/rss/ekonomi.xml",
        "https://www.dunya.com/feeds/news",
        "https://www.reuters.com/rss-feed/turkey-news-feed",
        "https://www.ekonomist.com.tr/rss",
        "https://www.para.com.tr/rss"
    ]

    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                articles.append({
                    'title': entry.get('title', ''),
                    'description': entry.get('summary', '') or entry.get('description', ''),
                    'url': entry.get('link', '#')
                })
            print(f"{ticker}: RSS feed ({feed_url}) üzerinden {len(feed.entries[:10])} haber çekildi.")
        except Exception as e:
            print(f"{ticker}: RSS feed ({feed_url}) çekme hatası: {e}")

    print(f"{ticker}: Toplam {len(articles)} haber çekildi (NewsAPI + RSS).")
    if not articles:
        print(f"{ticker}: Haber bulunamadı (sorgu: {query}).")
        return None, [], "İlgili haber bulunamadı, varsayılan nötr skor atandı."

    sentiments = []
    headlines_dict = {}
    irrelevant_keywords = ['nike', 'nso group', 'operación cataluña', 'catalunya', 'casus', 'spyware', 'whatsapp',
                           'meta', 'yayınları', 'tesadüf', 'festival', 'yarışma', 'polis', 'jandarma', 'kamuoyu']
    financial_keywords = ['hisse', 'kâr', 'kar', 'yatırım', 'bilanço', 'borsa', 'net alım', 'satış', 'piyasa', 'endeks']

    company_name_lower = company_name.lower()
    ticker_short_lower = ticker_short.lower()

    for article in articles:
        headline = article.get('title', '')
        description = article.get('description', '') or ''
        link = article.get('url', '#')
        text = f"{headline} {description}".lower()

        if any(keyword in text for keyword in irrelevant_keywords):
            print(
                f"{ticker}: Alakasız haber atlandı (anahtar kelime: {', '.join([k for k in irrelevant_keywords if k in text])}): {headline}")
            continue

        if not (company_name_lower in text or ticker_short_lower in text):
            print(f"{ticker}: Şirket adı veya ticker içermeyen haber atlandı: {headline}")
            continue

        if not any(keyword in text for keyword in financial_keywords):
            print(f"{ticker}: Finansal anahtar kelime içermeyen haber atlandı: {headline}")
            continue

        cleaned_text = clean_text(text)
        cleaned_text = remove_stopwords(cleaned_text)
        cleaned_text = lemmatize_text(cleaned_text)
        print(f"{ticker}: İşlenen metin: {cleaned_text}")

        if cleaned_text:
            sentiment, score, reason = analyze_sentiment_reason(cleaned_text, company_name=company_name)
            headlines_dict[headline] = (headline, score, link, sentiment, reason)
            sentiments.append(score)
            print(f"{ticker}: Analiz edilen haber: {headline}, Sentiment: {sentiment}, Skor: {score}, Sebep: {reason}")
            if len(headlines_dict) >= 10:
                break

    headlines = list(headlines_dict.values())

    if not headlines:
        print(f"{ticker}: İlgili haber bulunamadı (sorgu: {query}).")
        return None, [], "İlgili haber bulunamadı, varsayılan nötr skor atandı."

    avg_sentiment = np.mean(sentiments)
    print(
        f"{ticker}: {len(headlines)} ilgili haber bulundu, Başlıklar ve Skorlar: {[(h[0], h[1], h[3]) for h in headlines]}, Ortalama duygu skoru: {avg_sentiment}")
    cache[ticker] = (avg_sentiment, headlines, None)
    return avg_sentiment, headlines, None


# Tahmin nedenlerini analiz etme fonksiyonu (Güncellendi)
def analyze_reasons(data, sentiment_score, today_price, future_prices):
    reasons = {'1_day': '', '5_days': '', '1_month': ''}
    bist100_data = load_data("XU100.IS")
    market_trend = 'normal'
    if not bist100_data.empty:
        recent_bist100 = bist100_data['Close'][-5:].pct_change().mean()
        market_trend = 'yükseliyor' if recent_bist100 > 0 else 'düşüyor'
        print(f"BIST100 trendi: {market_trend}, Ortalama değişim: {recent_bist100:.4f}")

    rsi = data['RSI'].iloc[-1] if not data.empty else 50
    macd = data['MACD'].iloc[-1] if not data.empty else 0
    signal = data['Signal'].iloc[-1] if not data.empty else 0
    print(f"RSI: {rsi}, MACD: {macd}, Signal: {signal}")

    # Fiyat değişim oranlarını hesapla
    price_changes = [(price - today_price) / today_price * 100 for price in future_prices]
    print(
        f"Fiyat değişim oranları: 1 Gün: {price_changes[0]:.2f}%, 5 Gün: {price_changes[1]:.2f}%, 1 Ay: {price_changes[2]:.2f}%")

    for idx, (period, price, change) in enumerate(zip(['1_day', '5_days', '1_month'], future_prices, price_changes)):
        trend = 'yükseliş' if price > today_price else 'düşüş'
        reason = []

        if trend == 'yükseliş':
            if rsi < 30:
                reason.append("Hisse fiyatı çok düşmüştü, toparlanma beklenebilir.")
            elif 30 <= rsi < 50:
                reason.append("")

            if macd > signal:
                reason.append("Fiyat hareketleri, yakın zamanda yükseliş olabileceğini gösteriyor.")

            # Haber kontrolü ekleniyor
            if sentiment_score is not None:
                if sentiment_score > 0.3:
                    reason.append("Hakkındaki haberler oldukça olumlu, bu yükselişi destekliyor.")
                elif 0 < sentiment_score <= 0.3:
                    reason.append("Hakkındaki haberler hafif olumlu, bu yükselişe katkı sağlayabilir.")

            if market_trend == 'yükseliyor':
                reason.append("Genel borsa piyasası da yükselişte.")

            if period == '1_month' and sentiment_score is not None and 0 < sentiment_score <= 0.3:
                reason.append("Ancak uzun vadede artış sınırlı olabilir.")

        else:  # Düşüş trendi
            # Kısa vadeli (1 gün) için daha teknik odaklı nedenler
            if period == '1_day':
                if macd < signal:
                    reason.append("Fiyat hareketleri, yakın zamanda düşüş olabileceğini gösteriyor.")
                # Haber kontrolü ekleniyor
                if sentiment_score is not None:
                    if sentiment_score < -0.3:
                        reason.append("Hakkındaki haberler oldukça olumsuz, bu düşüşü tetikliyor.")
                    elif -0.3 <= sentiment_score < 0:
                        reason.append("Hakkındaki haberler hafif olumsuz, bu düşüşe katkı sağlayabilir.")
                if abs(change) < 0.5:
                    reason.append("Bu kısa vadeli bir teknik düzeltme olabilir.")

            # Orta vadeli (5 gün) için piyasa trendi ve haber etkisi
            elif period == '5_days':
                if macd < signal:
                    reason.append("Fiyat hareketleri, yakın zamanda düşüş olabileceğini gösteriyor.")
                if market_trend == 'düşüyor':
                    reason.append("Genel borsa piyasası da düşüşte.")
                # Haber kontrolü ekleniyor
                if sentiment_score is not None:
                    if sentiment_score < -0.3:
                        reason.append("Hakkındaki haberler oldukça olumsuz, bu düşüşü tetikliyor.")
                    elif -0.3 <= sentiment_score < 0:
                        reason.append("Hakkındaki haberler hafif olumsuz, bu düşüşe katkı sağlayabilir.")
                if 0.5 <= abs(change) <= 1.5:
                    reason.append("Orta vadeli bir düzeltme trendi görülüyor.")

            # Uzun vadeli (1 ay) için daha geniş kapsamlı nedenler
            elif period == '1_month':
                if rsi > 70:
                    reason.append("Hisse fiyatı çok yükseldi, uzun vadede düşüş gelebilir.")
                elif 50 < rsi <= 70:
                    reason.append("Hisse fiyatı yüksek seviyelerde, düşüş riski taşıyabilir.")
                elif 30 <= rsi < 50:
                    reason.append("Hisse toparlanma potansiyeline sahip ancak düşüş trendi baskın.")
                if market_trend == 'düşüyor':
                    reason.append("Genel borsa piyasası düşüşte, uzun vadede bu baskı devam edebilir.")
                # Haber kontrolü ekleniyor
                if sentiment_score is not None:
                    if sentiment_score < -0.3:
                        reason.append("Hakkındaki haberler oldukça olumsuz, uzun vadede düşüş etkisi sürebilir.")
                    elif -0.3 <= sentiment_score < 0:
                        reason.append("Hakkındaki haberler hafif olumsuz, uzun vadede düşüş etkisi sürebilir.")
                if abs(change) > 1.5:
                    reason.append("Uzun vadede daha belirgin bir düşüş bekleniyor.")

        # Her zaman bir neden sağlandığından emin olalım
        if not reason:
            if trend == 'yükseliş':
                reason.append("Hisse fiyatı genel piyasa dinamiklerine bağlı olarak yükseliyor.")
            else:
                reason.append("Hisse fiyatı genel piyasa dinamiklerine bağlı olarak düşüyor.")

        reasons[period] = " ".join(reason)

    return reasons



def create_dataset(dataset, time_step=20):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i + time_step])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)


# Veriyi training ve test olarak ayırma
def split_data(dataset, time_step=60, train_ratio=0.8):
    X, y = create_dataset(dataset, time_step)
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test



def predict_future(model, n_days, last_sequence, time_step, input_dim):
    predictions = []
    current_seq = last_sequence.copy()
    for _ in range(n_days):
        input_seq = current_seq[-time_step:]
        input_seq = input_seq.reshape(1, time_step, input_dim)
        pred = model.predict(input_seq, verbose=0)[0][0]
        last_row = current_seq[-1].copy()
        last_row[0] = pred
        current_seq = np.append(current_seq, [last_row], axis=0)
        predictions.append(pred)
    return predictions


# Model eğitme fonksiyonu (Güncellendi)
def train_model(X_train, y_train, X_test, y_test, input_shape):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(32))
    model.add(Dense(16))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Test seti üzerinde performansı değerlendir
    if X_test.size > 0 and y_test.size > 0:
        y_pred = model.predict(X_test, verbose=0)
        test_mae = mean_absolute_error(y_test, y_pred)
        print(f"Test Seti MAE: {test_mae:.2f} TL")
    else:
        print("Test seti verisi yetersiz, performans değerlendirmesi yapılamadı.")

    return model


# Grafik veri fonksiyonları
def get_chart_data(data, future_close=None, ticker=None):
    if data.empty:
        return {'dates': [], 'prices': []}
    dates = data.index[-250:].strftime('%Y-%m-%d').tolist()
    prices = data['Close'].values[-250:].tolist()
    return {'dates': dates, 'prices': prices}


def get_prediction_chart_data(data, future_close, ticker):
    if data.empty:
        return {'past': {'dates': [], 'prices': []}, 'future': {'dates': [], 'prices': []}}
    past_dates = data.index[-50:].strftime('%Y-%m-%d').tolist()
    past_prices = data['Close'].values[-50:].tolist()
    future_dates = pd.date_range(start=data.index[-1], periods=23, freq='B')[1:].strftime('%Y-%m-%d').tolist()
    return {
        'past': {'dates': past_dates, 'prices': past_prices},
        'future': {'dates': future_dates, 'prices': future_close.tolist()}
    }


# E-posta doğrulama fonksiyonu
def send_verification_email(email, token):
    verification_url = url_for('verify_email', token=token, _external=True)
    msg = Message('E-posta Doğrulama', sender=app.config['MAIL_USERNAME'], recipients=[email])
    msg.body = f"Lütfen e-posta adresinizi doğrulamak için aşağıdaki bağlantıya tıklayın:\n{verification_url}"
    try:
        mail.send(msg)
        return True
    except Exception as e:
        print(f"E-posta gönderme hatası: {e}")
        return False


# Doğruluk metriklerini hesaplayan fonksiyon (Güncellendi)
def calculate_metrics():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Tüm tahminleri çek
    cursor.execute("SELECT ticker, prediction_date, price_1_day, price_5_days, price_1_month FROM predictions")
    predictions = cursor.fetchall()
    conn.close()

    # Her zaman dilimi için metrikleri saklamak için listeler
    metrics = {
        '1_day': {'actual': [], 'predicted': [], 'actual_directions': [], 'predicted_directions': []},
        '5_days': {'actual': [], 'predicted': [], 'actual_directions': [], 'predicted_directions': []},
        '1_month': {'actual': [], 'predicted': [], 'actual_directions': [], 'predicted_directions': []}
    }

    # Her tahmin için gerçek verileri çek ve metrikleri hesapla
    for ticker, date, pred_1_day, pred_5_days, pred_1_month in predictions:
        stock = yf.Ticker(ticker)
        try:
            date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
            prev_date = date_obj - timedelta(days=1)  # Bir önceki gün
            actual_1_day_date = date_obj + timedelta(days=1)  # 1 gün sonrası
            actual_5_days_date = date_obj + timedelta(days=5)  # 5 gün sonrası
            actual_1_month_date = date_obj + timedelta(days=30)  # 30 gün sonrası

            # Gerçek verileri çek
            prev_data = stock.history(start=prev_date, end=prev_date)
            actual_1_day_data = stock.history(start=actual_1_day_date, end=actual_1_day_date)
            actual_5_days_data = stock.history(start=actual_5_days_date, end=actual_5_days_date)
            actual_1_month_data = stock.history(start=actual_1_month_date, end=actual_1_month_date)
        except Exception as e:
            print(f"Date parsing or data fetch error for {ticker} on {date}: {e}")
            continue

        if not prev_data.empty:
            prev_price = prev_data['Close'].iloc[0]

            # 1 günlük metrikler
            if not actual_1_day_data.empty:
                actual_1_day_price = actual_1_day_data['Close'].iloc[0]
                if actual_1_day_price is not None and pred_1_day is not None:
                    metrics['1_day']['actual'].append(actual_1_day_price)
                    metrics['1_day']['predicted'].append(pred_1_day)
                    actual_direction = 1 if actual_1_day_price > prev_price else 0
                    predicted_direction = 1 if pred_1_day > prev_price else 0
                    metrics['1_day']['actual_directions'].append(actual_direction)
                    metrics['1_day']['predicted_directions'].append(predicted_direction)

            # 5 günlük metrikler
            if not actual_5_days_data.empty:
                actual_5_days_price = actual_5_days_data['Close'].iloc[0]
                if actual_5_days_price is not None and pred_5_days is not None:
                    metrics['5_days']['actual'].append(actual_5_days_price)
                    metrics['5_days']['predicted'].append(pred_5_days)
                    actual_direction = 1 if actual_5_days_price > prev_price else 0
                    predicted_direction = 1 if pred_5_days > prev_price else 0
                    metrics['5_days']['actual_directions'].append(actual_direction)
                    metrics['5_days']['predicted_directions'].append(predicted_direction)

            # 1 aylık metrikler
            if not actual_1_month_data.empty:
                actual_1_month_price = actual_1_month_data['Close'].iloc[0]
                if actual_1_month_price is not None and pred_1_month is not None:
                    metrics['1_month']['actual'].append(actual_1_month_price)
                    metrics['1_month']['predicted'].append(pred_1_month)
                    actual_direction = 1 if actual_1_month_price > prev_price else 0
                    predicted_direction = 1 if pred_1_month > prev_price else 0
                    metrics['1_month']['actual_directions'].append(actual_direction)
                    metrics['1_month']['predicted_directions'].append(predicted_direction)

    # Her zaman dilimi için metrikleri hesapla
    results = {}
    overall_metrics = {'MAE': [], 'RMSE': [], 'MAPE': [], 'Accuracy': [], 'F1': [], 'Precision': [], 'Recall': []}

    for period in ['1_day', '5_days', '1_month']:
        actual = metrics[period]['actual']
        predicted = metrics[period]['predicted']
        actual_directions = metrics[period]['actual_directions']
        predicted_directions = metrics[period]['predicted_directions']

        if actual and predicted:
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((np.array(actual) - np.array(predicted)) / np.array(actual))) * 100
        else:
            mae = rmse = mape = None

        if actual_directions and predicted_directions:
            accuracy = accuracy_score(actual_directions, predicted_directions)
            f1 = f1_score(actual_directions, predicted_directions)
            precision = precision_score(actual_directions, predicted_directions)
            recall = recall_score(actual_directions, predicted_directions)
        else:
            accuracy = f1 = precision = recall = None

        results[period] = {
            'MAE': mae, 'RMSE': rmse, 'MAPE': mape,
            'Accuracy': accuracy, 'F1': f1, 'Precision': precision, 'Recall': recall
        }

        # Ortalama için değerleri sakla
        if mae is not None:
            overall_metrics['MAE'].append(mae)
        if rmse is not None:
            overall_metrics['RMSE'].append(rmse)
        if mape is not None:
            overall_metrics['MAPE'].append(mape)
        if accuracy is not None:
            overall_metrics['Accuracy'].append(accuracy)
        if f1 is not None:
            overall_metrics['F1'].append(f1)
        if precision is not None:
            overall_metrics['Precision'].append(precision)
        if recall is not None:
            overall_metrics['Recall'].append(recall)

    # Genel ortalamayı hesapla
    overall_results = {}
    for metric, values in overall_metrics.items():
        overall_results[metric] = np.mean(values) if values else None

    # Raporu yazdır ve dosyaya kaydet
    print("\n--- Performans Metrikleri (Detaylı) ---")
    with open('metrics_report.txt', 'w') as f:
        f.write("--- Performans Metrikleri (Detaylı) ---\n")
        for period in ['1_day', '5_days', '1_month']:
            print(f"\n{period.replace('_', ' ').title()}:")
            f.write(f"\n{period.replace('_', ' ').title()}:\n")
            for metric, value in results[period].items():
                if value is not None:
                    formatted_value = f"{value:.2f}" if metric != 'MAPE' else f"{value:.2f}%"
                    formatted_value += " TL" if metric in ['MAE', 'RMSE'] else ""
                    print(f"{metric}: {formatted_value}")
                    f.write(f"{metric}: {formatted_value}\n")
                else:
                    print(f"{metric}: Veri yetersiz")
                    f.write(f"{metric}: Veri yetersiz\n")

        print("\n--- Genel Ortalama Metrikler ---")
        f.write("\n--- Genel Ortalama Metrikler ---\n")
        for metric, value in overall_results.items():
            if value is not None:
                formatted_value = f"{value:.2f}" if metric != 'MAPE' else f"{value:.2f}%"
                formatted_value += " TL" if metric in ['MAE', 'RMSE'] else ""
                print(f"{metric}: {formatted_value}")
                f.write(f"{metric}: {formatted_value}\n")
            else:
                print(f"{metric}: Veri yetersiz")
                f.write(f"{metric}: Veri yetersiz\n")

    return results, overall_results


# Rotalar
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            flash('Tüm alanlar doldurulmalıdır.', 'error')
            return redirect(url_for('register'))

        is_valid, password_error = validate_password(password)
        if not is_valid:
            flash(password_error, 'error')
            return redirect(url_for('register'))

        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ? OR username = ?", (email, username))
        existing_user = c.fetchone()

        if existing_user:
            flash('E-posta veya kullanıcı adı zaten kullanılıyor.', 'error')
            conn.close()
            return redirect(url_for('register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        verification_token = str(uuid.uuid4())
        c.execute("INSERT INTO users (username, email, password, verification_token) VALUES (?, ?, ?, ?)",
                  (username, email, hashed_password, verification_token))
        conn.commit()
        conn.close()

        if send_verification_email(email, verification_token):
            flash('Kayıt başarılı! Lütfen e-posta adresinizi doğrulamak için gelen bağlantıya tıklayın.', 'success')
        else:
            flash('E-posta gönderilemedi. Lütfen tekrar deneyin.', 'error')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/verify_email/<token>')
def verify_email(token):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE verification_token = ?", (token,))
    user = c.fetchone()

    if not user:
        flash('Geçersiz veya süresi dolmuş doğrulama bağlantısı.', 'error')
        conn.close()
        return redirect(url_for('login'))

    c.execute("UPDATE users SET is_verified = 1, verification_token = NULL WHERE id = ?", (user[0],))
    conn.commit()
    conn.close()

    flash('E-posta adresiniz doğrulandı! Şimdi giriş yapabilirsiniz.', 'success')
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        conn.close()

        if not user:
            flash('Geçersiz e-posta veya şifre.', 'error')
            return redirect(url_for('login'))

        if not user[4]:
            flash('Lütfen önce e-posta adresinizi doğrulamayın.', 'error')
            return redirect(url_for('login'))

        if bcrypt.check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session.permanent = True
            flash('Giriş başarılı!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Geçersiz e-posta veya şifre.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('Çıkış yapıldı.', 'success')
    return redirect(url_for('login'))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash('Lütfen önce giriş yapın.', 'error')
        return redirect(url_for('login'))

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT username, email FROM users WHERE id = ?", (session['user_id'],))
    user = c.fetchone()

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email:
            flash('Kullanıcı adı ve e-posta zorunludur.', 'error')
            conn.close()
            return redirect(url_for('profile'))

        c.execute("SELECT * FROM users WHERE (username = ? OR email = ?) AND id != ?",
                  (username, email, session['user_id']))
        existing_user = c.fetchone()

        if existing_user:
            flash('Kullanıcı adı veya e-posta zaten kullanılıyor.', 'error')
            conn.close()
            return redirect(url_for('profile'))

        updates = []
        params = []
        if username != user[0]:
            updates.append("username = ?")
            params.append(username)
        if email != user[1]:
            updates.append("email = ?")
            params.append(email)
            c.execute("UPDATE users SET is_verified = 0 WHERE id = ?", (session['user_id'],))
            verification_token = str(uuid.uuid4())
            updates.append("verification_token = ?")
            params.append(verification_token)
            send_verification_email(email, verification_token)
            flash('E-posta adresiniz değişti. Lütfen yeni adresinizi doğrulayın.', 'success')

        if password:
            is_valid, password_error = validate_password(password)
            if not is_valid:
                flash(password_error, 'error')
                conn.close()
                return redirect(url_for('profile'))
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            updates.append("password = ?")
            params.append(hashed_password)

        if updates:
            params.append(session['user_id'])
            c.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
            conn.commit()
            session['username'] = username
            flash('Profiliniz güncellendi.', 'success')

        conn.close()
        return redirect(url_for('profile'))

    conn.close()
    return render_template('profile.html', user=user)


@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    if 'user_id' not in session:
        flash('Lütfen önce giriş yapın.', 'error')
        return redirect(url_for('login'))

    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    if request.method == 'POST':
        ticker = request.form.get('ticker', '').upper()
        action = request.form.get('action')

        if action == 'add':
            if not ticker:
                flash('Hisse kodu giriniz.', 'error')
            elif ticker not in bist100_list:
                flash('Geçersiz hisse kodu.', 'error')
            else:
                c.execute("SELECT * FROM portfolios WHERE user_id = ? AND ticker = ?", (session['user_id'], ticker))
                if c.fetchone():
                    flash('Bu hisse zaten portföyünüzde.', 'error')
                else:
                    c.execute("INSERT INTO portfolios (user_id, ticker) VALUES (?, ?)", (session['user_id'], ticker))
                    conn.commit()
                    flash('Hisse portföyünüze eklendi.', 'success')

        elif action == 'remove':
            c.execute("DELETE FROM portfolios WHERE user_id = ? AND ticker = ?", (session['user_id'], ticker))
            conn.commit()
            flash('Hisse portföyünüzden kaldırıldı.', 'success')

    c.execute("SELECT ticker FROM portfolios WHERE user_id = ?", (session['user_id'],))
    portfolio = [row[0] for row in c.fetchall()]
    portfolio_data = []
    for ticker in portfolio:
        data = load_data(ticker)
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            portfolio_data.append({'ticker': ticker, 'current_price': round(current_price, 2)})

    conn.close()
    return render_template('portfolio.html', portfolio=portfolio_data, bist100_list=bist100_list)


@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Lütfen önce giriş yapın.', 'error')
        return redirect(url_for('login'))

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute(
        "SELECT ticker, prediction_date, price_1_day, price_5_days, price_1_month FROM predictions WHERE user_id = ? ORDER BY prediction_date DESC",
        (session['user_id'],))
    predictions = c.fetchall()
    conn.close()

    return render_template('history.html', predictions=predictions)


@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user_id' not in session:
        flash('Lütfen önce giriş yapın.', 'error')
        return redirect(url_for('login'))

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT is_verified FROM users WHERE id = ?", (session['user_id'],))
    is_verified = c.fetchone()[0]
    conn.close()

    if not is_verified:
        flash('Lütfen e-posta adresinizi doğrulamak için gelen bağlantıya tıklayın.', 'error')
        return redirect(url_for('profile'))

    error = None
    predictions = None
    headlines = []
    ticker = None
    today_price = None
    prediction_chart_data = None
    reasons = None
    news_message = None
    history_chart_data = None
    selected_ticker = 'THYAO.IS'
    preference_details = None

    if request.method == 'POST':
        ticker_input = request.form.get('ticker', '').upper()
        selected_stock = request.form.get('selected_stock', '').upper()
        print(f"Form verileri - ticker: {ticker_input}, selected_stock: {selected_stock}")

        if selected_stock and selected_stock in bist100_list:
            selected_ticker = selected_stock
        elif ticker_input and ticker_input in bist100_list:
            selected_ticker = ticker_input
        else:
            error = "Geçersiz hisse kodu veya seçim yapılmadı."
            print(f"Hata: Geçersiz hisse kodu - ticker: {ticker_input}, selected_stock: {selected_stock}")

    print(f"Seçilen ticker: {selected_ticker}")

    if selected_ticker:
        data = load_data(selected_ticker)
        if not data.empty:
            history_chart_data = get_chart_data(data, ticker=selected_ticker)
            today_price = float(data['Close'].values[-1]) if not data.empty else None
            preference_details = get_preference_details(selected_ticker)
        else:
            error = f"{selected_ticker} için veri çekilemedi."

    if request.method == 'POST' and request.form.get('action') == 'predict':
        ticker = selected_ticker
        if not ticker:
            error = "Hisse kodu giriniz."
        else:
            is_weekend = datetime.now().weekday() in [5, 6]
            cache_key = (ticker, datetime.today().date())

            if cache_key in prediction_cache:
                cached_result = prediction_cache[cache_key]
                predictions = cached_result['predictions']
                headlines = cached_result['headlines']
                news_message = cached_result['news_message']
                prediction_chart_data = cached_result['prediction_chart_data']
                reasons = cached_result['reasons']
                today_price = cached_result['today_price']
                print(f"{ticker}: Önbellekten tahmin alındı: {predictions}")
            else:
                data = load_data(ticker)
                if data.empty:
                    error = "Hisse kodu geçersiz ya da veri çekilemedi."
                else:
                    data = compute_indicators(data, ticker)
                    sentiment_score, headlines, news_message = get_sentiment_score(ticker)
                    if not headlines:
                        news_message = news_message or f"{ticker} için ilgili son haberler bulunamadı."
                    data['Sentiment'] = 0.0 if sentiment_score is None else sentiment_score  # Sentiment sıfırlanır
                    data = data.dropna()

                    features = data[['Close', 'RSI', 'MACD', 'Sentiment', 'BIST100_Close']]
                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(features)
                    scaled[:, 3] *= 1.5
                    scaled[:, 4] *= 2.0

                    time_step = 60
                    # Veriyi training ve test olarak ayır (%80 training, %20 test)
                    X_train, X_test, y_train, y_test = split_data(scaled, time_step, train_ratio=0.8)
                    X_train = X_train.reshape(X_train.shape[0], time_step, X_train.shape[2])
                    X_test = X_test.reshape(X_test.shape[0], time_step, X_test.shape[2])

                    # Modeli sadece training verisiyle eğit
                    model = train_model(X_train, y_train, X_test, y_test, (time_step, X_train.shape[2]))

                    last_sequence = scaled[-time_step:]
                    future_scaled = predict_future(model, 22, last_sequence, time_step, X_train.shape[2])
                    future_close = scaler.inverse_transform(
                        np.hstack((np.array(future_scaled).reshape(-1, 1), np.zeros((22, 4))))
                    )[:, 0]

                    today_price = float(data['Close'].values[-1]) if not data.empty else None
                    predictions = {
                        '1_day': future_close[0],
                        '5_days': future_close[4],
                        '1_month': future_close[21],
                        'sentiment_score': sentiment_score if sentiment_score is not None else 0.0  # Skor sıfırlanır
                    }
                    reasons = analyze_reasons(data, sentiment_score, today_price,
                                              [future_close[0], future_close[4], future_close[21]])
                    prediction_chart_data = get_prediction_chart_data(data, future_close, ticker)

                    prediction_cache[cache_key] = {
                        'predictions': predictions,
                        'headlines': headlines,
                        'news_message': news_message,
                        'prediction_chart_data': prediction_chart_data,
                        'reasons': reasons,
                        'today_price': today_price
                    }
                    print(f"{ticker}: Yeni tahmin önbelleğe kaydedildi: {predictions}")

                    conn = sqlite3.connect('database.db')
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO predictions (user_id, ticker, prediction_date, price_1_day, price_5_days, price_1_month) VALUES (?, ?, ?, ?, ?, ?)",
                        (session['user_id'], ticker, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                         predictions['1_day'], predictions['5_days'], predictions['1_month']))
                    conn.commit()
                    conn.close()
                    backup_database()

    return render_template('index.html', bist100_list=bist100_list, error=error, predictions=predictions,
                           headlines=headlines or [], ticker=ticker, today_price=today_price,
                           prediction_chart_data=prediction_chart_data, reasons=reasons,
                           news_message=news_message, history_chart_data=history_chart_data,
                           selected_ticker=selected_ticker, ticker_to_company=ticker_to_company,
                           preference_details=preference_details)


@app.route('/info')
def info():
    if 'user_id' not in session:
        flash('Lütfen önce giriş yapın.', 'error')
        return redirect(url_for('login'))
    return render_template('info.html')


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)