# ai_model.py
# (Makine Öğrenimi Modülü)
# Bu modül, veri hazırlama, model eğitimi, değerlendirmesi ve tahmini gibi tüm makine öğrenimi ile ilgili işlevleri içerir.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib # Model ve scaler kaydetmek/yüklemek için
import os
import logging
from datetime import datetime, timedelta

# Loglama yapılandırması
logging.basicConfig(filename='crypto_app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logger = logging.getLogger(__name__)

# Model ve Scaler kaydetme yolu
MODEL_PATH = 'xgboost_model.joblib'
SCALER_PATH = 'scaler.joblib'
FEATURES_PATH = 'features.joblib' # Özellik isimlerini kaydetmek için

# Yardımcı fonksiyonları burada tanımlıyoruz veya doğrudan import ediyoruz.
from utils import add_technical_indicators, add_market_time_features


def prepare_data_for_model(data: pd.DataFrame, prediction_days: int = 1):
    """
    Model için veriyi hazırlar: Özellik mühendisliği, hedef değişken oluşturma,
    veri ölçeklendirme ve train/test setlerine ayırma (kronolojik).

    Args:
        data (pd.DataFrame): Ham fiyat verileri (Open, High, Low, Close, Volume).
        prediction_days (int): Kaç gün sonraki fiyatın tahmin edileceği.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_test_raw_df, features_cols, last_known_close_prices_for_test, status)
               veya hata durumunda (None, None, None, None, None, None, None, None, status)
    """
    if data.empty or not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        logger.error("Veri boş veya gerekli sütunlardan ('Open', 'High', 'Low', 'Close', 'Volume') biri eksik.")
        return None, None, None, None, None, None, None, None, "error_empty_data_or_missing_cols"

    df = data.copy()

    # --- Özellik Mühendisliği ---
    # Teknik göstergeler ekle
    # Bu fonksiyon, NaN değerleri kendi içinde doldurur ve tamamen NaN olan sütunları düşürür.
    df = add_technical_indicators(df)
    
    # Piyasa zamanı özelliklerini ekle
    df = add_market_time_features(df)

    # Geçmiş kapanış fiyatlarını özellik olarak ekle (lagged features)
    for i in range(1, 21): # Son 20 günün kapanış fiyatları (veya ihtiyaca göre ayarla)
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_Lag_{i}'] = df['Volume'].shift(i) # Hacim lag'leri de ekleyebiliriz

    logger.info("prepare_data_for_model: Özellik mühendisliği tamamlandı.")

    # HEDEF DEĞİŞKENİ OLUŞTUR: prediction_days sonraki kapanış fiyatı (mutlak değer olarak)
    df['Target_Price'] = df['Close'].shift(-prediction_days)
    
    logger.info(f"prepare_data_for_model: Hedef değişken (mutlak fiyat) {prediction_days} gün sonrası için oluşturuldu.")

    # NaN değerleri temizle (shift ve gösterge hesaplamalarından kaynaklananlar)
    # Target_Price'ın NaN olduğu satırları da kaldırır (en sondaki prediction_days kadar satır)
    initial_rows = len(df)
    data_processed = df.dropna()
    rows_dropped = initial_rows - len(data_processed)
    if rows_dropped > 0:
        logger.warning(f"prepare_data_for_model: NaN değerler nedeniyle {rows_dropped} satır düşürüldü.")

    if data_processed.empty:
        logger.warning("Veri işlendikten ve NaN değerler atıldıktan sonra boş kaldı. Yeterli geçmiş veri olmayabilir.")
        return None, None, None, None, None, None, None, None, "error_empty_after_dropna"

    # Özellik sütunlarını belirle
    # 'Target_Price' ve orijinal fiyat sütunları (Open, High, Low, Close, Volume) özelliklerden çıkarılır.
    # Sadece sayısal ve sonsuz olmayan değerlere sahip sütunları seç.
    excluded_cols = ['Target_Price', 'Open', 'High', 'Low', 'Close', 'Volume']
    features_cols = [col for col in data_processed.columns if col not in excluded_cols]
    
    # Sadece sayısal özellik sütunlarını al
    numeric_features_cols = data_processed[features_cols].select_dtypes(include=np.number).columns.tolist()
    features_cols = numeric_features_cols # Güncellenmiş özellik listesi

    if not features_cols:
        logger.error("Hiç özellik sütunu bulunamadı. Lütfen özellik mühendisliği adımlarını kontrol edin.")
        return None, None, None, None, None, None, None, None, "error_no_features_found"

    X = data_processed[features_cols]
    y = data_processed['Target_Price'] # Hedef mutlak fiyat

    # --- ZAMAN SERİSİ TRAIN-TEST BÖLME (KRONOLOJİK) ---
    # Veriyi kronolojik sıraya göre %80 eğitim, %20 test olarak ayır.
    train_size = int(len(X) * 0.8) # %80 eğitim için

    # Veri yeterince büyük mü kontrol et
    if train_size < 2 or len(X) - train_size < 1: # En az 2 eğitim, 1 test örneği
        logger.error(f"Eğitim ({train_size}) veya test ({len(X) - train_size}) seti boyutu yetersiz. Veri seti çok küçük.")
        return None, None, None, None, None, None, None, None, "error_insufficient_data_for_split"

    X_train_df = X.iloc[:train_size]
    X_test_df = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    # Test setindeki orijinal kapanış fiyatlarını sakla (değerlendirme için gerekli)
    last_known_close_prices_for_test = data_processed.loc[y_test.index, 'Close']
    
    # X_test_raw_df, ölçeklenmemiş X_test verisinin kendisidir,
    # gelecekteki bir günün tahminini yapmak için kullanılacak olan son veri noktasının
    # orijinal halini saklar.
    X_test_raw_df = X_test_df.copy()

    if X_train_df.empty or X_test_df.empty or y_train.empty or y_test.empty:
        logger.error("Eğitim veya test setleri bölme işleminden sonra boş kaldı. Veri aralığını veya boyutunu kontrol edin.")
        return None, None, None, None, None, None, None, None, "error_empty_train_test_sets"

    # --- Özellik Ölçeklendirme ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df) # Test setini eğitim seti scaler'ı ile ölçekle

    logger.info(f"Veri hazırlığı tamamlandı. Eğitim boyutu: {len(X_train_scaled)}, Test boyutu: {len(X_test_scaled)}")
    logger.info(f"Hazırlanan özellik sayısı: {len(features_cols)}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_test_raw_df, features_cols, last_known_close_prices_for_test, "success"


def train_xgboost_model(X_train: np.ndarray, y_train: pd.Series, use_grid_search: bool = False):
    """
    XGBoost Regressor modelini eğitir. İsteğe bağlı olarak GridSearchCV kullanabilir.
    Bu versiyon, early_stopping_rounds parametresini doğrudan fit metoduna geçirmeyi denemez
    veya callbacks kullanmaz. Model belirlenen n_estimators kadar eğitilir.

    Args:
        X_train (np.array): Eğitim özellikleri.
        y_train (pd.Series): Eğitim hedef değişkeni.
        use_grid_search (bool): GridSearchCV kullanılıp kullanılmayacağı.
        # early_stopping_rounds (int, optional): Bu versiyonda göz ardı edilir.

    Returns:
        tuple: (eğitilmiş model, durum mesajı)
    """
    if X_train is None or y_train is None or len(X_train) == 0:
        logger.error("Eğitim verisi boş veya yetersiz.")
        return None, "error_empty_training_data"

    # XGBoost Regressor modelini tanımla
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000, # Model bu kadar ağaçla eğitilir. Erken durdurma devre dışı.
        learning_rate=0.01,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1 # Tüm CPU çekirdeklerini kullan
    )

    best_model = None
    if use_grid_search:
        logger.info("GridSearchCV ile model eğitimi başlatılıyor... Bu biraz zaman alabilir.")
        param_grid = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05],
            'max_depth': [3, 5],
            'subsample': [0.7, 0.9],
            'colsample_bytree': [0.7, 0.9]
        }
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
        
        try:
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            logger.info(f"GridSearchCV en iyi parametreler: {grid_search.best_params_}")
            return best_model, "success_grid_search"
        except Exception as e:
            logger.error(f"GridSearchCV ile model eğitimi sırasında hata: {e}")
            return None, f"error_grid_search_failed: {e}"
    else:
        logger.info("Doğrudan XGBoost modeli eğitimi başlatılıyor (erken durdurma kapalı)...")
        try:
            # early_stopping_rounds parametresi artık burada doğrudan geçirilmiyor.
            xgb_model.fit(X_train, y_train)
            best_model = xgb_model
            logger.info("Doğrudan XGBoost modeli başarıyla eğitildi.")
            return best_model, "success_direct_train"
        except Exception as e:
            logger.error(f"Doğrudan model eğitimi sırasında hata: {e}")
            return None, f"error_direct_train_failed: {e}"

def evaluate_model(model, X_test_scaled: np.ndarray, y_test_true: pd.Series, y_pred_raw: np.ndarray, y_test_original_prices: pd.Series):
    """
    Modelin performans metriklerini hesaplar ve gösterir.
    Doğrudan mutlak fiyatlar üzerinden değerlendirme yapar.
    
    Args:
        model: Eğitilmiş makine öğrenimi modeli.
        X_test_scaled (np.ndarray): Ölçeklenmiş test özellikleri.
        y_test_true (pd.Series): Gerçek hedef değerleri (mutlak fiyatlar).
        y_pred_raw (np.ndarray): Modelin ham tahminleri (mutlak fiyatlar).
        y_test_original_prices (pd.Series): Test setindeki her bir tahminin yapıldığı zamandaki gerçek kapanış fiyatı.
                                            (Target_Price'ın bir gün/X gün öncesi)
        
    Returns:
        tuple: (dict: metrikler, pd.Series: y_test_abs, pd.Series: y_pred_abs)
    """
    # y_pred_raw zaten mutlak fiyat tahminidir.
    y_pred_abs = pd.Series(y_pred_raw.flatten(), index=y_test_true.index)

    if y_test_true.empty or y_pred_abs.empty or len(y_test_true) != len(y_pred_abs):
        logger.warning("evaluate_model: Değerlendirme için gerçek veya tahmin edilen değerler boş veya boyutları uyuşmuyor.")
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}, pd.Series(), pd.Series()

    # NaN veya inf değerleri olan satırları filtrele
    valid_indices = ~np.isnan(y_test_true) & ~np.isinf(y_test_true) & \
                    ~np.isnan(y_pred_abs) & ~np.isinf(y_pred_abs)
    
    y_test_abs_filtered = y_test_true[valid_indices]
    y_pred_abs_filtered = y_pred_abs[valid_indices]
    # y_test_original_prices'ı da filtrele, çünkü MAPE hesaplamasında kullanılacak
    y_test_original_prices_filtered = y_test_original_prices[valid_indices]

    if y_test_abs_filtered.empty:
        logger.warning("evaluate_model: Geçerli veri kalmadı. Metrikler hesaplanamıyor.")
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}, pd.Series(), pd.Series()

    mae = mean_absolute_error(y_test_abs_filtered, y_pred_abs_filtered)
    rmse = np.sqrt(mean_squared_error(y_test_abs_filtered, y_pred_abs_filtered))
    r2 = r2_score(y_test_abs_filtered, y_pred_abs_filtered)
    
    # MAPE hesaplaması: sıfıra bölmeyi önle
    # Gerçek değerin kendisi (y_test_abs_filtered) değil, tahminin yapıldığı zamandaki kapanış fiyatı (y_test_original_prices_filtered)
    # MAPE için referans olarak kullanılmalıdır.
    non_zero_mape_indices = y_test_original_prices_filtered != 0
    if non_zero_mape_indices.any():
        mape = np.mean(np.abs((y_test_abs_filtered[non_zero_mape_indices] - y_pred_abs_filtered[non_zero_mape_indices]) / y_test_original_prices_filtered[non_zero_mape_indices])) * 100
    else:
        mape = np.nan # Tüm referans fiyatlar sıfırsa MAPE hesaplanamaz

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }
    logger.info(f"Model metrikleri: {metrics}")
    
    return metrics, y_test_abs_filtered, y_pred_abs_filtered


def predict_next_day_price(model, scaler, last_data_point: pd.DataFrame, features_cols: list) -> float:
    """
    Modeli kullanarak bir sonraki günün (prediction_days sonraki) kapanış fiyatını tahmin eder.
    
    Args:
        model: Eğitilmiş makine öğrenimi modeli.
        scaler: Veri ölçekleyici (StandardScaler).
        last_data_point (pd.DataFrame): Tahmin edilecek son günün İŞLENMİŞ verisi (tek satır DataFrame).
                                        Bu DataFrame, modelin eğitiminde kullanılan özellik sütunlarına sahip olmalıdır.
        features_cols (list): Modelin beklediği özellik sütunlarının listesi.
        
    Returns:
        float: Tahmin edilen sonraki günün kapanış fiyatı.
    """
    if last_data_point.empty or model is None or scaler is None:
        logger.warning("predict_next_day_price: Tahmin için boş son veri noktası, model veya scaler sağlandı.")
        return np.nan
    
    # Sadece modelin beklediği özellikleri seç ve sıra uyumunu kontrol et
    try:
        input_df = last_data_point[features_cols]
    except KeyError as e:
        logger.error(f"Tahmin edilecek veride eksik veya yanlış özellik sütunu: {e}. "
                     f"Lütfen 'last_data_point'in 'features_cols' listesindeki tüm sütunları içerdiğinden emin olun.")
        return np.nan
    
    # NaN değerlerini kontrol et (varsa, model tahminini etkileyebilir)
    if input_df.isnull().sum().sum() > 0:
        logger.warning("predict_next_day_price: Tahmin için kullanılan 'last_data_point' içinde NaN değerler bulundu. Tahmin yapılamıyor.")
        return np.nan 

    # Ölçekleme
    input_scaled = scaler.transform(input_df)
    
    # Modelden doğrudan mutlak fiyatı tahmin et
    predicted_price = model.predict(input_scaled)[0]
    
    logger.info(f"predict_next_day_price: Tahmin edilen fiyat: {predicted_price:.2f}")
    return predicted_price

def save_model(model, scaler, features_cols, model_path=MODEL_PATH, scaler_path=SCALER_PATH, features_path=FEATURES_PATH):
    """Eğitilmiş modeli, ölçekleyiciyi ve özellik sütunlarını kaydeder."""
    try:
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(features_cols, features_path)
        logger.info(f"Model, ölçekleyici ve özellikler '{model_path}', '{scaler_path}' ve '{features_path}' konumlarına kaydedildi.")
        return "success_save"
    except Exception as e:
        logger.error(f"Model kaydedilirken hata oluştu: {e}")
        return f"error_save_failed: {e}"

def load_model(model_path=MODEL_PATH, scaler_path=SCALER_PATH, features_path=FEATURES_PATH):
    """Kaydedilmiş modeli, ölçekleyiciyi ve özellik isimlerini yükler."""
    model = None
    scaler = None
    features_cols = None
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            features_cols = joblib.load(features_path)
            logger.info("Model, scaler ve özellikler başarıyla yüklendi.")
            return model, scaler, features_cols, "success_load"
        else:
            logger.warning("Kaydedilmiş model, scaler veya özellik dosyası bulunamadı.")
            return None, None, None, "warning_not_found"
    except Exception as e:
        logger.error(f"Model yüklenirken hata oluştu: {e}")
        return None, None, None, f"error_load_failed: {e}"

if __name__ == "__main__":
    # Bu bölüm, modülün bağımsız olarak test edilmesi içindir ve Streamlit uygulamasına dahil edilmez.
    # Bu nedenle, bu bölümde Streamlit çağrıları kullanılmamalıdır.
    logger.info("AI Model Modülü Testi (Bağımsız Çalışma)")
    
    logger.info("Örnek veri seti oluşturuluyor...")
    # Yaklaşık 2 yıl (730 gün) veri oluştur.
    start_date = datetime.now() - timedelta(days=730)
    end_date = datetime.now()
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Daha fazla veri noktası için daha uzun bir aralık kullanıyoruz.
    # Rastgele veriye trend ekleyelim ki modelin öğrenebileceği bir şeyler olsun.
    np.random.seed(42) # Tekrarlanabilirlik için
    base_price = 1000
    prices = [base_price]
    for _ in range(1, len(dates)):
        change = np.random.normal(0, 5) # Küçük günlük değişim
        prices.append(prices[-1] + change)
    prices = np.array(prices)
    
    dummy_data = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'High': prices * (1 + np.random.uniform(0.005, 0.02, len(dates))),
        'Low': prices * (1 - np.random.uniform(0.005, 0.02, len(dates))),
        'Close': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'Volume': np.random.randint(100000, 5000000, len(dates))
    }, index=dates)

    # Negatif fiyatları 1'e sabitle (gerçekçi değilse)
    for col in ['Open', 'High', 'Low', 'Close']:
        dummy_data[col] = np.maximum(1, dummy_data[col])
    
    logger.info(f"Oluşturulan örnek veri boyutu: {dummy_data.shape}")
    logger.info(f"Örnek veri head:\n{dummy_data.head()}")
    
    prediction_days_test = 1 # Test için sabit gün

    # prepare_data_for_model testi
    X_train, X_test, y_train, y_test, scaler, X_test_raw, features_cols, last_known_close_prices_for_test, prep_status = \
        prepare_data_for_model(dummy_data.copy(), prediction_days=prediction_days_test)
    
    if prep_status == "success":
        logger.info(f"Veri hazırlığı başarılı. Eğitim seti boyutu: {X_train.shape}, Test seti boyutu: {X_test.shape}")
        logger.info(f"Özellik sütunları sayısı: {len(features_cols)}")
        # İlk 5 özellik sütununu logla
        logger.info(f"Özellik sütunları (ilk 5): {features_cols[:5]}...")

        # train_xgboost_model testi
        # GridSearchCV olmadan ve erken durdurma ile eğitim
        # early_stopping_rounds parametresi artık burada doğrudan geçirilmiyor.
        model, train_status = train_xgboost_model(X_train, y_train, use_grid_search=False) # early_stopping_rounds kaldırıldı
        
        if model:
            # Modelden tahmin al
            y_pred = model.predict(X_test)
            logger.info(f"Tahminlerin boyutu: {y_pred.shape}")
            logger.info(f"Gerçek test hedef boyutu: {y_test.shape}")

            # evaluate_model testi
            metrics_dict, y_test_abs_eval, y_pred_abs_eval = evaluate_model(model, X_test, y_test, y_pred, y_test_original_prices=last_known_close_prices_for_test)
            logger.info(f"Model Değerlendirme Metrikleri: {metrics_dict}")

            # save_model testi
            save_status = save_model(model, scaler, features_cols)
            logger.info(f"Model kaydetme durumu: {save_status}")

            # load_model testi
            loaded_model, loaded_scaler, loaded_features_cols, load_status = load_model()
            logger.info(f"Model yükleme durumu: {load_status}")
            if loaded_model and loaded_scaler and loaded_features_cols:
                logger.info("Model, scaler ve özellikler başarıyla yüklendi.")
                # predict_next_day_price testi
                if not X_test_raw.empty:
                    last_processed_data_point = X_test_raw.iloc[[-1]]
                    predicted_price = predict_next_day_price(loaded_model, loaded_scaler, last_processed_data_point, loaded_features_cols)
                    logger.info(f"Sonraki {prediction_days_test} Günlük Tahmin (Örnek): ${predicted_price:.2f}")
                else:
                    logger.warning("X_test_raw boş, son veri noktası tahmin edilemedi.")
            else:
                logger.error("Kaydedilmiş model yüklenemedi.")
        else:
            logger.error(f"Model eğitilemedi. Durum: {train_status}")
    else:
        logger.error(f"Model testi için yeterli örnek veri hazırlanamadı. Durum: {prep_status}")
