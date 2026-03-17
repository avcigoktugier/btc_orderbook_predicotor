"""
============================================================
BTC/USDT Orderbook ML Prediction System - Model Modülü
============================================================
İkili sınıflandırma modelleri:
  1. XGBoost Classifier (Ana Model)
  2. Random Forest Classifier (Yedek/Karşılaştırma)
  3. Ensemble (İki modelin birleşik tahmini)

Fonksiyonlar:
  - Model eğitimi (train)
  - Model değerlendirme (evaluate)
  - Tahmin (predict)
  - Model kaydetme/yükleme (save/load)
  - Feature importance analizi
============================================================
"""

import logging
import os
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # GUI olmadan grafik üret
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import config

logger = logging.getLogger(__name__)


# ============================================================
# 1. MODEL SINIFI
# ============================================================

class OrderbookPredictor:
    """
    Orderbook verilerinden fiyat yönü tahmin eden ML modeli.
    
    Kullanım:
        predictor = OrderbookPredictor()
        predictor.train(X_train, y_train, X_val, y_val)
        predictions = predictor.predict(X_test)
        predictor.save()
    """
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Args:
            model_type: "xgboost", "random_forest", veya "ensemble"
        """
        self.model_type = model_type
        self.xgb_model: Optional[XGBClassifier] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.is_trained = False
        
        logger.info(f"OrderbookPredictor başlatıldı: {model_type}")
    
    def _build_xgboost(self) -> XGBClassifier:
        """XGBoost modelini config parametreleriyle oluştur."""
        return XGBClassifier(**config.XGBOOST_PARAMS)
    
    def _build_random_forest(self) -> RandomForestClassifier:
        """Random Forest modelini config parametreleriyle oluştur."""
        return RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
    
    def train(self, 
              X_train: pd.DataFrame, 
              y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict:
        """
        Modeli eğit.
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim hedefleri
            X_val: Validation özellikleri (early stopping için)
            y_val: Validation hedefleri
            
        Returns:
            Dict: Eğitim metrikleri
        """
        self.feature_names = list(X_train.columns)
        
        # Özellik ölçekleme (StandardScaler)
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_val is not None:
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
        
        results = {}
        
        # --- XGBoost Eğitimi ---
        if self.model_type in ("xgboost", "ensemble"):
            print("\n  XGBoost eğitiliyor...")
            self.xgb_model = self._build_xgboost()
            
            fit_params = {}
            if X_val is not None:
                fit_params["eval_set"] = [(X_val_scaled, y_val)]
                fit_params["verbose"] = False
            
            self.xgb_model.fit(X_train_scaled, y_train, **fit_params)
            
            # Eğitim metrikleri
            train_pred = self.xgb_model.predict(X_train_scaled)
            results["xgb_train_accuracy"] = accuracy_score(y_train, train_pred)
            
            if X_val is not None:
                val_pred = self.xgb_model.predict(X_val_scaled)
                results["xgb_val_accuracy"] = accuracy_score(y_val, val_pred)
                
            logger.info(f"XGBoost eğitimi tamamlandı. Train Acc: {results['xgb_train_accuracy']:.4f}")
        
        # --- Random Forest Eğitimi ---
        if self.model_type in ("random_forest", "ensemble"):
            print("  Random Forest eğitiliyor...")
            self.rf_model = self._build_random_forest()
            self.rf_model.fit(X_train_scaled, y_train)
            
            train_pred = self.rf_model.predict(X_train_scaled)
            results["rf_train_accuracy"] = accuracy_score(y_train, train_pred)
            
            if X_val is not None:
                val_pred = self.rf_model.predict(X_val_scaled)
                results["rf_val_accuracy"] = accuracy_score(y_val, val_pred)
                
            logger.info(f"Random Forest eğitimi tamamlandı. Train Acc: {results['rf_train_accuracy']:.4f}")
        
        self.is_trained = True
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Tahmin yap.
        
        Args:
            X: Özellik DataFrame'i
            
        Returns:
            np.ndarray: Tahminler (0 veya 1)
        """
        if not self.is_trained:
            raise RuntimeError("Model henüz eğitilmemiş! Önce train() çağırın.")
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        if self.model_type == "xgboost":
            return self.xgb_model.predict(X_scaled)
        elif self.model_type == "random_forest":
            return self.rf_model.predict(X_scaled)
        elif self.model_type == "ensemble":
            return self._ensemble_predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Olasılık tahmini yap.
        
        Args:
            X: Özellik DataFrame'i
            
        Returns:
            np.ndarray: Sınıf olasılıkları [[p_0, p_1], ...]
        """
        if not self.is_trained:
            raise RuntimeError("Model henüz eğitilmemiş!")
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        if self.model_type == "xgboost":
            return self.xgb_model.predict_proba(X_scaled)
        elif self.model_type == "random_forest":
            return self.rf_model.predict_proba(X_scaled)
        elif self.model_type == "ensemble":
            return self._ensemble_predict_proba(X_scaled)
    
    def _ensemble_predict(self, X_scaled: pd.DataFrame) -> np.ndarray:
        """İki modelin birleşik tahmini (soft voting)."""
        proba = self._ensemble_predict_proba(X_scaled)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def _ensemble_predict_proba(self, X_scaled: pd.DataFrame) -> np.ndarray:
        """İki modelin birleşik olasılık tahmini."""
        xgb_proba = self.xgb_model.predict_proba(X_scaled)
        rf_proba = self.rf_model.predict_proba(X_scaled)
        
        # Eşit ağırlıklı ortalama (soft voting)
        return (xgb_proba + rf_proba) / 2
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Model performansını değerlendir.
        
        Returns:
            Dict: Tüm metrikler
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred),
        }
        
        return metrics
    
    def print_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Detaylı değerlendirme raporu yazdır."""
        metrics = self.evaluate(X_test, y_test)
        
        print(f"\n{'='*60}")
        print(f"  MODEL DEĞERLENDİRME RAPORU ({self.model_type.upper()})")
        print(f"{'='*60}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"\n  Confusion Matrix:")
        cm = metrics["confusion_matrix"]
        print(f"    Tahmin →     0     1")
        print(f"    Gerçek 0: {cm[0][0]:>5} {cm[0][1]:>5}")
        print(f"    Gerçek 1: {cm[1][0]:>5} {cm[1][1]:>5}")
        print(f"\n{metrics['classification_report']}")
        print(f"{'='*60}")
        
        return metrics
    
    # ============================================================
    # FEATURE IMPORTANCE (Özellik Önemi)
    # ============================================================
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        En önemli özellikleri döndür.
        
        Args:
            top_n: İlk kaç özellik
            
        Returns:
            pd.DataFrame: Özellik adı ve önem skoru
        """
        if self.model_type in ("xgboost", "ensemble") and self.xgb_model:
            importance = self.xgb_model.feature_importances_
        elif self.model_type == "random_forest" and self.rf_model:
            importance = self.rf_model.feature_importances_
        else:
            return pd.DataFrame()
        
        imp_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False).head(top_n)
        
        return imp_df
    
    def plot_feature_importance(self, top_n: int = 20, 
                                save_path: str = None) -> str:
        """
        Feature importance grafiği çiz ve kaydet.
        
        Returns:
            str: Kaydedilen dosya yolu
        """
        imp_df = self.get_feature_importance(top_n)
        
        if imp_df.empty:
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            imp_df["feature"][::-1], 
            imp_df["importance"][::-1],
            color="#2196F3"
        )
        ax.set_xlabel("Önem Skoru (Importance)")
        ax.set_title(f"Top {top_n} Özellik Önemi ({self.model_type.upper()})")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(config.MODEL_DIR, "feature_importance.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Feature importance grafiği kaydedildi: {save_path}")
        return save_path
    
    # ============================================================
    # MODEL KAYDETME / YÜKLEME
    # ============================================================
    
    def save(self, prefix: str = None):
        """
        Eğitilmiş modeli, scaler'ı ve feature isimlerini kaydet.
        
        Args:
            prefix: Dosya adı öneki (None ise config'den)
        """
        if not self.is_trained:
            raise RuntimeError("Kaydedilecek eğitilmiş model yok!")
        
        if self.xgb_model:
            path = config.XGBOOST_MODEL_PATH
            joblib.dump(self.xgb_model, path)
            logger.info(f"XGBoost modeli kaydedildi: {path}")
        
        if self.rf_model:
            path = config.RF_MODEL_PATH
            joblib.dump(self.rf_model, path)
            logger.info(f"Random Forest modeli kaydedildi: {path}")
        
        if self.scaler:
            joblib.dump(self.scaler, config.SCALER_PATH)
            logger.info(f"Scaler kaydedildi: {config.SCALER_PATH}")
        
        if self.feature_names:
            joblib.dump(self.feature_names, config.FEATURE_NAMES_PATH)
            logger.info(f"Feature isimleri kaydedildi: {config.FEATURE_NAMES_PATH}")
    
    def load(self):
        """Kaydedilmiş modeli yükle."""
        try:
            if self.model_type in ("xgboost", "ensemble"):
                self.xgb_model = joblib.load(config.XGBOOST_MODEL_PATH)
                logger.info("XGBoost modeli yüklendi.")
            
            if self.model_type in ("random_forest", "ensemble"):
                self.rf_model = joblib.load(config.RF_MODEL_PATH)
                logger.info("Random Forest modeli yüklendi.")
            
            self.scaler = joblib.load(config.SCALER_PATH)
            self.feature_names = joblib.load(config.FEATURE_NAMES_PATH)
            self.is_trained = True
            
            logger.info(
                f"Model yüklendi. Özellik sayısı: {len(self.feature_names)}"
            )
            
        except FileNotFoundError as e:
            logger.error(f"Model dosyası bulunamadı: {e}")
            raise


# ============================================================
# 2. VERİ BÖLME (TIME-SERIES AWARE)
# ============================================================

def split_time_series_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = "target",
    test_size: float = None,
    val_size: float = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, 
           pd.DataFrame, pd.Series]:
    """
    Zaman serisi verisi için kronolojik bölme.
    
    ÖNEMLİ: Zaman serisinde rastgele bölme yapılMAZ!
    Veri kronolojik sırada bölünür (data leakage önlenir).
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if val_size is None:
        val_size = config.VALIDATION_SIZE
    
    n = len(df)
    test_start = int(n * (1 - test_size))
    val_start = int(test_start * (1 - val_size / (1 - test_size)))
    
    train_df = df.iloc[:val_start]
    val_df = df.iloc[val_start:test_start]
    test_df = df.iloc[test_start:]
    
    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_val = val_df[feature_columns]
    y_val = val_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]
    
    print(f"\n  Veri Bölme (Kronolojik):")
    print(f"    Train:      {len(X_train):>6} satır ({len(X_train)/n*100:.1f}%)")
    print(f"    Validation: {len(X_val):>6} satır ({len(X_val)/n*100:.1f}%)")
    print(f"    Test:       {len(X_test):>6} satır ({len(X_test)/n*100:.1f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# 3. CROSS-VALIDATION (Zaman Serisi)
# ============================================================

def time_series_cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    n_splits: int = 5,
) -> Dict:
    """
    Zaman serisi cross-validation.
    
    TimeSeriesSplit kullanır: Her fold'da eğitim seti büyür,
    test seti her zaman gelecek veridir.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = {"accuracy": [], "f1": [], "roc_auc": []}
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        predictor = OrderbookPredictor(model_type=model_type)
        predictor.train(X_train, y_train)
        
        y_pred = predictor.predict(X_test)
        y_proba = predictor.predict_proba(X_test)[:, 1]
        
        scores["accuracy"].append(accuracy_score(y_test, y_pred))
        scores["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        try:
            scores["roc_auc"].append(roc_auc_score(y_test, y_proba))
        except ValueError:
            scores["roc_auc"].append(0.5)
        
        logger.info(
            f"Fold {fold+1}: Acc={scores['accuracy'][-1]:.4f}, "
            f"F1={scores['f1'][-1]:.4f}"
        )
    
    result = {
        "mean_accuracy": np.mean(scores["accuracy"]),
        "std_accuracy": np.std(scores["accuracy"]),
        "mean_f1": np.mean(scores["f1"]),
        "std_f1": np.std(scores["f1"]),
        "mean_roc_auc": np.mean(scores["roc_auc"]),
        "std_roc_auc": np.std(scores["roc_auc"]),
        "all_scores": scores,
    }
    
    print(f"\n  Time Series CV Sonuçları ({n_splits} fold):")
    print(f"    Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
    print(f"    F1-Score: {result['mean_f1']:.4f} ± {result['std_f1']:.4f}")
    print(f"    ROC-AUC:  {result['mean_roc_auc']:.4f} ± {result['std_roc_auc']:.4f}")
    
    return result
