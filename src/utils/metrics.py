"""
src/utils/metrics.py — Метрики и утилиты оценки моделей
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def full_regression_report(y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "") -> dict:
    """Полный набор регрессионных метрик."""
    residuals = y_pred - y_true
    return {
        "model": model_name,
        "r2":    r2_score(y_true, y_pred),
        "rmse":  np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae":   mean_absolute_error(y_true, y_pred),
        "bias":  residuals.mean(),             # систематическая ошибка
        "res_std": residuals.std(),            # разброс остатков
        "outliers_2pct": (np.abs(residuals) > 2).sum(),  # остатки > 2%
        "n_test": len(y_true),
    }


def compare_models(results: dict) -> pd.DataFrame:
    """Сводная таблица метрик нескольких моделей."""
    rows = []
    for name, r in results.items():
        if r.get("r2") is None:
            continue
        rows.append({
            "Модель": name,
            "R² Test": round(r.get("r2", 0), 4),
            "CV R²":   round(r.get("cv_r2", 0), 4) if r.get("cv_r2") else "—",
            "CV±":     round(r.get("cv_r2_std", 0), 4) if r.get("cv_r2_std") else "—",
            "RMSE, %": round(r.get("rmse", 0), 4),
            "MAE, %":  round(r.get("mae", 0), 4),
            "N train": r.get("n_train", "—"),
        })
    return pd.DataFrame(rows).sort_values("R² Test", ascending=False)
