"""
src/models/data_prep.py — Модельно-специфичная подготовка данных

Каждая модель требует своего формата данных:
  • GradientBoosting / RandomForest → широкая таблица с дельта-признаками
  • RNN                              → 3D тензор (n, timesteps, features)
  • MLP                              → широкая таблица + RobustScaler
  • IsolationForest                  → матрица 4 года × 5 признаков

Использование:
    from src.models.data_prep import prepare_all_datasets, DatasetBundle
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    MERGE_TABLE, DATA_PROCESSED,
    COL_DEPTH, COL_LEN, COL_WID, COL_DIST_LW, COL_DIST_CW,
    COL_KBD, COL_PF, COL_TUBE_LEN, COL_DIST_REF,
    FEATURES_GBRF, FEATURES_MLP, TARGET_COL,
    SEQ_FEATURE_COLS, SEQ_YEARS, SEQ_N_TIME_FEATS, YEAR_TO_ELAPSED,
    ISO_FEATURE_COLS, ISO_YEARS,
    HOTSPOT_START_M, HOTSPOT_END_M,
    RANDOM_STATE, TEST_SIZE,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# ══════════════════════════════════════════════════════════════════
# Контейнер для датасетов
# ══════════════════════════════════════════════════════════════════

@dataclass
class ModelDataset:
    """Датасет для одной модели."""
    X_train: np.ndarray
    X_test:  np.ndarray
    y_train: np.ndarray
    y_test:  np.ndarray
    feature_names: list[str]
    scaler: Optional[object] = None
    extra: Optional[dict]    = None  # доп. данные (y_scaler для RNN и т.д.)


# ══════════════════════════════════════════════════════════════════
# Шаг 1: Очистка и базовые признаки (общие для всех)
# ══════════════════════════════════════════════════════════════════

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает мерж-таблицу и создаёт производные признаки.
    Применяется до разделения по моделям.
    """
    df = df.copy()

    # ── Исправление физически невозможных значений ────────────────
    if "kbd_2020" in df.columns:
        df["kbd_2020"] = df["kbd_2020"].where(df["kbd_2020"] < 50, np.nan)
    if "pf_2020" in df.columns:
        df["pf_2020"] = df["pf_2020"].where(df["pf_2020"] > 0, np.nan)
    for yr in [2015, 2020, 2022, 2024]:
        col = f"{COL_DEPTH}_{yr}"
        if col in df.columns:
            df[col] = df[col].where(df[col].between(5, 80), np.nan)

    # ── Пространственные признаки ─────────────────────────────────
    df["dist_km"]   = df[COL_DIST_REF] / 1_000_000
    df["in_hotzone"] = df[COL_DIST_REF].between(
        HOTSPOT_START_M, HOTSPOT_END_M
    ).astype(int)

    # ── Дельта и скоростные признаки ─────────────────────────────
    d15 = f"{COL_DEPTH}_2015"
    d20 = f"{COL_DEPTH}_2020"
    d22 = f"{COL_DEPTH}_2022"
    if all(c in df.columns for c in [d15, d20, d22]):
        df["delta_15_20"] = df[d20] - df[d15]
        df["delta_20_22"] = df[d22] - df[d20]
        df["speed_15_22"] = df["delta_15_20"] / 5   # %/год
        df["speed_20_22"] = df["delta_20_22"] / 2   # %/год
        df["depth_accel"] = df["speed_20_22"] - df["speed_15_22"]
        df[f"depth_ratio_20_22"] = df[d22] / df[d20].replace(0, np.nan).fillna(1)

    # ── Логарифмические признаки геометрии ───────────────────────
    for yr in [2020, 2022]:
        for base_col in [COL_LEN, COL_WID, COL_DIST_LW, COL_DIST_CW]:
            col = f"{base_col}_{yr}"
            if col in df.columns:
                df[f"log_{base_col}_{yr}"] = np.log1p(df[col].clip(lower=0))

    # ── Изменение геометрии 2020→2022 ────────────────────────────
    for base_col in [COL_LEN, COL_WID]:
        c20 = f"log_{base_col}_2020"
        c22 = f"log_{base_col}_2022"
        if c20 in df.columns and c22 in df.columns:
            df[f"{base_col}_change"] = df[c22].fillna(0) - df[c20].fillna(0)

    # ── MNAR флаги (признак отсутствия = информация) ──────────────
    if f"{COL_KBD}_2020" in df.columns:
        df[f"{COL_KBD}_20_clean"] = df[f"{COL_KBD}_2020"].where(
            df[f"{COL_KBD}_2020"] < 50
        )
        df[f"{COL_KBD}_miss"] = df[f"{COL_KBD}_20_clean"].isna().astype(int)
    if f"{COL_PF}_2020" in df.columns:
        df[f"{COL_PF}_20_clean"] = df[f"{COL_PF}_2020"].where(
            df[f"{COL_PF}_2020"] > 0
        )

    return df


# ══════════════════════════════════════════════════════════════════
# Шаг 2: Подготовка датасета для GradientBoosting / RandomForest
# ══════════════════════════════════════════════════════════════════

def prepare_gbrf(df: pd.DataFrame, extra_features: list = None) -> ModelDataset:
    """
    Широкая таблица: 3 исторических значения + производные признаки.
    Без масштабирования (деревья инвариантны).
    """
    features = FEATURES_GBRF.copy()
    if extra_features:
        features = features + extra_features

    # Фильтр: нужны depth_2020 и depth_2022 и target
    required = [f"{COL_DEPTH}_2020", f"{COL_DEPTH}_2022", TARGET_COL]
    mask = df[required].notna().all(axis=1)
    df_clean = df[mask].copy()

    # Подмена NaN в признаках медианами (только для 2015 могут быть пропуски)
    available_feats = [f for f in features if f in df_clean.columns]
    for f in available_feats:
        df_clean[f] = df_clean[f].fillna(df_clean[f].median())

    X = df_clean[available_feats].values
    y = df_clean[TARGET_COL].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"  GB/RF: {X.shape[0]} объектов, {X.shape[1]} признаков")
    return ModelDataset(X_tr, X_te, y_tr, y_te, available_feats)


# ══════════════════════════════════════════════════════════════════
# Шаг 3: Подготовка датасета для RNN (последовательный формат)
# ══════════════════════════════════════════════════════════════════

def prepare_rnn(df: pd.DataFrame) -> ModelDataset:
    """
    Строит 3D тензор (n, timesteps, features) для RNN.
    
    Формат каждого временного шага:
      [depth_norm, log_len_norm, log_wid_norm, log_lw_norm, log_cw_norm, t_elapsed_norm]
    
    Нормализация: MinMaxScaler [-1, 1] по каждому каналу независимо.
    """
    years   = SEQ_YEARS
    t_el    = [YEAR_TO_ELAPSED[y] for y in years]
    t_max   = max(YEAR_TO_ELAPSED.values())

    # Нужны все 3 входных года + target
    req_cols = []
    for yr in years + [2024]:
        for fc in SEQ_FEATURE_COLS:
            req_cols.append(f"{fc}_{yr}")
    req_cols.append(TARGET_COL)

    mask = df[req_cols].notna().all(axis=1)
    df_clean = df[mask].copy()

    n = len(df_clean)
    T = len(years)
    F = len(SEQ_FEATURE_COLS) + 1  # +1 для нормированного времени

    # Построение сырого тензора
    X_raw = np.zeros((n, T, F))
    for ti, (yr, t) in enumerate(zip(years, t_el)):
        X_raw[:, ti, 0] = df_clean[f"{COL_DEPTH}_{yr}"].values
        X_raw[:, ti, 1] = np.log1p(df_clean[f"{COL_LEN}_{yr}"].values.clip(0))
        X_raw[:, ti, 2] = np.log1p(df_clean[f"{COL_WID}_{yr}"].values.clip(0))
        X_raw[:, ti, 3] = np.log1p(df_clean[f"{COL_DIST_LW}_{yr}"].values.clip(0))
        X_raw[:, ti, 4] = np.log1p(df_clean[f"{COL_DIST_CW}_{yr}"].values.clip(0))
        X_raw[:, ti, 5] = t / t_max  # нормированное время [0, 1]

    y = df_clean[TARGET_COL].values

    # Разбивка до нормализации (чтобы не было утечки)
    idx = np.arange(n)
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_tr_raw, X_te_raw = X_raw[idx_tr], X_raw[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    # MinMaxScaler по каждому каналу (кроме последнего — время уже нормировано)
    scalers = []
    X_tr = X_tr_raw.copy(); X_te = X_te_raw.copy()
    for fi in range(F - 1):
        sc = MinMaxScaler(feature_range=(-1, 1))
        flat_tr = X_tr_raw[:, :, fi].reshape(-1, 1)
        sc.fit(flat_tr)
        X_tr[:, :, fi] = sc.transform(X_tr_raw[:, :, fi].reshape(-1, 1)).reshape(-1, T)
        X_te[:, :, fi] = sc.transform(X_te_raw[:, :, fi].reshape(-1, 1)).reshape(-1, T)
        scalers.append(sc)

    # Нормализация цели
    y_min, y_range = y_tr.min(), y_tr.max() - y_tr.min()
    y_tr_norm = (y_tr - y_min) / y_range
    # y_te НЕ нормируем — модель будет выдавать нормированное, мы денормируем

    feat_names = [f"{fc}_{y}" for y in years for fc in SEQ_FEATURE_COLS] + ["t_elapsed"] * T

    print(f"  RNN: {n} объектов, shape {X_tr.shape}")
    return ModelDataset(
        X_train=X_tr, X_test=X_te,
        y_train=y_tr_norm, y_test=y_te,
        feature_names=feat_names,
        extra={"y_min": y_min, "y_range": y_range,
               "feature_scalers": scalers,
               "X_train_raw": X_tr_raw, "X_test_raw": X_te_raw}
    )


# ══════════════════════════════════════════════════════════════════
# Шаг 4: Подготовка датасета для MLP
# ══════════════════════════════════════════════════════════════════

def prepare_mlp(df: pd.DataFrame) -> ModelDataset:
    """
    Широкая таблица с расширенным набором признаков + RobustScaler.
    """
    required = [f"{COL_DEPTH}_2020", f"{COL_DEPTH}_2022", TARGET_COL]
    mask = df[required].notna().all(axis=1)
    df_clean = df[mask].copy()

    available_feats = [f for f in FEATURES_MLP if f in df_clean.columns]
    for f in available_feats:
        df_clean[f] = df_clean[f].fillna(df_clean[f].median())

    X = df_clean[available_feats].values
    y = df_clean[TARGET_COL].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print(f"  MLP: {X.shape[0]} объектов, {X.shape[1]} признаков (RobustScaler)")
    return ModelDataset(X_tr_s, X_te_s, y_tr, y_te, available_feats, scaler=scaler)


# ══════════════════════════════════════════════════════════════════
# Шаг 5: Подготовка датасета для IsolationForest
# ══════════════════════════════════════════════════════════════════

def prepare_isolation_forest(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Матрица 4 года × 5 признаков = 20 переменных.
    Возвращает X_all, labels (true_risky, growth_label), feature_names.
    """
    feature_names = []
    df_iso = df.copy()

    # Требуем хотя бы 2020 и 2024
    required = [f"{COL_DEPTH}_2020", f"{COL_DEPTH}_2024"]
    mask = df_iso[required].notna().all(axis=1)
    df_iso = df_iso[mask].copy()

    X_parts = []
    for yr in ISO_YEARS:
        for fc in ISO_FEATURE_COLS:
            col = f"{fc}_{yr}"
            feat_name = f"{fc}_{yr}"
            if col in df_iso.columns:
                vals = df_iso[col].fillna(df_iso[col].median()).values
            else:
                vals = np.zeros(len(df_iso))
            X_parts.append(vals)
            feature_names.append(feat_name)

    X_all = np.column_stack(X_parts)

    # Метки для валидации (не для обучения)
    true_risky = (
        df_iso.get("danger_2020", pd.Series(dtype=str)).isin(["(a)", "(b)"]) |
        df_iso.get("danger_2022", pd.Series(dtype=str)).isin(["(a)", "(b)"]) |
        df_iso.get("danger_2024", pd.Series(dtype=str)).isin(["(a)", "(b)"])
    ).astype(int).values

    d_delta = df_iso[f"{COL_DEPTH}_2024"].values - df_iso[f"{COL_DEPTH}_2020"].values
    growth_label = (d_delta > 2).astype(int)

    # "Нормальные" объекты — для обучения IsoForest
    normal_mask = (true_risky == 0) & (growth_label == 0)

    print(f"  IsoForest: {len(X_all)} объектов, {X_all.shape[1]} признаков")
    print(f"    Опасных (true_risky): {true_risky.sum()} | Рост>2%: {growth_label.sum()}")
    print(f"    Нормальных (train):   {normal_mask.sum()}")

    return X_all, X_all[normal_mask], {"true_risky": true_risky,
                                        "growth_label": growth_label,
                                        "normal_mask": normal_mask,
                                        "dist_ref": df_iso[COL_DIST_REF].values,
                                        "feature_names": feature_names}


# ══════════════════════════════════════════════════════════════════
# Главная функция — подготовка всех датасетов
# ══════════════════════════════════════════════════════════════════

def prepare_all_datasets(df: pd.DataFrame | None = None) -> dict:
    """
    Запускает полный пайплайн подготовки данных для всех моделей.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Мерж-таблица. Если None — загружается из MERGE_TABLE.

    Returns
    -------
    dict с ключами: 'gbrf', 'rf', 'rnn', 'mlp', 'iso'
    """
    if df is None:
        if not MERGE_TABLE.exists():
            raise FileNotFoundError(
                f"{MERGE_TABLE} не найден. Запустите сначала matching.py"
            )
        df = pd.read_csv(MERGE_TABLE, low_memory=False)
        print(f"Загружена мерж-таблица: {len(df)} строк")

    print("\nОчистка и инженерия признаков...")
    df = clean_and_engineer(df)

    print("\nПодготовка датасетов:")
    datasets = {}

    print("\n[1/5] Gradient Boosting / Random Forest:")
    datasets["gbrf"] = prepare_gbrf(df)

    print("\n[2/5] Random Forest (с доп. признаком depth_ratio):")
    datasets["rf"] = prepare_gbrf(df, extra_features=["depth_ratio_20_22"])

    print("\n[3/5] RNN (последовательный формат):")
    datasets["rnn"] = prepare_rnn(df)

    print("\n[4/5] MLP Neural Net:")
    datasets["mlp"] = prepare_mlp(df)

    print("\n[5/5] Isolation Forest:")
    X_all, X_normal, iso_meta = prepare_isolation_forest(df)
    datasets["iso"] = {"X_all": X_all, "X_normal": X_normal, "meta": iso_meta}

    print("\n✅ Все датасеты подготовлены")
    return datasets


# ══════════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    datasets = prepare_all_datasets()
    
    print("\nИтоговые размеры:")
    for name in ["gbrf", "rnn", "mlp"]:
        d = datasets[name]
        print(f"  {name}: train={d.X_train.shape}, test={d.X_test.shape}")
    
    iso = datasets["iso"]
    print(f"  iso: X_all={iso['X_all'].shape}, X_normal={iso['X_normal'].shape}")
