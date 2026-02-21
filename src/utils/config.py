"""
config.py — Центральный конфигурационный файл проекта

Все пути, имена столбцов, гиперпараметры и константы хранятся здесь.
Измените под свои данные, не трогая логику скриптов.
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════════
# ПУТИ
# ══════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parents[2]  # корень проекта

DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS        = ROOT / "outputs"
MODELS_DIR     = OUTPUTS / "models"
PLOTS_DIR      = OUTPUTS / "plots"

# Имена входных файлов (положите в data/raw/)
RAW_FILES = {
    2015: DATA_RAW / "vtd_2015.csv",
    2020: DATA_RAW / "vtd_2020.csv",
    2022: DATA_RAW / "vtd_2022.csv",
    2024: DATA_RAW / "vtd_2024.csv",
}

# Имена обработанных файлов
MERGE_TABLE    = DATA_PROCESSED / "merge_full.csv"
ML_READY_GB    = DATA_PROCESSED / "ds_gbrf.csv"
ML_READY_LSTM  = DATA_PROCESSED / "ds_lstm.csv"
ML_READY_MLP   = DATA_PROCESSED / "ds_mlp.csv"
ML_READY_ISO   = DATA_PROCESSED / "ds_iso.csv"

# ══════════════════════════════════════════════════════════════════
# НАЗВАНИЯ СТОЛБЦОВ В ИСХОДНЫХ ДАННЫХ
# Измените под реальные заголовки ваших CSV
# ══════════════════════════════════════════════════════════════════

# Обязательные столбцы (без суффикса года)
COL_DIST_REF  = "dist_ref"      # расстояние от начала, мм (единый ref для матчинга)
COL_DIST_M    = "dist_m"        # расстояние, м (альтернатива)
COL_PIPE_NUM  = "pipe_num"      # номер трубы
COL_DEPTH     = "depth_pct"     # глубина дефекта, % от толщины стенки
COL_LEN       = "len_mm"        # длина дефекта, мм
COL_WID       = "wid_mm"        # ширина дефекта, мм
COL_DIST_LW   = "dist_lw"       # расстояние до продольного шва, мм
COL_DIST_CW   = "dist_cw"       # расстояние до кольцевого шва, мм
COL_TUBE_LEN  = "tube_len"      # длина трубы, мм
COL_KBD       = "kbd"           # коэффициент безопасного давления
COL_PF        = "pf"            # давление разрушения, МПа
COL_DANGER    = "danger"        # категория опасности: (a), (b), (c)

# Возможные альтернативные имена в ваших данных (для автоопределения)
COL_ALIASES = {
    COL_DEPTH:    ["Глубина дефекта, % стенки", "depth_percent", "depth_%"],
    COL_LEN:      ["Длина, мм", "length_mm", "len"],
    COL_WID:      ["Ширина, мм", "width_mm", "wid"],
    COL_DIST_LW:  ["Минимальное расстояние до продольного шва, мм", "dist_to_lw"],
    COL_DIST_CW:  ["Минимальное расстояние до кольцевого шва, мм", "dist_to_cw"],
    COL_PIPE_NUM: ["Номер трубы", "pipe_number", "pipe_id"],
    COL_KBD:      ["КБД", "kbd_value", "safety_factor"],
    COL_PF:       ["Pf, МПа", "failure_pressure", "pf_mpa"],
    COL_DANGER:   ["Категория", "danger_class", "category"],
}

# ══════════════════════════════════════════════════════════════════
# ПАРАМЕТРЫ АНАЛИЗА
# ══════════════════════════════════════════════════════════════════

# Годы пробегов ВТД
INSPECTION_YEARS = [2015, 2020, 2022, 2024]

# Базовый год для матчинга
BASE_YEAR = 2020

# Год-цель для предсказания
TARGET_YEAR = 2024

# Временны́е метки в годах от начала (2015=0)
YEAR_TO_ELAPSED = {2015: 0, 2020: 5, 2022: 7, 2024: 9}

# Параметры алгоритма матчинга
MATCH_DISTANCE_TOLERANCE_M = 1.0    # ±1 метр допуск по расстоянию
MATCH_REQUIRE_PIPE_NUM    = True     # требовать совпадения номера трубы

# Горячая зона
HOTSPOT_START_M = 136_000            # начало зоны, мм (136 км)
HOTSPOT_END_M   = 145_000            # конец зоны, мм (145 км)

# Пороги классификации опасности (MAOP = 7.35 МПа для данного трубопровода)
MAOP_MPA = 7.35
DANGER_A_RATIO = 1.1                 # Pf ≤ 1.1 × MAOP → категория (a)
DANGER_B_RATIO = 1.4                 # Pf ≤ 1.4 × MAOP → категория (b)

# Пороговые глубины дефектов
DEPTH_CRITICAL_PCT  = 40.0           # критический порог
DEPTH_WARNING_PCT   = 20.0           # предупреждающий порог
DEPTH_MIN_DETECTABLE = 5.0           # минимальная детектируемая глубина ВТД
DEPTH_VALID_MAX     = 80.0           # максимально возможная глубина

# ══════════════════════════════════════════════════════════════════
# ПРИЗНАКИ ДЛЯ ML-МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════

TARGET_COL = f"{COL_DEPTH}_{TARGET_YEAR}"

# Признаки для GradientBoosting и RandomForest
FEATURES_GBRF = [
    f"{COL_DEPTH}_2015",
    f"{COL_DEPTH}_2020",
    f"{COL_DEPTH}_2022",
    "delta_15_20",       # depth_2020 - depth_2015
    "delta_20_22",       # depth_2022 - depth_2020
    "speed_15_22",       # delta_15_20 / 5  (%/год)
    "speed_20_22",       # delta_20_22 / 2  (%/год)
    f"log_{COL_LEN}_20",
    f"log_{COL_WID}_20",
    f"log_{COL_DIST_LW}_20",
    f"log_{COL_DIST_CW}_20",
    f"log_{COL_LEN}_22",
    f"log_{COL_WID}_22",
    f"{COL_TUBE_LEN}_2020",
    "dist_km",
    "in_hotzone",
]

# Признаки для MLP (расширенный набор)
FEATURES_MLP = FEATURES_GBRF + [
    f"{COL_KBD}_20_clean",
    f"{COL_PF}_20_clean",
    f"{COL_KBD}_miss",
    "depth_accel",       # speed_20_22 - speed_15_22
    "len_change",        # log(len_22) - log(len_20)
    "wid_change",        # log(wid_22) - log(wid_20)
]

# Последовательные признаки для RNN (per timestep)
SEQ_FEATURE_COLS = [COL_DEPTH, COL_LEN, COL_WID, COL_DIST_LW, COL_DIST_CW]
SEQ_YEARS        = [2015, 2020, 2022]   # входные шаги
SEQ_N_TIME_FEATS = len(SEQ_FEATURE_COLS) + 1  # +1 для нормированного времени

# Признаки для Isolation Forest (все 4 года × 5 измерений)
ISO_YEARS = [2015, 2020, 2022, 2024]
ISO_FEATURE_COLS = [COL_DEPTH, COL_LEN, COL_WID, COL_DIST_LW, COL_DIST_CW]

# ══════════════════════════════════════════════════════════════════
# ГИПЕРПАРАМЕТРЫ МОДЕЛЕЙ (лучшие из grid search)
# ══════════════════════════════════════════════════════════════════

GB_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=5,
    validation_fraction=0.1,
    n_iter_no_change=20,
    tol=1e-4,
    random_state=42,
)

RF_PARAMS = dict(
    n_estimators=400,
    max_depth=8,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
)

MLP_PARAMS = dict(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    alpha=0.001,
    learning_rate_init=5e-4,
    learning_rate="adaptive",
    max_iter=600,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42,
)

RNN_PARAMS = dict(
    hidden_dim=24,
    lr_initial=8e-4,
    lr_decay=3e-4,
    lr_decay_epoch=50,
    clip=1.0,
    n_epochs=100,
    batch_size=32,
    seed=42,
)

ISO_PARAMS = dict(
    n_estimators=400,
    contamination=0.05,
    random_state=42,
    n_jobs=-1,
)

# ══════════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════

MODEL_COLORS = {
    "GradientBoosting": "#2563EB",
    "RandomForest":     "#16A34A",
    "RNN":              "#D97706",
    "MLP":              "#7C3AED",
    "IsolationForest":  "#DC2626",
}

DANGER_COLORS = {
    "(a)": "#DC2626",
    "(b)": "#D97706",
    "(c)": "#16A34A",
}

RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5
