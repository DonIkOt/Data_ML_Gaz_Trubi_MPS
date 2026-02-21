"""
src/data/matching.py — Продольный матчинг дефектов между пробегами ВТД

Алгоритм двойного критерия:
  1. Близость по расстоянию: |dist_A - dist_B| ≤ TOLERANCE_M метров
  2. Совпадение номера трубы: pipe_num_A == pipe_num_B
  При конкурирующих совпадениях — выбирается ближайший.

Результат: широкая таблица, где каждая строка = один дефект,
прослеживаемый через несколько пробегов.

Использование:
    python src/data/matching.py
    
    from src.data.matching import match_years, build_merge_table
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    DATA_PROCESSED, INSPECTION_YEARS, BASE_YEAR,
    COL_DIST_REF, COL_PIPE_NUM,
    MATCH_DISTANCE_TOLERANCE_M, MATCH_REQUIRE_PIPE_NUM,
    MERGE_TABLE,
)


# ══════════════════════════════════════════════════════════════════
# Матчинг двух пробегов
# ══════════════════════════════════════════════════════════════════

def match_two_years(
    df_base: pd.DataFrame,
    df_other: pd.DataFrame,
    year_base: int,
    year_other: int,
    tolerance_m: float = MATCH_DISTANCE_TOLERANCE_M,
    require_pipe_match: bool = MATCH_REQUIRE_PIPE_NUM,
) -> pd.DataFrame:
    """
    Сопоставляет дефекты двух пробегов.

    Parameters
    ----------
    df_base : pd.DataFrame
        Базовый пробег (строки — источник)
    df_other : pd.DataFrame
        Сопоставляемый пробег
    year_base, year_other : int
        Годы пробегов
    tolerance_m : float
        Допуск по расстоянию в метрах
    require_pipe_match : bool
        Требовать совпадения номера трубы

    Returns
    -------
    pd.DataFrame
        Таблица совпадений с колонками match_idx_base, match_idx_other, dist_diff_m
    """
    tolerance_mm = tolerance_m * 1000  # перевод в мм, если dist_ref в мм

    base_pos  = df_base[COL_DIST_REF].values
    other_pos = df_other[COL_DIST_REF].values
    
    base_pipe  = df_base[COL_PIPE_NUM].astype(str).values if COL_PIPE_NUM in df_base.columns else None
    other_pipe = df_other[COL_PIPE_NUM].astype(str).values if COL_PIPE_NUM in df_other.columns else None

    matches = []

    for i, pos_b in enumerate(base_pos):
        # Найти кандидатов в диапазоне допуска
        dist_diff = np.abs(other_pos - pos_b)
        candidates = np.where(dist_diff <= tolerance_mm)[0]

        if len(candidates) == 0:
            continue

        # Фильтр по номеру трубы
        if require_pipe_match and base_pipe is not None and other_pipe is not None:
            pipe_match = other_pipe[candidates] == base_pipe[i]
            candidates = candidates[pipe_match]

        if len(candidates) == 0:
            continue

        # Выбор ближайшего
        best_j = candidates[np.argmin(dist_diff[candidates])]
        matches.append({
            "idx_base":    i,
            "idx_other":   best_j,
            "dist_diff_m": dist_diff[best_j] / 1000,
        })

    result = pd.DataFrame(matches)
    
    # Убрать дублирующиеся совпадения (один other → несколько base)
    if len(result) > 0:
        result = result.sort_values("dist_diff_m")
        result = result.drop_duplicates(subset=["idx_other"], keep="first")
        result = result.sort_values("idx_base").reset_index(drop=True)

    match_rate = len(result) / len(df_base) * 100 if len(df_base) > 0 else 0
    print(f"  {year_base} ↔ {year_other}: {len(result)} совпадений из {len(df_base)} "
          f"({match_rate:.1f}%), ср. Δдист = {result['dist_diff_m'].mean():.3f} м")

    return result


# ══════════════════════════════════════════════════════════════════
# Построение продольной мерж-таблицы
# ══════════════════════════════════════════════════════════════════

def build_merge_table(
    datasets: dict[int, pd.DataFrame],
    base_year: int = BASE_YEAR,
) -> pd.DataFrame:
    """
    Строит широкую таблицу: каждая строка = один дефект,
    все пробеги — в столбцах с суффиксом года.

    Parameters
    ----------
    datasets : dict[int, pd.DataFrame]
        {год: датафрейм} — результат load_all_years()
    base_year : int
        Год, относительно которого строится матчинг (обычно 2020)

    Returns
    -------
    pd.DataFrame
        Широкая таблица (2,000–2,300 строк × 80–90 столбцов)
    """
    print("=" * 60)
    print(f"Построение мерж-таблицы (базовый год: {base_year})")
    print("=" * 60)

    if base_year not in datasets:
        raise ValueError(f"Базовый год {base_year} отсутствует в datasets")

    df_base = datasets[base_year].copy()
    
    # Добавить суффикс базового года ко всем столбцам (кроме dist_ref, pipe_num)
    id_cols = [COL_DIST_REF, COL_PIPE_NUM]
    base_cols = {c: f"{c}_{base_year}" for c in df_base.columns if c not in id_cols}
    df_merged = df_base.rename(columns=base_cols)

    # Матчинг с каждым другим годом
    other_years = [y for y in sorted(datasets.keys()) if y != base_year]

    for year in other_years:
        df_other = datasets[year]
        match_df  = match_two_years(df_base, df_other, base_year, year)

        if len(match_df) == 0:
            print(f"  ВНИМАНИЕ: нет совпадений с {year}")
            continue

        # Извлечь строки из df_other по индексам совпадений
        matched_other = df_other.iloc[match_df["idx_other"].values].copy()
        matched_other.index = df_base.iloc[match_df["idx_base"].values].index

        # Добавить суффикс года
        other_cols = {c: f"{c}_{year}" for c in matched_other.columns if c not in id_cols}
        matched_other = matched_other.rename(columns=other_cols)

        # Объединить в мерж-таблицу
        df_merged = df_merged.join(matched_other, how="left")
        print(f"  Добавлен {year}: {len(match_df)} строк совмещено")

    # Производные столбцы
    df_merged = _add_derived_columns(df_merged, base_year, other_years)

    print(f"\nМерж-таблица: {df_merged.shape[0]} строк × {df_merged.shape[1]} столбцов")
    return df_merged


def _add_derived_columns(
    df: pd.DataFrame,
    base_year: int,
    other_years: list[int],
) -> pd.DataFrame:
    """Добавляет производные столбцы: дельты, скорости, флаги."""
    from src.utils.config import HOTSPOT_START_M, HOTSPOT_END_M

    df = df.copy()

    # Горячая зона
    df["in_hotzone"] = df[COL_DIST_REF].between(HOTSPOT_START_M, HOTSPOT_END_M).astype(int)
    df["dist_km"]    = df[COL_DIST_REF] / 1_000_000  # мм → км

    # Дельты глубины между годами
    from src.utils.config import COL_DEPTH, YEAR_TO_ELAPSED
    years_sorted = sorted([base_year] + other_years)

    for i in range(len(years_sorted) - 1):
        y1, y2 = years_sorted[i], years_sorted[i + 1]
        col1 = f"{COL_DEPTH}_{y1}"
        col2 = f"{COL_DEPTH}_{y2}"
        if col1 in df.columns and col2 in df.columns:
            dt = YEAR_TO_ELAPSED[y2] - YEAR_TO_ELAPSED[y1]
            df[f"delta_{y1}_{y2}"] = df[col2] - df[col1]
            df[f"speed_{y1}_{y2}"] = df[f"delta_{y1}_{y2}"] / dt if dt > 0 else np.nan

    # Признак роста (цель для классификационной формулировки)
    target_year = max(years_sorted)
    col_target = f"{COL_DEPTH}_{target_year}"
    col_base   = f"{COL_DEPTH}_{base_year}"
    if col_target in df.columns and col_base in df.columns:
        df["depth_increased"] = (df[col_target] > df[col_base]).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Загрузить очищенные данные
    datasets = {}
    for year in INSPECTION_YEARS:
        path = DATA_PROCESSED / f"vtd_{year}_clean.csv"
        if path.exists():
            datasets[year] = pd.read_csv(path, low_memory=False)
            print(f"Загружен {year}: {len(datasets[year])} строк")
        else:
            print(f"ПРЕДУПРЕЖДЕНИЕ: {path} не найден. Сначала запустите loader.py")

    if len(datasets) < 2:
        print("Недостаточно данных для матчинга (нужно минимум 2 года)")
        sys.exit(1)

    merge_table = build_merge_table(datasets)

    # Сохранение
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    merge_table.to_csv(MERGE_TABLE, index=False)
    print(f"\nМерж-таблица сохранена: {MERGE_TABLE}")

    # Краткая статистика
    print("\nПервые строки:")
    print(merge_table.head(3).to_string())
    print("\nПропуски по ключевым столбцам:")
    key_cols = [c for c in merge_table.columns if "depth_pct" in c]
    print(merge_table[key_cols].isnull().sum())
