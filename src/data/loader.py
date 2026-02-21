"""
src/data/loader.py — Загрузка и первичная очистка данных ВТД

Загружает CSV-файлы каждого пробега, стандартизирует имена столбцов,
применяет базовую очистку и сохраняет унифицированные файлы.

Использование:
    python src/data/loader.py
    
    # Или из кода:
    from src.data.loader import load_inspection_year, load_all_years
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    RAW_FILES, DATA_PROCESSED, INSPECTION_YEARS,
    COL_ALIASES, COL_DEPTH, COL_LEN, COL_WID, COL_DIST_LW,
    COL_DIST_CW, COL_PIPE_NUM, COL_KBD, COL_PF, COL_DANGER,
    COL_DIST_REF, COL_TUBE_LEN,
    DEPTH_MIN_DETECTABLE, DEPTH_VALID_MAX
)


# ══════════════════════════════════════════════════════════════════
# Автоопределение имён столбцов
# ══════════════════════════════════════════════════════════════════

def _find_column(df: pd.DataFrame, target_name: str) -> str | None:
    """Ищет столбец по точному имени или по алиасам."""
    if target_name in df.columns:
        return target_name
    aliases = COL_ALIASES.get(target_name, [])
    for alias in aliases:
        if alias in df.columns:
            return alias
    # Нечёткий поиск (нижний регистр)
    target_lower = target_name.lower().replace("_", " ")
    for col in df.columns:
        if target_lower in col.lower():
            return col
    return None


def standardize_columns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Переименовывает столбцы в стандартные имена.
    Добавляет суффикс года к ключевым метрикам.
    """
    rename_map = {}
    standard_cols = {
        COL_DEPTH: COL_DEPTH,
        COL_LEN:   COL_LEN,
        COL_WID:   COL_WID,
        COL_DIST_LW: COL_DIST_LW,
        COL_DIST_CW: COL_DIST_CW,
        COL_PIPE_NUM: COL_PIPE_NUM,
        COL_KBD:   COL_KBD,
        COL_PF:    COL_PF,
        COL_DANGER: COL_DANGER,
        COL_TUBE_LEN: COL_TUBE_LEN,
        COL_DIST_REF: COL_DIST_REF,
    }
    
    for std_name in standard_cols:
        found = _find_column(df, std_name)
        if found and found != std_name:
            rename_map[found] = std_name
    
    df = df.rename(columns=rename_map)
    
    # Если dist_ref нет — попробуем вычислить из dist_m * 1000
    if COL_DIST_REF not in df.columns:
        dist_m_col = _find_column(df, "dist_m")
        if dist_m_col:
            df[COL_DIST_REF] = df[dist_m_col] * 1000
    
    return df


# ══════════════════════════════════════════════════════════════════
# Очистка данных
# ══════════════════════════════════════════════════════════════════

def clean_inspection_data(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Базовая очистка данных одного пробега:
    - Исправляет физически невозможные значения
    - Удаляет дублирующиеся записи
    - Приводит типы
    """
    df = df.copy()
    n_start = len(df)
    
    # 1. Глубина: допустимый диапазон [DEPTH_MIN_DETECTABLE, DEPTH_VALID_MAX]
    if COL_DEPTH in df.columns:
        invalid_depth = ~df[COL_DEPTH].between(DEPTH_MIN_DETECTABLE, DEPTH_VALID_MAX)
        n_invalid = invalid_depth.sum()
        if n_invalid > 0:
            print(f"  [{year}] Удалено {n_invalid} строк с недопустимой глубиной "
                  f"(вне [{DEPTH_MIN_DETECTABLE}, {DEPTH_VALID_MAX}]%)")
            df.loc[invalid_depth, COL_DEPTH] = np.nan
    
    # 2. КБД: физически невозможен > 50
    if COL_KBD in df.columns:
        invalid_kbd = df[COL_KBD] > 50
        df.loc[invalid_kbd, COL_KBD] = np.nan
        if invalid_kbd.sum() > 0:
            print(f"  [{year}] Обнулено {invalid_kbd.sum()} значений КБД > 50")
    
    # 3. Pf: не может быть ≤ 0
    if COL_PF in df.columns:
        df.loc[df[COL_PF] <= 0, COL_PF] = np.nan
    
    # 4. Длина, ширина: не может быть отрицательной
    for col in [COL_LEN, COL_WID, COL_DIST_LW, COL_DIST_CW]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    
    # 5. Удаление явных дублей по позиции
    if COL_DIST_REF in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=[COL_DIST_REF, COL_PIPE_NUM], keep='first')
        n_dupl = n_before - len(df)
        if n_dupl > 0:
            print(f"  [{year}] Удалено {n_dupl} дублирующихся записей")
    
    # 6. Числовые типы
    numeric_cols = [COL_DEPTH, COL_LEN, COL_WID, COL_DIST_LW, COL_DIST_CW,
                    COL_KBD, COL_PF, COL_TUBE_LEN, COL_DIST_REF]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"  [{year}] Итого: {len(df)} строк (исходно {n_start})")
    return df


# ══════════════════════════════════════════════════════════════════
# Основные функции загрузки
# ══════════════════════════════════════════════════════════════════

def load_inspection_year(year: int, filepath: Path | None = None) -> pd.DataFrame:
    """
    Загружает данные одного пробега ВТД.
    
    Parameters
    ----------
    year : int
        Год пробега (2015, 2020, 2022, 2024)
    filepath : Path, optional
        Путь к файлу. По умолчанию берётся из config.RAW_FILES[year]
    
    Returns
    -------
    pd.DataFrame
        Очищенный датафрейм с стандартными именами столбцов
    """
    if filepath is None:
        filepath = RAW_FILES.get(year)
    
    if filepath is None or not Path(filepath).exists():
        raise FileNotFoundError(
            f"Файл для {year} не найден: {filepath}\n"
            f"Положите файл в data/raw/ и обновите config.RAW_FILES"
        )
    
    print(f"Загрузка {year}: {filepath}")
    
    # Попытка с разными кодировками
    for enc in ['utf-8', 'cp1251', 'latin-1']:
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Не удалось прочитать {filepath} ни с одной кодировкой")
    
    print(f"  [{year}] Загружено {len(df)} строк, {len(df.columns)} столбцов")
    
    df = standardize_columns(df, year)
    df = clean_inspection_data(df, year)
    df['year'] = year
    
    return df


def load_all_years(save: bool = True) -> dict[int, pd.DataFrame]:
    """
    Загружает данные всех пробегов.
    
    Returns
    -------
    dict[int, pd.DataFrame]
        Словарь {год: датафрейм}
    """
    print("=" * 60)
    print("Загрузка данных ВТД всех пробегов")
    print("=" * 60)
    
    datasets = {}
    for year in INSPECTION_YEARS:
        try:
            df = load_inspection_year(year)
            datasets[year] = df
        except FileNotFoundError as e:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: {e}")
            continue
    
    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        for year, df in datasets.items():
            out_path = DATA_PROCESSED / f"vtd_{year}_clean.csv"
            df.to_csv(out_path, index=False)
            print(f"  Сохранено: {out_path}")
    
    print(f"\nЗагружено пробегов: {len(datasets)}")
    return datasets


def get_summary_stats(datasets: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Сводная статистика по всем пробегам."""
    rows = []
    for year, df in datasets.items():
        row = {"year": year, "n_defects": len(df)}
        if COL_DEPTH in df.columns:
            row.update({
                "depth_mean": df[COL_DEPTH].mean(),
                "depth_median": df[COL_DEPTH].median(),
                "depth_max": df[COL_DEPTH].max(),
                "depth_std": df[COL_DEPTH].std(),
            })
        if COL_DANGER in df.columns:
            vc = df[COL_DANGER].value_counts()
            for cat in ['(a)', '(b)', '(c)']:
                row[f"danger_{cat}"] = vc.get(cat, 0)
        rows.append(row)
    
    return pd.DataFrame(rows).set_index("year")


# ══════════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    datasets = load_all_years(save=True)
    
    print("\n" + "=" * 60)
    print("СВОДНАЯ СТАТИСТИКА")
    print("=" * 60)
    stats = get_summary_stats(datasets)
    print(stats.to_string())
