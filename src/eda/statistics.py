"""
src/eda/statistics.py — Описательная статистика по данным ВТД

Использование:
    python src/eda/statistics.py
    from src.eda.statistics import describe_all_years, plot_depth_distribution
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    MERGE_TABLE, PLOTS_DIR, INSPECTION_YEARS,
    COL_DEPTH, COL_LEN, COL_WID, COL_DIST_REF, COL_DANGER
)


def describe_all_years(df: pd.DataFrame) -> pd.DataFrame:
    """Сводная статистика глубины по всем годам."""
    rows = []
    for yr in INSPECTION_YEARS:
        col = f"{COL_DEPTH}_{yr}"
        if col not in df.columns:
            continue
        d = df[col].dropna()
        rows.append({
            "Год": yr, "N": len(d),
            "Среднее, %": round(d.mean(), 2),
            "Медиана, %": round(d.median(), 2),
            "Макс, %": round(d.max(), 2),
            "СКО, %": round(d.std(), 2),
        })
    return pd.DataFrame(rows)


def plot_depth_distribution(df: pd.DataFrame, save_path: Path = None):
    """Гистограммы распределения глубины по годам."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {2015: '#94A3B8', 2020: '#2563EB', 2022: '#16A34A', 2024: '#DC2626'}
    for yr, clr in colors.items():
        col = f"{COL_DEPTH}_{yr}"
        if col in df.columns:
            d = df[col].dropna()
            ax.hist(d, bins=25, alpha=0.55, color=clr, label=f"{yr} (n={len(d)})",
                    density=True, edgecolor='white', lw=0.5)
    ax.axvline(20, color='orange', ls='--', lw=1.5, label='20% — предупреждение')
    ax.axvline(40, color='red', ls='--', lw=1.5, label='40% — критический')
    ax.set_xlabel("Глубина дефекта, % от толщины стенки", fontsize=11)
    ax.set_ylabel("Плотность", fontsize=11)
    ax.set_title("Распределение глубины дефектов по годам", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', facecolor='white')
    return fig


if __name__ == "__main__":
    if not MERGE_TABLE.exists():
        print("Мерж-таблица не найдена. Запустите matching.py")
        sys.exit(1)
    df = pd.read_csv(MERGE_TABLE, low_memory=False)
    stats = describe_all_years(df)
    print("Статистика глубины по годам:")
    print(stats.to_string(index=False))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_depth_distribution(df, PLOTS_DIR / "depth_distribution.png")
    print(f"График сохранён в {PLOTS_DIR}/depth_distribution.png")
