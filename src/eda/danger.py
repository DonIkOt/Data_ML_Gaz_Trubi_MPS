"""
src/eda/danger.py — Анализ категорий опасности дефектов

Использование:
    python src/eda/danger.py
    from src.eda.danger import DangerAnalyzer
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
    COL_DEPTH, COL_WID, COL_LEN, COL_DIST_CW, COL_KBD, COL_PF, COL_DANGER,
    MAOP_MPA
)


class DangerAnalyzer:
    """Анализ категорий опасности и их динамики."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def distribution_by_year(self) -> pd.DataFrame:
        rows = []
        for yr in INSPECTION_YEARS:
            col = f"{COL_DANGER}_{yr}"
            if col not in self.df.columns:
                continue
            vc = self.df[col].value_counts()
            total = vc.sum()
            rows.append({
                "Год": yr,
                "n_total": total,
                "(a)_n": vc.get("(a)", 0),
                "(b)_n": vc.get("(b)", 0),
                "(c)_n": vc.get("(c)", 0),
                "(a)_%": round(vc.get("(a)", 0) / total * 100, 2),
                "(b)_%": round(vc.get("(b)", 0) / total * 100, 2),
                "(c)_%": round(vc.get("(c)", 0) / total * 100, 2),
            })
        return pd.DataFrame(rows)

    def parameter_stats_by_class(self, year: int = 2020) -> pd.DataFrame:
        """Статистика параметров по категориям опасности."""
        danger_col = f"{COL_DANGER}_{year}"
        if danger_col not in self.df.columns:
            return pd.DataFrame()
        rows = []
        for cat in ["(a)", "(b)", "(c)"]:
            sub = self.df[self.df[danger_col] == cat]
            row = {"Категория": cat, "N": len(sub)}
            for col_base in [COL_DEPTH, COL_WID, COL_LEN, COL_KBD, COL_PF]:
                col = f"{col_base}_{year}"
                if col in sub.columns:
                    row[f"{col_base}_mean"] = sub[col].mean()
                    row[f"{col_base}_median"] = sub[col].median()
            rows.append(row)
        return pd.DataFrame(rows)

    def transitions(self, year_from: int = 2020, year_to: int = 2024) -> pd.DataFrame:
        """Матрица переходов между категориями."""
        col_from = f"{COL_DANGER}_{year_from}"
        col_to   = f"{COL_DANGER}_{year_to}"
        if col_from not in self.df.columns or col_to not in self.df.columns:
            return pd.DataFrame()
        sub = self.df[[col_from, col_to]].dropna()
        return pd.crosstab(sub[col_from], sub[col_to], margins=True)

    def run_full_analysis(self, save_plots: bool = True):
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        print("=== Анализ категорий опасности ===")
        dist = self.distribution_by_year()
        print("\nРаспределение по годам:")
        print(dist.to_string(index=False))
        print("\nПараметры по классам (2020):")
        ps = self.parameter_stats_by_class(2020)
        print(ps.to_string(index=False))
        print("\nПереходы 2020→2024:")
        tr = self.transitions(2020, 2024)
        print(tr.to_string())
        return {"distribution": dist, "params": ps, "transitions": tr}


if __name__ == "__main__":
    if not MERGE_TABLE.exists():
        print("Мерж-таблица не найдена.")
        sys.exit(1)
    df = pd.read_csv(MERGE_TABLE, low_memory=False)
    analyzer = DangerAnalyzer(df)
    analyzer.run_full_analysis()
