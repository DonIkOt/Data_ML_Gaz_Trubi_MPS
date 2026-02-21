"""
src/eda/hotspot.py — Анализ горячих зон повышенной дефектности

Идентифицирует участки трассы с аномально высокой плотностью дефектов,
сравнивает характеристики дефектов внутри и вне зон.

Использование:
    python src/eda/hotspot.py
    
    from src.eda.hotspot import HotspotAnalyzer
    analyzer = HotspotAnalyzer(merge_df)
    analyzer.run_full_analysis()
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    MERGE_TABLE, PLOTS_DIR, INSPECTION_YEARS,
    COL_DEPTH, COL_LEN, COL_WID, COL_DIST_LW, COL_DIST_CW,
    COL_DIST_REF, COL_DANGER,
    HOTSPOT_START_M, HOTSPOT_END_M,
)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.dpi': 130,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
})


class HotspotAnalyzer:
    """Класс для анализа горячих зон трубопровода."""

    def __init__(
        self,
        df: pd.DataFrame,
        hotspot_start_m: float = HOTSPOT_START_M,
        hotspot_end_m:   float = HOTSPOT_END_M,
    ):
        self.df = df.copy()
        self.hs_start = hotspot_start_m
        self.hs_end   = hotspot_end_m
        self._add_zone_flag()

    def _add_zone_flag(self):
        self.df["in_hotzone"] = self.df[COL_DIST_REF].between(
            self.hs_start, self.hs_end
        ).astype(int)

    @property
    def zone_length_km(self) -> float:
        return (self.hs_end - self.hs_start) / 1_000_000

    @property
    def pipe_length_km(self) -> float:
        return (self.df[COL_DIST_REF].max() - self.df[COL_DIST_REF].min()) / 1_000_000

    # ─── Плотность по километрам ─────────────────────────────────────────────

    def density_by_km(self, bin_size_km: float = 1.0) -> pd.DataFrame:
        """Считает плотность дефектов (дефектов/км) по километровым бинам."""
        self.df["km"] = self.df[COL_DIST_REF] / 1_000_000
        km_min = self.df["km"].min()
        km_max = self.df["km"].max()
        bins = np.arange(km_min, km_max + bin_size_km, bin_size_km)
        
        self.df["km_bin"] = pd.cut(self.df["km"], bins=bins, labels=bins[:-1])
        density = (
            self.df.groupby("km_bin", observed=True)
            .size()
            .reset_index(name="n_defects")
        )
        density["density_per_km"] = density["n_defects"] / bin_size_km
        density["km_bin"] = density["km_bin"].astype(float)
        density["in_hotzone"] = density["km_bin"].between(
            self.hs_start / 1_000_000, self.hs_end / 1_000_000
        )
        return density

    def zone_vs_rest_stats(self, year_col_suffix: int = 2020) -> dict:
        """Сравнивает статистику дефектов в зоне и вне её."""
        zone = self.df[self.df["in_hotzone"] == 1]
        rest = self.df[self.df["in_hotzone"] == 0]
        
        results = {}
        cols_to_compare = [
            f"{COL_DEPTH}_{year_col_suffix}",
            f"{COL_LEN}_{year_col_suffix}",
            f"{COL_WID}_{year_col_suffix}",
            f"{COL_DIST_LW}_{year_col_suffix}",
            f"{COL_DIST_CW}_{year_col_suffix}",
        ]
        
        for col in cols_to_compare:
            if col not in self.df.columns:
                continue
            z_vals = zone[col].dropna()
            r_vals = rest[col].dropna()
            results[col] = {
                "zone_median": z_vals.median(),
                "rest_median": r_vals.median(),
                "zone_mean":   z_vals.mean(),
                "rest_mean":   r_vals.mean(),
                "ratio_median": z_vals.median() / r_vals.median() if r_vals.median() != 0 else np.nan,
                "n_zone": len(z_vals),
                "n_rest": len(r_vals),
            }
        
        return results

    def density_comparison_by_year(self) -> pd.DataFrame:
        """Плотность дефектов в зоне vs трасса для каждого года."""
        rows = []
        rest_len_km = self.pipe_length_km - self.zone_length_km
        
        for year in INSPECTION_YEARS:
            depth_col = f"{COL_DEPTH}_{year}"
            if depth_col not in self.df.columns:
                continue
            subset = self.df.dropna(subset=[depth_col])
            zone_n = subset["in_hotzone"].sum()
            rest_n = len(subset) - zone_n
            
            zone_density = zone_n / self.zone_length_km if self.zone_length_km > 0 else 0
            rest_density = rest_n / rest_len_km if rest_len_km > 0 else 0
            
            rows.append({
                "year": year,
                "zone_n": zone_n,
                "rest_n": rest_n,
                "zone_density": zone_density,
                "rest_density": rest_density,
                "ratio": zone_density / rest_density if rest_density > 0 else np.nan,
            })
        
        return pd.DataFrame(rows)

    # ─── Визуализация ─────────────────────────────────────────────────────────

    def plot_density_map(self, save_path: Path | None = None) -> plt.Figure:
        """Карта плотности дефектов вдоль трассы."""
        density = self.density_by_km()
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        zone_mask = density["in_hotzone"]
        ax.bar(density.loc[~zone_mask, "km_bin"],
               density.loc[~zone_mask, "density_per_km"],
               width=0.9, color="#2563EB", alpha=0.7, label="Основная трасса")
        ax.bar(density.loc[zone_mask, "km_bin"],
               density.loc[zone_mask, "density_per_km"],
               width=0.9, color="#DC2626", alpha=0.85, label=f"Зона {self.hs_start/1e6:.0f}–{self.hs_end/1e6:.0f} км")
        
        ax.axvspan(self.hs_start / 1e6, self.hs_end / 1e6,
                   alpha=0.08, color="#DC2626")
        ax.set_xlabel("Расстояние от НК, км", fontsize=11)
        ax.set_ylabel("Дефектов / км", fontsize=11)
        ax.set_title("Плотность дефектов вдоль трассы трубопровода", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', facecolor='white')
        return fig

    def plot_density_by_year(self, save_path: Path | None = None) -> plt.Figure:
        """Плотность зона vs трасса по годам."""
        comp = self.density_comparison_by_year()
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        
        ax = axes[0]
        x = np.arange(len(comp)); w = 0.35
        ax.bar(x - w/2, comp["zone_density"], w, color="#DC2626", alpha=0.85, label="Зона")
        ax.bar(x + w/2, comp["rest_density"], w, color="#2563EB", alpha=0.85, label="Трасса")
        ax.set_xticks(x); ax.set_xticklabels(comp["year"])
        ax.set_xlabel("Год пробега"); ax.set_ylabel("Дефектов / км")
        ax.set_title("Плотность дефектов: зона vs трасса", fontsize=12, fontweight='bold')
        ax.legend()
        
        ax = axes[1]
        colors = ["#DC2626" if r > 3 else "#D97706" if r > 2 else "#16A34A"
                  for r in comp["ratio"]]
        bars = ax.bar(comp["year"], comp["ratio"], color=colors, alpha=0.85, width=2.5)
        for bar, v in zip(bars, comp["ratio"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05, f"{v:.1f}×",
                    ha='center', fontsize=12, fontweight='bold')
        ax.axhline(1, color='gray', ls='--', lw=1.5, alpha=0.7)
        ax.set_xlabel("Год пробега"); ax.set_ylabel("Коэффициент превышения")
        ax.set_title("Кратность превышения плотности\n(зона / трасса)", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', facecolor='white')
        return fig

    def plot_parameter_comparison(self, year: int = 2020,
                                  save_path: Path | None = None) -> plt.Figure:
        """Сравнение параметров дефектов в зоне и вне её (boxplot)."""
        zone = self.df[self.df["in_hotzone"] == 1]
        rest = self.df[self.df["in_hotzone"] == 0]
        
        params = {
            f"Длина, мм\n({COL_LEN}_{year})": f"{COL_LEN}_{year}",
            f"Ширина, мм\n({COL_WID}_{year})": f"{COL_WID}_{year}",
            f"Глубина, %\n({COL_DEPTH}_{year})": f"{COL_DEPTH}_{year}",
            f"До прод. шва, мм\n({COL_DIST_LW}_{year})": f"{COL_DIST_LW}_{year}",
            f"До кольц. шва, мм\n({COL_DIST_CW}_{year})": f"{COL_DIST_CW}_{year}",
        }
        params = {k: v for k, v in params.items() if v in self.df.columns}
        
        fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 5))
        if len(params) == 1:
            axes = [axes]
        
        for ax, (label, col) in zip(axes, params.items()):
            data_plot = [zone[col].dropna().values, rest[col].dropna().values]
            bp = ax.boxplot(data_plot, labels=["Зона", "Трасса"],
                           patch_artist=True, notch=False,
                           medianprops=dict(color='black', lw=2))
            bp['boxes'][0].set_facecolor("#FEE2E2")
            bp['boxes'][1].set_facecolor("#DBEAFE")
            ax.set_title(label, fontsize=9)
        
        fig.suptitle(f"Параметры дефектов: зона 136–145 км vs трасса ({year})",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', facecolor='white')
        return fig

    # ─── Полный анализ ────────────────────────────────────────────────────────

    def run_full_analysis(self, save_plots: bool = True) -> dict:
        """Запуск полного анализа горячей зоны."""
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print(f"АНАЛИЗ ГОРЯЧЕЙ ЗОНЫ: {self.hs_start/1e6:.0f}–{self.hs_end/1e6:.0f} км")
        print("=" * 60)
        
        # Плотность по годам
        comp = self.density_comparison_by_year()
        print("\nПлотность дефектов (дефектов/км):")
        print(comp.to_string(index=False))
        
        # Сравнение параметров
        stats = self.zone_vs_rest_stats()
        print("\nСравнение параметров (медиана):")
        for col, s in stats.items():
            print(f"  {col}: зона={s['zone_median']:.1f}, трасса={s['rest_median']:.1f}, "
                  f"ratio={s['ratio_median']:.2f}x")
        
        # Графики
        if save_plots:
            self.plot_density_map(PLOTS_DIR / "hotspot_density_map.png")
            self.plot_density_by_year(PLOTS_DIR / "hotspot_by_year.png")
            self.plot_parameter_comparison(PLOTS_DIR / "hotspot_params.png")
            print(f"\nГрафики сохранены в {PLOTS_DIR}")
        
        return {"density_comparison": comp, "parameter_stats": stats}


# ══════════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not MERGE_TABLE.exists():
        print(f"Файл {MERGE_TABLE} не найден. Сначала запустите matching.py")
        sys.exit(1)
    
    df = pd.read_csv(MERGE_TABLE, low_memory=False)
    print(f"Загружена мерж-таблица: {len(df)} строк")
    
    analyzer = HotspotAnalyzer(df)
    results  = analyzer.run_full_analysis(save_plots=True)
