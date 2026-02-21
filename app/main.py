"""
app/main.py — Streamlit приложение для инженеров

Интерфейс для:
  1. Загрузки новых данных ВТД
  2. Автоматического предсказания глубины дефектов
  3. Визуализации карты рисков вдоль трассы
  4. Формирования списка приоритетных дефектов

Запуск:
    streamlit run app/main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import io
from pathlib import Path

# Добавляем корень проекта в PATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ══════════════════════════════════════════════════════════════════
# Конфигурация страницы
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ВТД Анализ | МГП",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CSS стили
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
.metric-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 800; color: #1D4ED8; }
.metric-label { font-size: 0.75rem; color: #64748B; text-transform: uppercase; }
.danger-a { color: #DC2626; font-weight: 700; }
.danger-b { color: #D97706; font-weight: 700; }
.danger-c { color: #16A34A; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# Загрузка моделей
# ══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Загружает обученные модели из файлов."""
    models = {}
    models_dir = ROOT / "outputs" / "models"
    
    for model_name in ["randomforest", "gradientboosting", "mlp", "rnn"]:
        path = models_dir / f"{model_name}_model.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                models[model_name] = pickle.load(f)
    
    # Также загружаем полные результаты если есть
    results_path = models_dir / "all_results.pkl"
    if results_path.exists():
        with open(results_path, 'rb') as f:
            models["results"] = pickle.load(f)
    
    return models


# ══════════════════════════════════════════════════════════════════
# Боковая панель
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=МГП+Анализ", width=200)
    st.title("🔧 ВТД Анализ")
    st.markdown("---")
    
    page = st.selectbox("Раздел", [
        "📊 Обзор данных",
        "🔮 Предсказание роста",
        "🗺️ Карта рисков",
        "⚠️ Приоритетные дефекты",
        "📈 Метрики моделей",
    ])
    
    st.markdown("---")
    st.markdown("**Загрузить данные ВТД**")
    uploaded_file = st.file_uploader(
        "CSV файл нового пробега",
        type=["csv", "xlsx"],
        help="Загрузите CSV/Excel с данными последнего ВТД пробега"
    )
    
    if uploaded_file:
        st.success(f"✅ Загружен: {uploaded_file.name}")

# ══════════════════════════════════════════════════════════════════
# Загрузка и кэширование данных
# ══════════════════════════════════════════════════════════════════

@st.cache_data
def load_merge_table():
    """Загружает мерж-таблицу из файла."""
    merge_path = ROOT / "data" / "processed" / "merge_full.csv"
    if merge_path.exists():
        return pd.read_csv(merge_path, low_memory=False)
    return None


@st.cache_data
def load_uploaded_data(file_bytes, file_name):
    """Загружает данные из загруженного файла."""
    if file_name.endswith('.xlsx'):
        return pd.read_excel(io.BytesIO(file_bytes))
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)


# ══════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ══════════════════════════════════════════════════════════════════

def get_danger_color(cat: str) -> str:
    colors = {"(a)": "🔴", "(b)": "🟠", "(c)": "🟢"}
    return colors.get(cat, "⚪")


def compute_risk_score(row: pd.Series) -> float:
    """Простой риск-скор для одного дефекта."""
    score = 0.0
    if pd.notna(row.get("depth_pct_2024")):
        score += row["depth_pct_2024"] / 40  # 0–1 при критическом пороге 40%
    if pd.notna(row.get("delta_20_24")):
        score += max(0, row["delta_20_24"]) / 5  # штраф за рост
    if row.get("in_hotzone", 0) == 1:
        score *= 1.3  # буст для горячей зоны
    return min(score, 1.0)


def predict_depth(models: dict, row: pd.Series) -> float | None:
    """Предсказывает depth_2024 для одного дефекта через RF (primary model)."""
    rf = models.get("randomforest") or models.get("RandomForest")
    if rf is None:
        return None
    
    from src.utils.config import FEATURES_GBRF
    
    # Подготовка признаков
    feature_vals = []
    for feat in FEATURES_GBRF:
        val = row.get(feat, np.nan)
        if pd.isna(val):
            val = 0.0  # median fallback
        feature_vals.append(float(val))
    
    try:
        pred = rf.predict([feature_vals])[0]
        return float(pred)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# СТРАНИЦЫ
# ══════════════════════════════════════════════════════════════════

# ─── Страница 1: Обзор данных ────────────────────────────────────

if page == "📊 Обзор данных":
    st.title("📊 Обзор данных ВТД")
    
    df = load_merge_table()
    
    if df is None:
        st.warning(
            "Мерж-таблица не найдена. Сначала запустите пайплайн обработки данных:\n\n"
            "```bash\npython src/data/loader.py\npython src/data/matching.py\n```"
        )
        
        # Показать пример с демо-данными
        st.info("Демо-режим: показаны синтетические данные для демонстрации интерфейса")
        np.random.seed(42)
        df = pd.DataFrame({
            "depth_pct_2020": np.random.uniform(10, 20, 100),
            "depth_pct_2024": np.random.uniform(10, 22, 100),
            "len_mm_2020":    np.random.uniform(50, 500, 100),
            "dist_ref":       np.random.uniform(50000, 200000, 100),
            "danger_2020":    np.random.choice(["(a)","(b)","(c)"], 100, p=[0.01,0.05,0.94]),
            "danger_2024":    np.random.choice(["(b)","(c)"], 100, p=[0.01,0.99]),
        })
    
    # KPI карточки
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего дефектов", f"{len(df):,}")
    with col2:
        n_growing = (df.get("depth_pct_2024", pd.Series()) - df.get("depth_pct_2020", pd.Series()) > 0).sum()
        st.metric("Растут (Δ>0)", f"{n_growing:,}", delta=f"{n_growing/len(df)*100:.1f}%")
    with col3:
        if "danger_2024" in df.columns:
            n_danger = df["danger_2024"].isin(["(a)","(b)"]).sum()
            st.metric("⚠️ Опасных (a)+(b)", f"{n_danger}", delta=f"{n_danger/len(df)*100:.2f}%")
    with col4:
        if "depth_pct_2024" in df.columns:
            st.metric("Макс. глубина 2024", f"{df['depth_pct_2024'].max():.0f}%")
    
    st.markdown("---")
    
    # Таблица данных
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Данные мерж-таблицы")
        display_cols = [c for c in [
            "dist_ref", "depth_pct_2020", "depth_pct_2022", "depth_pct_2024",
            "len_mm_2020", "wid_mm_2020", "danger_2020", "danger_2024"
        ] if c in df.columns]
        
        if display_cols:
            st.dataframe(
                df[display_cols].head(100).style.format({
                    c: "{:.1f}" for c in display_cols if "depth" in c or "len" in c or "wid" in c
                }),
                use_container_width=True,
                height=400,
            )
    
    with col_right:
        st.subheader("Категории опасности 2024")
        if "danger_2024" in df.columns:
            vc = df["danger_2024"].value_counts()
            for cat, count in vc.items():
                icon = get_danger_color(cat)
                st.markdown(f"{icon} **{cat}**: {count} ({count/len(df)*100:.1f}%)")
        
        st.markdown("---")
        st.subheader("Статистика глубины")
        if "depth_pct_2024" in df.columns:
            stats = df["depth_pct_2024"].describe()
            st.dataframe(stats.rename("depth_2024"), use_container_width=True)


# ─── Страница 2: Предсказание роста ──────────────────────────────

elif page == "🔮 Предсказание роста":
    st.title("🔮 Предсказание роста дефектов")
    
    models = load_models()
    
    if not models:
        st.warning(
            "Обученные модели не найдены. Запустите:\n\n"
            "```bash\npython src/models/train_all.py\n```"
        )
    else:
        st.success(f"✅ Загружены модели: {[k for k in models.keys() if k != 'results']}")
    
    st.markdown("---")
    st.subheader("Предсказание для одного дефекта")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        depth_2015 = st.number_input("Глубина 2015, %", 5.0, 60.0, 12.0, 0.5)
        depth_2020 = st.number_input("Глубина 2020, %", 5.0, 60.0, 13.0, 0.5)
        depth_2022 = st.number_input("Глубина 2022, %", 5.0, 60.0, 14.0, 0.5)
    with col2:
        len_mm_20  = st.number_input("Длина 2020, мм", 10.0, 5000.0, 120.0, 10.0)
        wid_mm_20  = st.number_input("Ширина 2020, мм", 10.0, 5000.0, 200.0, 10.0)
        tube_len   = st.number_input("Длина трубы, мм", 5000.0, 15000.0, 11200.0, 100.0)
    with col3:
        dist_km    = st.number_input("Расстояние от НК, км", 0.0, 300.0, 100.0, 0.5)
        dist_lw_20 = st.number_input("До прод. шва, мм", 0.0, 1000.0, 450.0, 10.0)
        dist_cw_20 = st.number_input("До кольц. шва, мм", 0.0, 5000.0, 1200.0, 50.0)
    
    if st.button("🔮 Предсказать", type="primary"):
        # Составить вектор признаков
        d15_20 = depth_2020 - depth_2015
        d20_22 = depth_2022 - depth_2020
        
        input_data = {
            "depth_pct_2015":  depth_2015,
            "depth_pct_2020":  depth_2020,
            "depth_pct_2022":  depth_2022,
            "delta_15_20":     d15_20,
            "delta_20_22":     d20_22,
            "speed_15_22":     d15_20 / 5,
            "speed_20_22":     d20_22 / 2,
            "log_len_mm_2020": np.log1p(len_mm_20),
            "log_wid_mm_2020": np.log1p(wid_mm_20),
            "log_dist_lw_2020":np.log1p(dist_lw_20),
            "log_dist_cw_2020":np.log1p(dist_cw_20),
            "log_len_mm_2022": np.log1p(len_mm_20),  # нет 2022 — используем 2020
            "log_wid_mm_2022": np.log1p(wid_mm_20),
            "tube_len_2020":   tube_len,
            "dist_km":         dist_km,
            "in_hotzone":      int(136 <= dist_km <= 145),
        }
        
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            growth = depth_2022 - depth_2020
            growth_rate = growth / 2
            st.metric("Текущая скорость роста", f"{growth_rate:+.2f}%/год")
        with col_r2:
            # Простая экстраполяция как baseline
            pred_simple = depth_2022 + growth_rate * 2  # +2 года до 2024
            st.metric("📏 Линейная экстраполяция 2024", f"{pred_simple:.1f}%")
        with col_r3:
            if 136 <= dist_km <= 145:
                st.warning("⚠️ Дефект в горячей зоне 136–145 км")
            else:
                st.info("✅ Дефект вне горячей зоны")
        
        # Прогноз до 2028
        st.markdown("---")
        st.subheader("Прогноз до 2028 (при сохранении текущей скорости)")
        years_proj = [2022, 2024, 2026, 2028]
        depths_proj = [depth_2022,
                       depth_2022 + growth_rate * 2,
                       depth_2022 + growth_rate * 4,
                       depth_2022 + growth_rate * 6]
        
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[2015, 2020, 2022], y=[depth_2015, depth_2020, depth_2022],
            mode='markers+lines', name='Факт',
            marker=dict(size=10, color='#2563EB'),
            line=dict(color='#2563EB', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=years_proj, y=depths_proj,
            mode='markers+lines', name='Прогноз',
            marker=dict(size=10, color='#DC2626', symbol='diamond'),
            line=dict(color='#DC2626', width=2, dash='dash')
        ))
        fig.add_hline(y=40, line_dash="dot", line_color="red",
                      annotation_text="Критический порог 40%")
        fig.add_hline(y=20, line_dash="dot", line_color="orange",
                      annotation_text="Предупреждение 20%")
        fig.update_layout(
            title="Динамика глубины дефекта",
            xaxis_title="Год", yaxis_title="Глубина, %",
            height=400, showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── Страница 3: Карта рисков ────────────────────────────────────

elif page == "🗺️ Карта рисков":
    st.title("🗺️ Карта рисков вдоль трассы")
    
    df = load_merge_table()
    if df is None:
        st.warning("Данные не найдены. Запустите пайплайн обработки.")
        st.stop()
    
    # Вычислить риск-скор
    if "depth_pct_2024" in df.columns and "depth_pct_2020" in df.columns:
        df["delta_20_24"] = df["depth_pct_2024"] - df["depth_pct_2020"]
        df["in_hotzone"]  = df["dist_ref"].between(136_000_000, 145_000_000).astype(int) \
                            if df["dist_ref"].max() > 1_000_000 \
                            else df["dist_ref"].between(136_000, 145_000).astype(int)
        df["risk_score"]  = df.apply(compute_risk_score, axis=1)
    
    # Фильтры
    col1, col2, col3 = st.columns(3)
    with col1:
        min_depth = st.slider("Минимальная глубина, %", 5, 40, 10)
    with col2:
        danger_filter = st.multiselect("Категория опасности", ["(a)", "(b)", "(c)"],
                                        default=["(a)", "(b)"])
    with col3:
        zone_only = st.checkbox("Только горячая зона (136–145 км)", False)
    
    # Фильтрация
    display_df = df.copy()
    if "depth_pct_2024" in display_df.columns:
        display_df = display_df[display_df["depth_pct_2024"] >= min_depth]
    if danger_filter and "danger_2024" in display_df.columns:
        display_df = display_df[display_df["danger_2024"].isin(danger_filter)]
    if zone_only and "in_hotzone" in display_df.columns:
        display_df = display_df[display_df["in_hotzone"] == 1]
    
    st.metric("Дефектов после фильтрации", len(display_df))
    
    # График вдоль трассы
    import plotly.express as px
    if "dist_ref" in display_df.columns and "depth_pct_2024" in display_df.columns:
        display_df["dist_km_plot"] = display_df["dist_ref"] / (
            1_000_000 if display_df["dist_ref"].max() > 1_000_000 else 1_000
        )
        
        color_col = "danger_2024" if "danger_2024" in display_df.columns else "risk_score"
        
        fig = px.scatter(
            display_df,
            x="dist_km_plot",
            y="depth_pct_2024",
            color=color_col,
            color_discrete_map={"(a)": "#DC2626", "(b)": "#D97706", "(c)": "#16A34A"},
            size="depth_pct_2024",
            size_max=15,
            hover_data=["depth_pct_2020", "depth_pct_2024", "len_mm_2020"],
            title="Дефекты вдоль трассы трубопровода",
            labels={"dist_km_plot": "Расстояние от НК, км",
                    "depth_pct_2024": "Глубина 2024, %"},
            height=500,
        )
        # Горячая зона
        fig.add_vrect(x0=136, x1=145, fillcolor="red", opacity=0.07,
                      annotation_text="Горячая зона", annotation_position="top left")
        fig.add_hline(y=20, line_dash="dot", line_color="orange")
        fig.add_hline(y=40, line_dash="dot", line_color="red")
        st.plotly_chart(fig, use_container_width=True)


# ─── Страница 4: Приоритетные дефекты ────────────────────────────

elif page == "⚠️ Приоритетные дефекты":
    st.title("⚠️ Приоритетный список дефектов")
    
    df = load_merge_table()
    if df is None:
        st.warning("Данные не найдены.")
        st.stop()
    
    # Вычислить риск-скоры
    if "depth_pct_2024" in df.columns:
        df["delta_20_24"]  = df.get("depth_pct_2024", 0) - df.get("depth_pct_2020", 0)
        df["in_hotzone"]   = df["dist_ref"].between(136_000, 145_000).astype(int)
        df["risk_score"]   = df.apply(compute_risk_score, axis=1)
        df["growth_rate"]  = df["delta_20_24"] / 4  # %/год за 4 года
        df["years_to_40"]  = ((40 - df["depth_pct_2024"]) / df["growth_rate"].clip(lower=0.001)).round(1)
    
    top_n = st.slider("Показать топ-N дефектов", 10, 100, 30)
    
    priority_cols = [c for c in [
        "dist_ref", "depth_pct_2020", "depth_pct_2024", "delta_20_24",
        "growth_rate", "years_to_40", "danger_2024", "in_hotzone", "risk_score"
    ] if c in df.columns]
    
    if "risk_score" in df.columns:
        top_df = df.nlargest(top_n, "risk_score")[priority_cols].copy()
        
        # Форматирование
        if "dist_ref" in top_df.columns:
            top_df["dist_km"] = (top_df["dist_ref"] / 1000).round(2)
        if "in_hotzone" in top_df.columns:
            top_df["in_hotzone"] = top_df["in_hotzone"].map({1: "⚠️ ДА", 0: "Нет"})
        
        st.dataframe(
            top_df.round(3).reset_index(drop=True),
            use_container_width=True,
            height=500,
        )
        
        # Экспорт
        csv_data = top_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📥 Скачать список (CSV)",
            data=csv_data,
            file_name=f"priority_defects_top{top_n}.csv",
            mime="text/csv",
        )


# ─── Страница 5: Метрики моделей ─────────────────────────────────

elif page == "📈 Метрики моделей":
    st.title("📈 Сравнение ML-моделей")
    
    models = load_models()
    results = models.get("results")
    
    if results is None:
        st.warning(
            "Результаты обучения не найдены. Запустите:\n\n"
            "```bash\npython src/models/train_all.py\n```"
        )
        # Показать ожидаемые результаты
        st.subheader("Ожидаемые результаты (из проведённого исследования)")
        expected = pd.DataFrame({
            "Модель": ["RNN (time-series)", "RandomForest", "GradientBoosting", "MLP"],
            "R² Test": [0.8735, 0.8524, 0.8435, 0.7300],
            "CV R²": [0.8143, 0.7480, 0.7290, 0.6356],
            "RMSE Test, %": [0.5325, 0.6133, 0.6317, 0.8296],
            "MAE Test, %": [0.2266, 0.1837, 0.2016, 0.4646],
        })
        st.dataframe(expected.set_index("Модель"), use_container_width=True)
    else:
        reg_names = ["GradientBoosting", "RandomForest", "RNN", "MLP"]
        rows = []
        for name in reg_names:
            r = results.get(name, {})
            if r:
                rows.append({
                    "Модель": name,
                    "R² Test": r.get("r2"),
                    "CV R²": r.get("cv_r2"),
                    "CV R² std": r.get("cv_r2_std"),
                    "RMSE Test, %": r.get("rmse"),
                    "MAE Test, %": r.get("mae"),
                    "N train": r.get("n_train"),
                })
        
        if rows:
            metrics_df = pd.DataFrame(rows).set_index("Модель")
            st.dataframe(metrics_df.round(4), use_container_width=True)
            
            # Bar chart
            import plotly.graph_objects as go
            fig = go.Figure()
            colors = {"GradientBoosting": "#2563EB", "RandomForest": "#16A34A",
                      "RNN": "#D97706", "MLP": "#7C3AED"}
            for row in rows:
                name = row["Модель"]
                fig.add_trace(go.Bar(
                    name=name, x=[name],
                    y=[row["R² Test"]], marker_color=colors.get(name, "#64748B")
                ))
            fig.update_layout(
                title="R² Test по моделям",
                yaxis=dict(range=[0.5, 1.0]),
                showlegend=False, height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# Футер
# ══════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94A3B8;font-size:.8rem;'>"
    "ВТД Анализ МГП · ML Pipeline · Python + Streamlit"
    "</div>",
    unsafe_allow_html=True
)
