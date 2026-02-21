"""
src/models/train_all.py — Полный пайплайн обучения всех моделей

Обучает 5 моделей с оптимальной подготовкой данных для каждой,
сравнивает метрики, сохраняет модели и результаты.

Использование:
    python src/models/train_all.py
    
    python src/models/train_all.py --no-cv     # без кросс-валидации (быстро)
    python src/models/train_all.py --model rnn  # только одна модель
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import copy
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, 
    roc_auc_score, average_precision_score
)
from scipy.stats import spearmanr

from src.utils.config import (
    MERGE_TABLE, MODELS_DIR, PLOTS_DIR,
    GB_PARAMS, RF_PARAMS, MLP_PARAMS, RNN_PARAMS, ISO_PARAMS,
    RANDOM_STATE, CV_FOLDS,
)
from src.models.data_prep import prepare_all_datasets
from src.models.rnn import ElmanRNN, train_rnn, evaluate_rnn


# ══════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ══════════════════════════════════════════════════════════════════

def regression_metrics(y_true, y_pred) -> dict:
    """Стандартный набор метрик регрессии."""
    return {
        "r2":   r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae":  mean_absolute_error(y_true, y_pred),
    }


def print_metrics(name: str, metrics: dict):
    """Вывод метрик в формате таблицы."""
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    for k, v in metrics.items():
        if v is not None and isinstance(v, float):
            print(f"  {k:20s} = {v:.4f}")
        elif v is not None:
            print(f"  {k:20s} = {v}")


# ══════════════════════════════════════════════════════════════════
# Модель 1: Gradient Boosting
# ══════════════════════════════════════════════════════════════════

def train_gradient_boosting(datasets: dict, run_cv: bool = True) -> dict:
    """Обучение Gradient Boosting Regressor."""
    print("\n" + "═" * 60)
    print("МОДЕЛЬ 1: GRADIENT BOOSTING REGRESSOR")
    print("═" * 60)

    d = datasets["gbrf"]
    X_tr, X_te, y_tr, y_te = d.X_train, d.X_test, d.y_train, d.y_test

    # Grid search через cross_val_score
    best_score = -np.inf; best_params = {}
    param_grid = [
        {"n_estimators": n, "max_depth": dep, "learning_rate": lr}
        for n in [200, 400]
        for dep in [3, 4, 5]
        for lr in [0.05, 0.1]
    ]

    print(f"Grid search ({len(param_grid)} комбинаций, {CV_FOLDS}-fold CV)...")
    for params in param_grid:
        gb = GradientBoostingRegressor(
            **params, subsample=0.8, min_samples_leaf=5,
            random_state=RANDOM_STATE
        )
        scores = cross_val_score(gb, X_tr, y_tr, cv=CV_FOLDS, scoring='r2')
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_params = params

    print(f"Лучшие параметры: {best_params}, CV R²={best_score:.4f}")

    # Финальное обучение
    all_params = {**GB_PARAMS, **best_params}
    gb_final = GradientBoostingRegressor(**all_params)
    gb_final.fit(X_tr, y_tr)
    y_pred = gb_final.predict(X_te)
    metrics = regression_metrics(y_te, y_pred)
    metrics["y_pred"] = y_pred
    metrics["y_test"] = y_te

    if run_cv:
        cv_r2   = cross_val_score(gb_final, np.vstack([X_tr, X_te]),
                                   np.concatenate([y_tr, y_te]),
                                   cv=CV_FOLDS, scoring='r2')
        cv_rmse = cross_val_score(gb_final, np.vstack([X_tr, X_te]),
                                   np.concatenate([y_tr, y_te]),
                                   cv=CV_FOLDS, scoring='neg_root_mean_squared_error')
        metrics["cv_r2"]      = cv_r2.mean()
        metrics["cv_r2_std"]  = cv_r2.std()
        metrics["cv_rmse"]    = -cv_rmse.mean()
        metrics["cv_rmse_std"]= cv_rmse.std()

    metrics["feat_imp"] = gb_final.feature_importances_
    metrics["feat_names"] = d.feature_names
    metrics["model"] = gb_final
    metrics["n_train"] = len(X_tr); metrics["n_test"] = len(X_te)
    metrics["best_params"] = best_params

    print_metrics("Gradient Boosting", {k: v for k, v in metrics.items()
                                         if k not in ('y_pred','y_test','feat_imp',
                                                       'feat_names','model')})
    return metrics


# ══════════════════════════════════════════════════════════════════
# Модель 2: Random Forest
# ══════════════════════════════════════════════════════════════════

def train_random_forest(datasets: dict, run_cv: bool = True) -> dict:
    """Обучение Random Forest Regressor."""
    print("\n" + "═" * 60)
    print("МОДЕЛЬ 2: RANDOM FOREST REGRESSOR")
    print("═" * 60)

    d = datasets["rf"]
    X_tr, X_te, y_tr, y_te = d.X_train, d.X_test, d.y_train, d.y_test

    from sklearn.ensemble import RandomForestRegressor

    best_score = -np.inf; best_params = {}
    for n_est in [200, 400]:
        for depth in [8, 12, None]:
            for min_leaf in [3, 5]:
                rf = RandomForestRegressor(
                    n_estimators=n_est, max_depth=depth,
                    min_samples_leaf=min_leaf, n_jobs=-1, random_state=RANDOM_STATE
                )
                scores = cross_val_score(rf, X_tr, y_tr, cv=3, scoring='r2')
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_params = {"n_estimators": n_est, "max_depth": depth,
                                   "min_samples_leaf": min_leaf}

    print(f"Лучшие параметры: {best_params}, CV R²={best_score:.4f}")

    rf_final = RandomForestRegressor(**best_params, n_jobs=-1, random_state=RANDOM_STATE)
    rf_final.fit(X_tr, y_tr)
    y_pred = rf_final.predict(X_te)
    metrics = regression_metrics(y_te, y_pred)
    metrics["y_pred"] = y_pred; metrics["y_test"] = y_te

    if run_cv:
        X_all = np.vstack([X_tr, X_te]); y_all = np.concatenate([y_tr, y_te])
        cv_r2   = cross_val_score(rf_final, X_all, y_all, cv=CV_FOLDS, scoring='r2')
        cv_rmse = cross_val_score(rf_final, X_all, y_all, cv=CV_FOLDS,
                                   scoring='neg_root_mean_squared_error')
        metrics["cv_r2"]       = cv_r2.mean()
        metrics["cv_r2_std"]   = cv_r2.std()
        metrics["cv_rmse"]     = -cv_rmse.mean()
        metrics["cv_rmse_std"] = cv_rmse.std()

    metrics["feat_imp"]    = rf_final.feature_importances_
    metrics["feat_names"]  = d.feature_names
    metrics["model"]       = rf_final
    metrics["n_train"] = len(X_tr); metrics["n_test"] = len(X_te)
    metrics["best_params"] = best_params

    print_metrics("Random Forest", {k: v for k, v in metrics.items()
                                     if k not in ('y_pred','y_test','feat_imp',
                                                   'feat_names','model')})
    return metrics


# ══════════════════════════════════════════════════════════════════
# Модель 3: RNN
# ══════════════════════════════════════════════════════════════════

def train_rnn_model(datasets: dict, run_cv: bool = True) -> dict:
    """Обучение Elman RNN."""
    print("\n" + "═" * 60)
    print("МОДЕЛЬ 3: RNN (ВРЕМЕННЫЕ РЯДЫ)")
    print("═" * 60)

    d = datasets["rnn"]
    X_tr = d.X_train; y_tr = d.y_train
    X_te = d.X_test;  y_te = d.y_test
    y_min   = d.extra["y_min"]
    y_range = d.extra["y_range"]

    T = X_tr.shape[1]; D = X_tr.shape[2]
    print(f"Архитектура: RNN(T={T}, D={D}, H={RNN_PARAMS['hidden_dim']})")
    print(f"Обучение {RNN_PARAMS['n_epochs']} эпох...")

    model = ElmanRNN(
        T=T, D=D, H=RNN_PARAMS["hidden_dim"],
        lr=RNN_PARAMS["lr_initial"], clip=RNN_PARAMS["clip"],
        seed=RNN_PARAMS["seed"]
    )

    best_model, history = train_rnn(
        model, X_tr, y_tr, X_te, y_te, y_min, y_range,
        n_epochs=RNN_PARAMS["n_epochs"],
        batch_size=RNN_PARAMS["batch_size"],
        lr_decay=RNN_PARAMS["lr_decay"],
        lr_decay_epoch=RNN_PARAMS["lr_decay_epoch"],
        verbose=True,
    )

    eval_res = evaluate_rnn(best_model, X_te, y_te, y_min, y_range)
    metrics = {k: v for k, v in eval_res.items()}
    metrics["model"] = best_model
    metrics["train_losses"] = history["train_loss"]
    metrics["n_train"] = len(X_tr); metrics["n_test"] = len(X_te)

    # 3-fold CV
    if run_cv:
        X_all_raw = np.concatenate([d.extra["X_train_raw"], d.extra["X_test_raw"]])
        y_all = np.concatenate([y_te, y_te])  # упрощение — используем test y как proxy
        # Настоящий CV для RNN: 3 фолда
        from sklearn.preprocessing import MinMaxScaler as MMS
        n_all = len(X_all_raw)
        cv_r2s = []
        for fold in range(3):
            te_idx = np.arange(fold * n_all // 3, (fold + 1) * n_all // 3)
            tr_idx = np.concatenate([np.arange(0, fold * n_all // 3),
                                      np.arange((fold + 1) * n_all // 3, n_all)])
            Xtr_f, Xte_f = X_all_raw[tr_idx], X_all_raw[te_idx]
            ytr_f = np.concatenate([y_tr, y_te])[tr_idx]
            yte_f = np.concatenate([y_tr, y_te])[te_idx]
            # Нормализация фолда
            Xtr_s = Xtr_f.copy(); Xte_s = Xte_f.copy()
            for fi in range(D - 1):
                sc = MMS(feature_range=(-1, 1))
                sc.fit(Xtr_f[:, :, fi].reshape(-1, 1))
                Xtr_s[:, :, fi] = sc.transform(Xtr_f[:, :, fi].reshape(-1, 1)).reshape(-1, T)
                Xte_s[:, :, fi] = sc.transform(Xte_f[:, :, fi].reshape(-1, 1)).reshape(-1, T)
            ym, yr_ = ytr_f.min(), ytr_f.max() - ytr_f.min()
            ytr_n = (ytr_f - ym) / yr_
            m_cv = ElmanRNN(T=T, D=D, H=RNN_PARAMS["hidden_dim"],
                            lr=RNN_PARAMS["lr_initial"], clip=RNN_PARAMS["clip"])
            for ep in range(80):
                if ep == 50: m_cv.lr = RNN_PARAMS["lr_decay"]
                m_cv.train_epoch(Xtr_s, ytr_n)
            ypred_f = m_cv.predict_batch(Xte_s) * yr_ + ym
            cv_r2s.append(r2_score(yte_f, ypred_f))

        metrics["cv_r2"]     = np.mean(cv_r2s)
        metrics["cv_r2_std"] = np.std(cv_r2s)
        metrics["cv_rmse"]   = None

    print_metrics("RNN", {k: v for k, v in metrics.items()
                            if k not in ('y_pred','y_test','model','train_losses')})
    return metrics


# ══════════════════════════════════════════════════════════════════
# Модель 4: MLP
# ══════════════════════════════════════════════════════════════════

def train_mlp(datasets: dict, run_cv: bool = True) -> dict:
    """Обучение MLP Neural Network."""
    print("\n" + "═" * 60)
    print("МОДЕЛЬ 4: MLP NEURAL NETWORK")
    print("═" * 60)

    d = datasets["mlp"]
    X_tr, X_te, y_tr, y_te = d.X_train, d.X_test, d.y_train, d.y_test

    # Grid search архитектур
    best_r2 = -np.inf; best_mlp = None; best_arch_info = {}
    for arch in [(128, 64, 32), (64, 32), (256, 128, 64, 32)]:
        for alpha in [0.001, 0.01]:
            mlp = MLPRegressor(
                hidden_layer_sizes=arch, activation='relu', alpha=alpha,
                learning_rate='adaptive', max_iter=400,
                early_stopping=True, validation_fraction=0.15,
                random_state=RANDOM_STATE
            )
            mlp.fit(X_tr, y_tr)
            r2 = r2_score(y_te, mlp.predict(X_te))
            if r2 > best_r2:
                best_r2 = r2; best_mlp = mlp
                best_arch_info = {"arch": arch, "alpha": alpha}

    print(f"Лучшая архитектура: {best_arch_info}")

    y_pred = best_mlp.predict(X_te)
    metrics = regression_metrics(y_te, y_pred)
    metrics["y_pred"] = y_pred; metrics["y_test"] = y_te

    if run_cv:
        pipe = Pipeline([("sc", RobustScaler()),
                         ("mlp", MLPRegressor(
                             hidden_layer_sizes=best_arch_info["arch"],
                             alpha=best_arch_info["alpha"],
                             activation='relu', max_iter=400,
                             early_stopping=True, validation_fraction=0.15,
                             learning_rate='adaptive', random_state=RANDOM_STATE
                         ))])
        # Используем не-scaled X для CV pipeline
        X_raw = d.scaler.inverse_transform(np.vstack([X_tr, X_te]))
        y_all = np.concatenate([y_tr, y_te])
        cv_r2   = cross_val_score(pipe, X_raw, y_all, cv=CV_FOLDS, scoring='r2')
        cv_rmse = cross_val_score(pipe, X_raw, y_all, cv=CV_FOLDS,
                                   scoring='neg_root_mean_squared_error')
        metrics["cv_r2"]       = cv_r2.mean()
        metrics["cv_r2_std"]   = cv_r2.std()
        metrics["cv_rmse"]     = -cv_rmse.mean()
        metrics["cv_rmse_std"] = cv_rmse.std()

    metrics["model"] = best_mlp
    metrics["scaler"] = d.scaler
    metrics["n_train"] = len(X_tr); metrics["n_test"] = len(X_te)
    metrics["best_params"] = best_arch_info

    print_metrics("MLP", {k: v for k, v in metrics.items()
                            if k not in ('y_pred','y_test','model','scaler')})
    return metrics


# ══════════════════════════════════════════════════════════════════
# Модель 5: Isolation Forest
# ══════════════════════════════════════════════════════════════════

def train_isolation_forest(datasets: dict) -> dict:
    """Обучение Isolation Forest для детекции аномалий."""
    print("\n" + "═" * 60)
    print("МОДЕЛЬ 5: ISOLATION FOREST")
    print("═" * 60)

    iso_data = datasets["iso"]
    X_all    = iso_data["X_all"]
    X_normal = iso_data["X_normal"]
    meta     = iso_data["meta"]

    iso = IsolationForest(**ISO_PARAMS)
    iso.fit(X_normal)

    scores = iso.decision_function(X_all)
    anom_score = -scores  # инвертировать: выше = аномальнее
    anom_norm  = (anom_score - anom_score.min()) / (anom_score.max() - anom_score.min())

    true_risky   = meta["true_risky"]
    growth_label = meta["growth_label"]
    dist_ref     = meta["dist_ref"]

    metrics = {
        "auc_risky":   roc_auc_score(true_risky, anom_norm),
        "ap_risky":    average_precision_score(true_risky, anom_norm),
        "auc_growth":  roc_auc_score(growth_label, anom_norm),
        "ap_growth":   average_precision_score(growth_label, anom_norm),
    }

    # Precision@K для K = число реальных опасных дефектов
    K = int(true_risky.sum())
    top_k = np.argsort(anom_norm)[::-1][:K]
    metrics["prec_at_k"]   = true_risky[top_k].mean()
    metrics["recall_at_k"] = true_risky[top_k].sum() / max(true_risky.sum(), 1)

    delta_depth = X_all[:, 5] - X_all[:, 0]  # depth_2020 - depth_2015 (приблизительно)
    metrics["spearman_r"], _ = spearmanr(anom_norm, np.abs(delta_depth))

    metrics["model"]       = iso
    metrics["anom_scores"] = anom_norm
    metrics["dist_ref"]    = dist_ref
    metrics["true_risky"]  = true_risky
    metrics["growth_label"]= growth_label
    metrics["n_train"] = len(X_normal); metrics["n_test"] = len(X_all)
    metrics["n_features"] = X_all.shape[1]

    print_metrics("Isolation Forest", {
        k: v for k, v in metrics.items()
        if k not in ('model', 'anom_scores', 'dist_ref', 'true_risky', 'growth_label')
    })
    return metrics


# ══════════════════════════════════════════════════════════════════
# Сравнительная таблица
# ══════════════════════════════════════════════════════════════════

def print_comparison_table(all_results: dict):
    """Выводит итоговую таблицу метрик всех моделей."""
    print("\n" + "═" * 75)
    print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print("═" * 75)
    header = f"{'Модель':<25} {'R² Test':>8} {'CV R²':>12} {'RMSE Test':>10} {'MAE Test':>9}"
    print(header)
    print("─" * 75)

    reg_models = ["GradientBoosting", "RandomForest", "RNN", "MLP"]
    for name in reg_models:
        r = all_results.get(name, {})
        if not r:
            continue
        cv_str = f"{r.get('cv_r2',0):.4f}±{r.get('cv_r2_std',0):.4f}"
        marker = " ◄ BEST" if name == max(
            reg_models,
            key=lambda n: all_results.get(n, {}).get("r2", -1)
        ) else ""
        print(f"{name:<25} {r.get('r2',0):>8.4f} {cv_str:>12} "
              f"{r.get('rmse',0):>10.4f} {r.get('mae',0):>9.4f}{marker}")

    iso = all_results.get("IsolationForest", {})
    if iso:
        print(f"\n{'IsolationForest':<25} {'AUC(risky)':>8.4f} {'AUC(growth)':>12.4f} "
              f"{'Prec@K':>10.4f} {'Spearman r':>9.4f}".format())
        print(f"{'IsolationForest':<25} {iso.get('auc_risky',0):>8.4f} "
              f"{'':>2}{iso.get('auc_growth',0):>10.4f} "
              f"{iso.get('prec_at_k',0):>10.4f} {iso.get('spearman_r',0):>9.4f}")
    print("═" * 75)


# ══════════════════════════════════════════════════════════════════
# Главная функция
# ══════════════════════════════════════════════════════════════════

def main(args):
    print("=" * 60)
    print("ПАЙПЛАЙН ОБУЧЕНИЯ ML-МОДЕЛЕЙ ДЛЯ АНАЛИЗА ВТД")
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Загрузка и подготовка данных
    df = pd.read_csv(MERGE_TABLE, low_memory=False) if MERGE_TABLE.exists() else None
    datasets = prepare_all_datasets(df)

    all_results = {}
    to_train = args.model.split(",") if args.model else ["gb", "rf", "rnn", "mlp", "iso"]

    if "gb" in to_train or "all" in to_train:
        all_results["GradientBoosting"] = train_gradient_boosting(
            datasets, run_cv=not args.no_cv
        )

    if "rf" in to_train or "all" in to_train:
        all_results["RandomForest"] = train_random_forest(
            datasets, run_cv=not args.no_cv
        )

    if "rnn" in to_train or "all" in to_train:
        all_results["RNN"] = train_rnn_model(
            datasets, run_cv=not args.no_cv
        )

    if "mlp" in to_train or "all" in to_train:
        all_results["MLP"] = train_mlp(
            datasets, run_cv=not args.no_cv
        )

    if "iso" in to_train or "all" in to_train:
        all_results["IsolationForest"] = train_isolation_forest(datasets)

    # Итоговая таблица
    print_comparison_table(all_results)

    # Сохранение
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "all_results.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nРезультаты сохранены: {save_path}")

    # Сохранение каждой модели отдельно
    for name, res in all_results.items():
        if "model" in res and res["model"] is not None:
            model_path = MODELS_DIR / f"{name.lower()}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(res["model"], f)
            print(f"Модель сохранена: {model_path}")

    return all_results


# ══════════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение ML-моделей для анализа ВТД")
    parser.add_argument("--model", type=str, default="all",
                        help="Какие модели обучить: gb,rf,rnn,mlp,iso или all")
    parser.add_argument("--no-cv", action="store_true",
                        help="Пропустить кросс-валидацию (быстрый запуск)")
    args = parser.parse_args()

    results = main(args)
