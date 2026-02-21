"""
run_pipeline.py — Запуск полного пайплайна одной командой

Этапы:
  1. Загрузка и очистка данных ВТД (4 пробега)
  2. Продольный матчинг дефектов
  3. Разведочный анализ (EDA)
  4. Обучение ML-моделей
  5. Генерация отчётов и графиков

Использование:
    python run_pipeline.py                    # Полный пайплайн
    python run_pipeline.py --skip-models      # Только EDA, без ML
    python run_pipeline.py --only-models      # Только ML (данные уже обработаны)
    python run_pipeline.py --model rnn,rf     # Только указанные модели
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.config import DATA_PROCESSED, OUTPUTS, MERGE_TABLE


def print_step(n: int, total: int, title: str):
    print(f"\n{'═'*65}")
    print(f"  ШАГ {n}/{total}: {title}")
    print(f"{'═'*65}")


def main():
    parser = argparse.ArgumentParser(description="Полный ML-пайплайн анализа ВТД")
    parser.add_argument("--skip-models", action="store_true",
                        help="Пропустить обучение ML-моделей")
    parser.add_argument("--only-models", action="store_true",
                        help="Запустить только ML (данные уже обработаны)")
    parser.add_argument("--model", type=str, default="all",
                        help="Какие модели: gb,rf,rnn,mlp,iso или all")
    parser.add_argument("--no-cv", action="store_true",
                        help="Без кросс-валидации (быстро)")
    args = parser.parse_args()

    t_start = time.time()
    TOTAL_STEPS = 4 if args.skip_models else 5

    # Создание директорий
    for d in [DATA_PROCESSED, OUTPUTS / "models", OUTPUTS / "plots"]:
        d.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────
    if not args.only_models:

        print_step(1, TOTAL_STEPS, "ЗАГРУЗКА И ОЧИСТКА ДАННЫХ ВТД")
        try:
            from src.data.loader import load_all_years
            datasets = load_all_years(save=True)
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            print("Убедитесь, что CSV-файлы находятся в data/raw/ и "
                  "настройте пути в src/utils/config.py")
            sys.exit(1)

        print_step(2, TOTAL_STEPS, "ПРОДОЛЬНЫЙ МАТЧИНГ ДЕФЕКТОВ")
        try:
            from src.data.matching import build_merge_table
            merge_df = build_merge_table(datasets)
            merge_df.to_csv(MERGE_TABLE, index=False)
            print(f"✅ Мерж-таблица сохранена: {MERGE_TABLE}")
        except Exception as e:
            print(f"❌ Ошибка матчинга: {e}")
            sys.exit(1)

        print_step(3, TOTAL_STEPS, "РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ")
        try:
            from src.eda.hotspot import HotspotAnalyzer
            import pandas as pd
            df = pd.read_csv(MERGE_TABLE, low_memory=False)
            analyzer = HotspotAnalyzer(df)
            analyzer.run_full_analysis(save_plots=True)
            print("✅ Графики горячей зоны сохранены")
        except Exception as e:
            print(f"⚠️ EDA завершился с ошибкой: {e} (продолжаем)")

    # ────────────────────────────────────────────────────
    if not args.skip_models:
        step_n = 1 if args.only_models else 4
        print_step(step_n, TOTAL_STEPS, "ОБУЧЕНИЕ ML-МОДЕЛЕЙ")
        try:
            from src.models.train_all import main as train_main

            class FakeArgs:
                model = args.model
                no_cv = args.no_cv

            results = train_main(FakeArgs())
            print("✅ Обучение завершено")

            # Итоговая таблица
            print("\n" + "─" * 55)
            print("ИТОГОВЫЕ МЕТРИКИ:")
            for name, r in results.items():
                if r.get("r2") is not None:
                    print(f"  {name:<22} R²={r['r2']:.4f}  RMSE={r['rmse']:.4f}%")
        except Exception as e:
            print(f"❌ Ошибка обучения моделей: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    # ────────────────────────────────────────────────────
    t_total = time.time() - t_start
    print(f"\n{'═'*65}")
    print(f"✅ ПАЙПЛАЙН ЗАВЕРШЁН за {t_total/60:.1f} минут")
    print(f"{'═'*65}")
    print(f"\nРезультаты:")
    print(f"  Данные:   {DATA_PROCESSED}/")
    print(f"  Модели:   {OUTPUTS}/models/")
    print(f"  Графики:  {OUTPUTS}/plots/")
    print(f"\nЗапуск приложения:")
    print(f"  streamlit run app/main.py")


if __name__ == "__main__":
    main()
