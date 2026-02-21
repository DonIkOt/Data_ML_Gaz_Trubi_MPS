"""
tests/test_matching.py — Тесты алгоритма матчинга дефектов
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.matching import match_two_years


def make_test_df(positions, pipe_nums=None):
    if pipe_nums is None:
        pipe_nums = [f"P{i}" for i in range(len(positions))]
    return pd.DataFrame({
        "dist_ref": [p * 1000 for p in positions],  # в мм
        "pipe_num": pipe_nums,
    })


def test_exact_match():
    """Точное совпадение по позиции и номеру трубы."""
    df1 = make_test_df([100, 200, 300], ["P1", "P2", "P3"])
    df2 = make_test_df([100, 200, 300], ["P1", "P2", "P3"])
    matches = match_two_years(df1, df2, 2020, 2022, tolerance_m=1.0)
    assert len(matches) == 3, f"Ожидалось 3 совпадения, получено {len(matches)}"
    print("✅ test_exact_match пройден")


def test_tolerance_match():
    """Совпадение в пределах допуска 1м."""
    df1 = make_test_df([100.0], ["P1"])
    df2 = make_test_df([100.8], ["P1"])  # 0.8м — в пределах допуска
    matches = match_two_years(df1, df2, 2020, 2022, tolerance_m=1.0)
    assert len(matches) == 1
    print("✅ test_tolerance_match пройден")


def test_no_match_outside_tolerance():
    """Нет совпадений за пределами допуска."""
    df1 = make_test_df([100.0], ["P1"])
    df2 = make_test_df([102.0], ["P1"])  # 2м — вне допуска
    matches = match_two_years(df1, df2, 2020, 2022, tolerance_m=1.0)
    assert len(matches) == 0
    print("✅ test_no_match_outside_tolerance пройден")


def test_pipe_num_filter():
    """Нет совпадений при несовпадении номера трубы."""
    df1 = make_test_df([100.0], ["P1"])
    df2 = make_test_df([100.0], ["P2"])  # другой номер трубы
    matches = match_two_years(df1, df2, 2020, 2022,
                               tolerance_m=1.0, require_pipe_match=True)
    assert len(matches) == 0
    print("✅ test_pipe_num_filter пройден")


def test_closest_match_selected():
    """При нескольких кандидатах — выбирается ближайший."""
    df1 = make_test_df([100.0], ["P1"])
    df2 = make_test_df([100.3, 100.8], ["P1", "P1"])
    matches = match_two_years(df1, df2, 2020, 2022, tolerance_m=1.0)
    assert len(matches) == 1
    assert matches.iloc[0]["dist_diff_m"] < 0.4  # выбран ближайший (0.3м)
    print("✅ test_closest_match_selected пройден")


if __name__ == "__main__":
    test_exact_match()
    test_tolerance_match()
    test_no_match_outside_tolerance()
    test_pipe_num_filter()
    test_closest_match_selected()
    print("\n🎉 Все тесты пройдены!")
