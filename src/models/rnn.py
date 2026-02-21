"""
src/models/rnn.py — Elman RNN для регрессии (чистый NumPy)

Реализация рекуррентной нейронной сети без внешних DL-фреймворков.
Обучение через BPTT (backpropagation through time).
Оптимизатор: Adam с gradient clipping и LR decay.

Причина выбора Elman RNN вместо LSTM:
  - При T=3 временных шагах LSTM избыточен (cell state не успевает накопить информацию)
  - Более стабильный градиентный поток на малых последовательностях
  - Достаточно для захвата тренда за 3 точки

Использование:
    from src.models.rnn import ElmanRNN, train_rnn, evaluate_rnn
    
    model = ElmanRNN(T=3, D=6, H=24)
    model, history = train_rnn(model, X_train, y_train)
    metrics = evaluate_rnn(model, X_test, y_test, y_min, y_range)
"""

import numpy as np
import copy
from typing import Optional
from pathlib import Path
import pickle


class ElmanRNN:
    """
    Elman RNN для задачи регрессии.
    
    Архитектура:
      Вход (T × D) → Рекуррентный слой (H нейронов, tanh) → Линейный выход (1)
    
    Parameters
    ----------
    T : int
        Количество временных шагов
    D : int
        Размерность входного вектора на каждом шаге
    H : int
        Размер скрытого слоя
    lr : float
        Начальная скорость обучения
    clip : float
        Порог gradient clipping (по норме)
    seed : int
        Инициализация генератора случайных чисел
    """

    def __init__(self, T: int, D: int, H: int = 24,
                 lr: float = 8e-4, clip: float = 1.0, seed: int = 42):
        self.T = T; self.D = D; self.H = H
        self.lr = lr; self.clip = clip

        rng = np.random.RandomState(seed)
        s = 0.05  # Малая инициализация — ключ к стабильности

        # Веса рекуррентного слоя: h_t = tanh(Wx @ x_t + Wh @ h_{t-1} + bh)
        self.Wx = rng.randn(H, D) * s
        self.Wh = rng.randn(H, H) * s
        self.bh = np.zeros(H)

        # Выходной слой: y = Wo @ h_T + bo
        self.Wo = rng.randn(1, H) * s
        self.bo = np.zeros(1)

        # Adam состояние
        self._param_names = ["Wx", "Wh", "bh", "Wo"]
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._param_names}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._param_names}
        self._t = 0  # счётчик шагов Adam

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(np.clip(x, -6, 6))

    @staticmethod
    def _dtanh(y: np.ndarray) -> np.ndarray:
        """Производная tanh(x) через выход y = tanh(x): 1 - y²"""
        return 1 - y ** 2

    def forward(self, xs: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Прямой проход.
        
        Parameters
        ----------
        xs : np.ndarray, shape (T, D)
        
        Returns
        -------
        (output, cache)
            output: float — предсказанное значение
            cache: list[(x_t, h_prev, h_t)] для каждого шага
        """
        h = np.zeros(self.H)
        cache = []
        for t in range(self.T):
            x = xs[t]
            h_new = self._tanh(self.Wx @ x + self.Wh @ h + self.bh)
            cache.append((x.copy(), h.copy(), h_new.copy()))
            h = h_new
        out = (self.Wo @ h + self.bo)[0]
        return out, cache

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Предсказание для батча. X shape: (n, T, D)"""
        return np.array([self.forward(x)[0] for x in X])

    def _adam_update(self, name: str, param: np.ndarray, grad: np.ndarray,
                     b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> np.ndarray:
        """Один шаг Adam."""
        self._t += 1
        # Gradient clipping по норме
        norm = np.linalg.norm(grad)
        if norm > self.clip:
            grad = grad * (self.clip / norm)
        # Adam
        m = b1 * self._m[name] + (1 - b1) * grad
        v = b2 * self._v[name] + (1 - b2) * grad ** 2
        self._m[name] = m; self._v[name] = v
        m_hat = m / (1 - b1 ** self._t)
        v_hat = v / (1 - b2 ** self._t)
        self._t -= 1  # скорректируем перед возвратом
        return param - self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def train_epoch(self, X: np.ndarray, y: np.ndarray,
                    batch_size: int = 32) -> float:
        """
        Одна эпоха обучения (SGD + BPTT).
        
        Parameters
        ----------
        X : np.ndarray, shape (n, T, D) — нормализованные последовательности
        y : np.ndarray, shape (n,) — нормализованные целевые значения
        batch_size : int
        
        Returns
        -------
        float — средний MSE loss
        """
        idx = np.random.permutation(len(X))
        total_loss = 0.0

        for start in range(0, len(X), batch_size):
            batch = idx[start:start + batch_size]
            Xb, yb = X[batch], y[batch]
            n = len(batch)

            # Накопление градиентов по батчу
            dWx = np.zeros_like(self.Wx)
            dWh = np.zeros_like(self.Wh)
            dbh = np.zeros_like(self.bh)
            dWo = np.zeros_like(self.Wo)
            dbo = np.zeros_like(self.bo)

            for xi, yi in zip(Xb, yb):
                pred, cache = self.forward(xi)
                err = pred - yi
                total_loss += 0.5 * err ** 2

                # Градиент выходного слоя
                h_last = cache[-1][2]
                dWo += err * h_last[np.newaxis, :]
                dbo += np.array([err])

                # BPTT
                dh = (self.Wo.T * err).ravel()  # shape: (H,)
                for t in reversed(range(self.T)):
                    x_t, h_prev, h_t = cache[t]
                    delta = dh * self._dtanh(h_t)
                    dWx += np.outer(delta, x_t)
                    dWh += np.outer(delta, h_prev)
                    dbh += delta
                    dh = self.Wh.T @ delta

            # Adam update (инкремент t один раз за батч)
            self._t -= 1
            self.Wx = self._adam_update("Wx", self.Wx, dWx / n)
            self.Wh = self._adam_update("Wh", self.Wh, dWh / n)
            self.bh = self._adam_update("bh", self.bh, dbh / n)
            self.Wo = self._adam_update("Wo", self.Wo, dWo / n)
            self.bo -= self.lr * dbo / n
            self._t += 1

        return total_loss / len(X)

    def save(self, path: Path):
        """Сохранить модель."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "ElmanRNN":
        """Загрузить модель."""
        with open(path, 'rb') as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════════════
# Функции обучения и оценки
# ══════════════════════════════════════════════════════════════════

def train_rnn(
    model: ElmanRNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    y_min: float = 0.0,
    y_range: float = 1.0,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr_decay: float = 3e-4,
    lr_decay_epoch: int = 50,
    verbose: bool = True,
) -> tuple["ElmanRNN", dict]:
    """
    Полное обучение RNN.
    
    Parameters
    ----------
    model : ElmanRNN
    X_train : np.ndarray, shape (n, T, D) — нормализованные входы
    y_train : np.ndarray, shape (n,) — нормализованные цели
    X_val, y_val : optional validation set (y_val — НЕ нормированный, реальные значения)
    y_min, y_range : параметры нормализации цели
    n_epochs : int
    batch_size : int
    lr_decay : float — LR после lr_decay_epoch
    lr_decay_epoch : int
    verbose : bool
    
    Returns
    -------
    (best_model, history_dict)
    """
    from sklearn.metrics import r2_score, mean_squared_error

    history = {"train_loss": [], "val_r2": [], "val_rmse": []}
    best_r2 = -999.0
    best_model = copy.deepcopy(model)

    for epoch in range(1, n_epochs + 1):
        # LR decay
        if epoch == lr_decay_epoch:
            model.lr = lr_decay
            if verbose:
                print(f"  [ep {epoch}] LR decay → {lr_decay}")

        loss = model.train_epoch(X_train, y_train, batch_size)
        history["train_loss"].append(loss)

        if np.isnan(loss):
            print(f"  NaN loss на эпохе {epoch}. Остановка.")
            break

        # Валидация
        if X_val is not None and y_val is not None:
            y_pred_norm = model.predict_batch(X_val)
            y_pred = y_pred_norm * y_range + y_min
            r2   = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            history["val_r2"].append(r2)
            history["val_rmse"].append(rmse)

            if r2 > best_r2:
                best_r2 = r2
                best_model = copy.deepcopy(model)

            if verbose and epoch % 20 == 0:
                print(f"  [ep {epoch:3d}] loss={loss:.5f} | val R²={r2:.4f} RMSE={rmse:.4f}%")
        elif verbose and epoch % 20 == 0:
            print(f"  [ep {epoch:3d}] loss={loss:.5f}")

    if verbose:
        print(f"\nЛучший val R² = {best_r2:.4f}")

    return best_model, history


def evaluate_rnn(
    model: ElmanRNN,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_min: float,
    y_range: float,
) -> dict:
    """
    Оценка модели на тестовой выборке.
    
    Parameters
    ----------
    X_test : shape (n, T, D)
    y_test : реальные значения (не нормированные)
    y_min, y_range : параметры нормализации цели
    
    Returns
    -------
    dict с метриками: rmse, mae, r2, y_pred
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    y_pred_norm = model.predict_batch(X_test)
    y_pred = y_pred_norm * y_range + y_min

    return {
        "rmse":   np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae":    mean_absolute_error(y_test, y_pred),
        "r2":     r2_score(y_test, y_pred),
        "y_pred": y_pred,
        "y_test": y_test,
    }


# ══════════════════════════════════════════════════════════════════
# Точка входа (демонстрация)
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.models.data_prep import prepare_all_datasets
    from src.utils.config import RNN_PARAMS, MODELS_DIR

    print("Подготовка данных для RNN...")
    datasets = prepare_all_datasets()
    rnn_data = datasets["rnn"]

    X_tr = rnn_data.X_train
    y_tr = rnn_data.y_train
    X_te = rnn_data.X_test
    y_te = rnn_data.y_test
    y_min   = rnn_data.extra["y_min"]
    y_range = rnn_data.extra["y_range"]

    T = X_tr.shape[1]; D = X_tr.shape[2]
    print(f"\nАрхитектура: T={T}, D={D}, H={RNN_PARAMS['hidden_dim']}")

    model = ElmanRNN(T=T, D=D, **{k: v for k, v in RNN_PARAMS.items()
                                   if k in ["hidden_dim", "lr_initial", "clip", "seed"]
                                   and k.replace("_initial", "")},
                     H=RNN_PARAMS["hidden_dim"],
                     lr=RNN_PARAMS["lr_initial"])

    print("\nОбучение RNN...")
    best_model, history = train_rnn(
        model, X_tr, y_tr, X_te, y_te, y_min, y_range,
        n_epochs=RNN_PARAMS["n_epochs"],
        batch_size=RNN_PARAMS["batch_size"],
        lr_decay=RNN_PARAMS["lr_decay"],
        lr_decay_epoch=RNN_PARAMS["lr_decay_epoch"],
    )

    metrics = evaluate_rnn(best_model, X_te, y_te, y_min, y_range)
    print(f"\nТестовые метрики:")
    print(f"  R²   = {metrics['r2']:.4f}")
    print(f"  RMSE = {metrics['rmse']:.4f}%")
    print(f"  MAE  = {metrics['mae']:.4f}%")

    # Сохранение
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model.save(MODELS_DIR / "rnn_model.pkl")
    print(f"\nМодель сохранена: {MODELS_DIR}/rnn_model.pkl")
