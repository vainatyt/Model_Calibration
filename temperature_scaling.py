import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


# --- 1. ФУНКЦИЯ РАСЧЕТА ECE ДЛЯ МНОГОКЛАССОВОГО / БИНАРНОГО СЛУЧАЯ ---
def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_idx = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            bin_idx = bin_idx | (y_prob == bin_edges[i + 1])

        bin_size = np.sum(bin_idx)
        if bin_size > 0:
            actual_accuracy = np.mean(y_true[bin_idx])
            predicted_confidence = np.mean(y_prob[bin_idx])
            ece += (bin_size / n_samples) * np.abs(
                actual_accuracy - predicted_confidence
            )
    return ece


# --- 2. РЕАЛИЗАЦИЯ TEMPERATURE SCALING ---
class TemperatureScaler:

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, y_true):
        logits_t = torch.from_numpy(logits).float()
        y_true_t = torch.from_numpy(y_true).long()

        def objective(t):
            t_val = max(t, 1e-3)
            return F.cross_entropy(logits_t / t_val, y_true_t).item()

        result = minimize(
            objective, [1.0], method="Nelder-Mead", bounds=[(1e-3, 10.0)]
        )
        self.temperature = max(result.x[0], 1e-3)
        return self

    def predict_proba(self, logits):
        logits_t = torch.from_numpy(logits).float()
        with torch.no_grad():
            probs = F.softmax(logits_t / self.temperature, dim=1)
        return probs.numpy()


# --- 3. ПОДГОТОВКА ДАННЫХ (Симуляция избыточно уверенной нейросети) ---
np.random.seed(42)
n_samples = 5000

# Истинные метки (бинарная классификация для простоты визуализации кривой)
y_val = np.random.randint(0, 2, size=n_samples)
y_test = np.random.randint(0, 2, size=n_samples)

# Генерируем случайные сырые логиты (имитируем выходы нейросети)
# Убираем жесткое умножение и искусственный сдвиг классов, 
# чтобы логиты распределялись естественно для классической модели
val_logits = np.random.normal(0, 1.5, size=(n_samples, 2))
val_logits[y_val == 1, 1] += 1.5 

test_logits = np.random.normal(0, 1.5, size=(n_samples, 2))
test_logits[y_test == 1, 1] += 1.5

# Сырые вероятности без калибровки получаем ЧЕРЕЗ SOFTMAX от сырых логитов
# (А не через Мин-Макс сжатие отступов SVM!)
raw_probs = F.softmax(torch.from_numpy(test_logits).float(), dim=1).numpy()[:, 1]

# --- 4. КАЛИБРОВКА ---
# Сырые вероятности без калибровки (обычный Softmax)
raw_probs = (
    F.softmax(torch.from_numpy(test_logits).float(), dim=1).numpy()[:, 1]
)

# Настройка температуры на валидационной выборке и применение на тесте
scaler = TemperatureScaler()
scaler.fit(val_logits, y_val)
calibrated_probs = scaler.predict_proba(test_logits)[:, 1]

# --- 5. РАСЧЕТ МЕТРИК ---
metrics = {
    "Без калибровки": {
        "brier": brier_score_loss(y_test, raw_probs),
        "loss": log_loss(y_test, raw_probs),
        "ece": expected_calibration_error(y_test, raw_probs),
    },
    "Temperature Scaling": {
        "brier": brier_score_loss(y_test, calibrated_probs),
        "loss": log_loss(y_test, calibrated_probs),
        "ece": expected_calibration_error(y_test, calibrated_probs),
    },
}

# Расчет точек для диаграммы
prob_true_raw, prob_pred_raw = calibration_curve(
    y_test, raw_probs, n_bins=10, strategy="uniform"
)
prob_true_cal, prob_pred_cal = calibration_curve(
    y_test, calibrated_probs, n_bins=10, strategy="uniform"
)

# --- 6. ПОСТРОЕНИЕ ДВУХПАНЕЛЬНОГО ГРАФИКА ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [3,1]}
)

# Верхняя панель: Кривые калибровки
ax1.plot([0,1],[0,1], "k:", label="Идеальная калибровка")
ax1.plot(
    prob_pred_raw, prob_true_raw, "s-", color="red", label="Без калибровки"
)
ax1.plot(
    prob_pred_cal,
    prob_true_cal,
    "^-",
    color="purple",
    label=f"Temperature Scaling (T={scaler.temperature:.2f})",
)

ax1.set_ylabel("Доля истинных положительных ответов (True labels)")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.grid(True)

# Формирование текстового заголовка с метриками качества
b_raw, l_raw, e_raw = (
    metrics["Без калибровки"]["brier"],
    metrics["Без калибровки"]["loss"],
    metrics["Без калибровки"]["ece"],
)
b_cal, l_cal, e_cal = (
    metrics["Temperature Scaling"]["brier"],
    metrics["Temperature Scaling"]["loss"],
    metrics["Temperature Scaling"]["ece"],
)

title_text = (
    "Калибровка вероятностей методом Temperature Scaling\n\n"
    f"Без калибровки:  Brier = {b_raw:.4f}  |  Log Loss = {l_raw:.4f}  |  ECE = {e_raw:.4f}\n"
    f"Temperature:     Brier = {b_cal:.4f}  |  Log Loss = {l_cal:.4f}  |  ECE = {e_cal:.4f}"
)
ax1.set_title(title_text, loc="left", fontsize=11, fontfamily="monospace")

# Нижняя панель: Гистограммы распределения предсказаний
ax2.hist(
    raw_probs,
    bins=10,
    label="Без калибровки (Переуверенная модель)",
    color="red",
    histtype="step",
    lw=2,
)
ax2.hist(
    calibrated_probs,
    bins=10,
    label="После Temperature Scaling (Сглаженная)",
    color="purple",
    histtype="step",
    lw=2,
)
ax2.set_xlabel("Предсказанная вероятность (Predicted probability)")
ax2.set_ylabel("Количество объектов")
ax2.legend(loc="upper center")
ax2.grid(True)

plt.tight_layout()
plt.show()
