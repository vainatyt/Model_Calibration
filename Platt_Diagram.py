import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


# --- ФУНКЦИЯ ДЛЯ РАСЧЕТА ECE ---
def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # Определяем границы интервалов, как это делает calibration_curve(strategy='uniform')
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    # Итерируемся по каждому бину и считаем взвешенное отклонение
    for i in range(n_bins):
        # Находим индексы элементов, попавших в текущий интервал
        bin_idx = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        # Обрабатываем включение правой границы для самого последнего бина
        if i == n_bins - 1:
            bin_idx = bin_idx | (y_prob == bin_edges[i + 1])

        bin_size = np.sum(bin_idx)
        if bin_size > 0:
            # Разница между средней точностью и средней уверенностью в бине
            actual_accuracy = np.mean(y_true[bin_idx])
            predicted_confidence = np.mean(y_prob[bin_idx])
            ece += (bin_size / n_samples) * np.abs(
                actual_accuracy - predicted_confidence
            )

    return ece


# -------------------------------

# 1. Генерация данных
X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=15, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# 2. Обучение базовой модели (LinearSVC)
svm = LinearSVC(max_iter=10000, random_state=42)
svm.fit(X_train, y_train)

svm_scores = svm.decision_function(X_test)
svm_probs = (svm_scores - svm_scores.min()) / (
    svm_scores.max() - svm_scores.min()
)

# 3. Обучение калиброванной модели (Масштабирование Платта)
calibrated_svm = CalibratedClassifierCV(estimator=svm, method="sigmoid", cv=5)
calibrated_svm.fit(X_train, y_train)
platt_probs = calibrated_svm.predict_proba(X_test)[:, 1]

# 4. Расчет метрик (Brier, Log Loss, ECE)
brier_svm = brier_score_loss(y_test, svm_probs)
brier_platt = brier_score_loss(y_test, platt_probs)

loss_svm = log_loss(y_test, svm_probs)
loss_platt = log_loss(y_test, platt_probs)

ece_svm = expected_calibration_error(y_test, svm_probs, n_bins=10)
ece_platt = expected_calibration_error(y_test, platt_probs, n_bins=10)

# 5. Расчет точек для кривой калибровки
prob_true_svm, prob_pred_svm = calibration_curve(
    y_test, svm_probs, n_bins=10, strategy="uniform"
)
prob_true_platt, prob_pred_platt = calibration_curve(
    y_test, platt_probs, n_bins=10, strategy="uniform"
)

# 6. Отрисовка двухпанельного графика
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 9), gridspec_kw={"height_ratios":[3, 1]}
)

# --- ВЕРХНИЙ ГРАФИК: Кривые калибровки ---
# Диагональ идеальной калибровки
ax1.plot([0, 1], [0, 1], "k:", label="Идеальная калибровка")

# Линия без калибровки
ax1.plot(
    prob_pred_svm,
    prob_true_svm,
    "s-",
    color="red",
    label="Без калибровки",
)

# Линия после метода Платта
ax1.plot(
    prob_pred_platt,
    prob_true_platt,
    "o-",
    color="green",
    label="Метод Платта (Откалиброванная)",
)

# Настройка текста и осей верхней панели
ax1.set_ylabel("Доля истинных положительных ответов (True labels)")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.grid(True)

# Отображение метрик над верхним графиком
title_text = (
    "Сравнение качества калибровки вероятностей\n\n"
    f"Без калибровки:  Brier = {brier_svm:.4f}  |  Log Loss = {loss_svm:.4f}  |  ECE = {ece_svm:.4f}\n"
    f"Метод Платта:    Brier = {brier_platt:.4f}  |  Log Loss = {loss_platt:.4f}  |  ECE = {ece_platt:.4f}"
)
ax1.set_title(title_text, loc="left", fontsize=11, fontfamily="monospace")


# --- НИЖНИЙ ГРАФИК: Распределение предсказанных вероятностей ---
hist_kwargs = dict(bins=10, histtype="step", lw=2)

ax2.hist(svm_probs, color="red", label="Без калибровки", **hist_kwargs)
ax2.hist(
    platt_probs,
    color="green",
    label="Метод Платта (Откалиброванная)",
    **hist_kwargs,
)

ax2.set_xlabel("Предсказанная вероятность (Predicted probability)")
ax2.set_ylabel("Количество объектов")
ax2.legend(loc="upper center", ncol=2)
ax2.grid(True)

plt.tight_layout()
plt.show()
