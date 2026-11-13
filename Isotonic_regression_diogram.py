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


# 1. Подготовка данных
X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=15, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# 2. Обучение базовой модели и получение "псевдо-вероятностей"
svm = LinearSVC(max_iter=10000, random_state=42)
svm.fit(X_train, y_train)

svm_scores = svm.decision_function(X_test)
svm_probs = (svm_scores - svm_scores.min()) / (
    svm_scores.max() - svm_scores.min()
)

# 3. Калибровка Изотонической регрессией (Isotonic)
calibrated_iso = CalibratedClassifierCV(estimator=svm, method="isotonic", cv=5)
calibrated_iso.fit(X_train, y_train)
iso_probs = calibrated_iso.predict_proba(X_test)[:, 1]

# 4. Расчет метрик качества
metrics = {
    "Без калибровки": {
        "brier": brier_score_loss(y_test, svm_probs),
        "loss": log_loss(y_test, svm_probs),
        "ece": expected_calibration_error(y_test, svm_probs),
    },
    "Изотоническая регрессия": {
        "brier": brier_score_loss(y_test, iso_probs),
        "loss": log_loss(y_test, iso_probs),
        "ece": expected_calibration_error(y_test, iso_probs),
    },
}

# 5. Расчет точек кривой калибровки
prob_true_svm, prob_pred_svm = calibration_curve(
    y_test, svm_probs, n_bins=10, strategy="uniform"
)
prob_true_iso, prob_pred_iso = calibration_curve(
    y_test, iso_probs, n_bins=10, strategy="uniform"
)

# 6. Построение двухпанельного графика
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 9), gridspec_kw={"height_ratios":[3,1]}
)

# --- ВЕРХНИЙ ГРАФИК: Кривые калибровки ---
ax1.plot([0, 1], [0, 1], "k:", label="Идеальная калибровка")
ax1.plot(prob_pred_svm, prob_true_svm, "s-", color="red", label="Без калибровки")
ax1.plot(
    prob_pred_iso,
    prob_true_iso,
    "^-",
    color="blue",
    label="Изотоническая регрессия (Откалиброванная)",
)

ax1.set_ylabel("Доля истинных положительных ответов (True labels)")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.grid(True)

# Вынос метрик из словаря
b_none = metrics["Без калибровки"]["brier"]
l_none = metrics["Без калибровки"]["loss"]
e_none = metrics["Без калибровки"]["ece"]

b_iso = metrics["Изотоническая регрессия"]["brier"]
l_iso = metrics["Изотоническая регрессия"]["loss"]
e_iso = metrics["Изотоническая регрессия"]["ece"]

# Создаем f-строку заголовка
title_text = (
    "Калибровка вероятностей методом Isotonic Regression\n\n"
    f"Без калибровки:  Brier = {b_none:.4f}  |  Log Loss = {l_none:.4f}  |  ECE = {e_none:.4f}\n"
    f"Изотоническая:   Brier = {b_iso:.4f}  |  Log Loss = {l_iso:.4f}  |  ECE = {e_iso:.4f}"
)
ax1.set_title(title_text, loc="left", fontsize=11, fontfamily="monospace")


# --- НИЖНИЙ ГРАФИК: Распределение предсказанных вероятностей ---
hist_kwargs = dict(bins=10, histtype="step", lw=2)

# Добавлена красная гистограмма для сырых вероятностей
ax2.hist(svm_probs, color="red", label="Без калибровки", **hist_kwargs)

# Синяя гистограмма для изотонической регрессии
ax2.hist(
    iso_probs,
    color="blue",
    label="Изотоническая регрессия (Откалиброванная)",
    **hist_kwargs,
)

ax2.set_xlabel("Предсказанная вероятность (Predicted probability)")
ax2.set_ylabel("Количество объектов")
ax2.legend(loc="upper center", ncol=2)  # Добавлен ncol=2 для размещения в один ряд
ax2.grid(True)

plt.tight_layout()
plt.show()
