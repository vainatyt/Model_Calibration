import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt

# Функция расчета Expected Calibration Error (ECE)
def calculate_ece(y_true, y_prob, n_bins=5):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if np.sum(mask) > 0:
            bin_conf = np.mean(y_prob[mask])
            bin_acc = np.mean(y_true[mask])
            ece += np.abs(bin_conf - bin_acc) * np.sum(mask)
    return ece / len(y_true)

# Исходный датасет
dataset_dict = {
    'Outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast','sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy','sunny', 'overcast', 'rainy', 'sunny', 'sunny', 'rainy', 'overcast','rainy', 'sunny', 'overcast', 'sunny', 'overcast', 'rainy', 'overcast'],
    'Temperature': [85.0, 80.0, 83.0, 70.0, 68.0, 65.0, 64.0, 72.0, 69.0, 75.0, 75.0,72.0, 81.0, 71.0, 81.0, 74.0, 76.0, 78.0, 82.0, 67.0, 85.0, 73.0,88.0, 77.0, 79.0, 80.0, 66.0, 84.0],
    'Humidity': [85.0, 90.0, 78.0, 96.0, 80.0, 70.0, 65.0, 95.0, 70.0, 80.0, 70.0,90.0, 75.0, 80.0, 88.0, 92.0, 85.0, 75.0, 92.0, 90.0, 85.0, 88.0,65.0, 70.0, 60.0, 95.0, 70.0, 78.0],
    'Wind': [False, True, False, False, False, True, True, False, False, False, True,True, False, True, True, False, False, True, False, True, True, False,True, False, False, True, False, False],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes','Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes','Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes']
}

# Подготовка данных
df = pd.DataFrame(dataset_dict)
df = pd.get_dummies(df, columns=['Outlook'], prefix='', prefix_sep='', dtype=int)
df['Wind'] = df['Wind'].astype(int)
df['Play'] = (df['Play'] == 'Yes').astype(int)
df = df[['sunny', 'overcast', 'rainy', 'Temperature', 'Humidity', 'Wind', 'Play']]

X, y = df.drop('Play', axis=1), df['Play']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, shuffle=False)

scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[['Temperature', 'Humidity']] = scaler.fit_transform(X_train[['Temperature', 'Humidity']])
X_test[['Temperature', 'Humidity']] = scaler.transform(X_test[['Temperature', 'Humidity']])

# Инициализация моделей
models = {
    'k-Nearest Neighbors': KNeighborsClassifier(n_neighbors=4, weights='distance'),
    'Bernoulli Naive Bayes': BernoulliNB(),
    'Logistic Regression': LogisticRegression(C=1.5, random_state=42),
    'Multilayer Perceptron': MLPClassifier(hidden_layer_sizes=(4, 2), random_state=42, max_iter=2000)
}

# Сбор результатов предсказаний
results = {}
for name, model in models.items():
    # Без калибровки (Без изменений)
    model.fit(X_train, y_train)
    prob_uncal = model.predict_proba(X_test)[:, 1]
    
    # С калибровкой (Синие графики)
    cal_model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
    cal_model.fit(X_train, y_train)
    prob_cal = cal_model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'uncal': {
            'Brier': brier_score_loss(y_test, prob_uncal),
            'LogLoss': log_loss(y_test, prob_uncal),
            'ECE': calculate_ece(y_test, prob_uncal),
            'prob': prob_uncal
        },
        'cal': {
            'Brier': brier_score_loss(y_test, prob_cal),
            'LogLoss': log_loss(y_test, prob_cal),
            'ECE': calculate_ece(y_test, prob_cal),
            'prob': prob_cal
        }
    }

# Цикл по моделям — теперь каждая модель создает свою отдельную Figure
for name, data in results.items():
    # ИЗМЕНЕНО: Создаем новую отдельную фигуру с сеткой 1 строка х 2 колонки для текущего алгоритма
    fig, (ax_uncal, ax_cal) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"Model: {name}", fontsize=12, fontweight='bold')
    
    # ---- Левый график: ДО калибровки (Красный, без изменений) ----
    ax_uncal.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    p_true_uncal, p_pred_uncal = calibration_curve(y_test, data['uncal']['prob'], n_bins=5, strategy='uniform')
    ax_uncal.plot(p_pred_uncal, p_true_uncal, color='orangered', marker='o', label='Before Calibration', linewidth=2)
    
    title_uncal = f"Uncalibrated\nBrier: {data['uncal']['Brier']:.3f} | LogLoss: {data['uncal']['LogLoss']:.3f} | ECE: {data['uncal']['ECE']:.3f}"
    ax_uncal.set_title(title_uncal, fontsize=9, pad=6)
    ax_uncal.set_xlabel('Predicted Probability', fontsize=8)
    ax_uncal.set_ylabel('True Probability', fontsize=8)
    ax_uncal.grid(True, alpha=0.3)
    ax_uncal.legend(fontsize=8, loc='upper left')
    ax_uncal.set_xlim([-0.05, 1.05])
    ax_uncal.set_ylim([-0.05, 1.05])
    
    # ---- Правый график: ПОСЛЕ калибровки (Синий, измененный) ----
    ax_cal.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    p_true_cal, p_pred_cal = calibration_curve(y_test, data['cal']['prob'], n_bins=5, strategy='uniform')
    ax_cal.plot(p_pred_cal, p_true_cal, color='dodgerblue', marker='^', label='After Calibration', linewidth=2)
    
    title_cal = f"Calibrated (Prefit)\nBrier: {data['cal']['Brier']:.3f} | LogLoss: {data['cal']['LogLoss']:.3f} | ECE: {data['cal']['ECE']:.3f}"
    ax_cal.set_title(title_cal, fontsize=9, pad=6)
    ax_cal.set_xlabel('Predicted Probability', fontsize=8)
    ax_cal.set_ylabel('True Probability', fontsize=8)
    ax_cal.grid(True, alpha=0.3)
    ax_cal.legend(fontsize=8, loc='upper left')
    ax_cal.set_xlim([-0.05, 1.05])
    ax_cal.set_ylim([-0.05, 1.05])

    plt.tight_layout()

# Показываем все 4 фигуры одновременно
plt.show()
