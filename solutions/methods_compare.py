import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot


def alt_estimator(X, Y, n=100):
    k = n // 3
    X_lower = X[:k].mean()
    Y_lower = Y[:k].mean()
    X_upper = X[-k:].mean()
    Y_upper = Y[-k:].mean()
    beta1_alt = (Y_upper - Y_lower) / (X_upper - X_lower)
    beta0_alt = Y.mean() - beta1_alt * X.mean()
    return beta0_alt, beta1_alt

def method_of_moments(X, Y):
    beta1_mm = (np.sum((X - X.mean()) * (Y - Y.mean()))) / (np.sum((X - X.mean()) ** 2))
    beta0_mm = Y.mean() - beta1_mm * X.mean()
    return beta0_mm, beta1_mm

def mnk_estimator(X, Y):
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    beta1_mnk = (np.sum((X - X_mean) * (Y - Y_mean))) / (np.sum((X - X_mean) ** 2))
    beta0_mnk = Y_mean - beta1_mnk * X_mean

    return beta0_mnk, beta1_mnk

def manual_variance(data, mean_val):
    N = len(data)
    sum_sq = 0.0
    for j in range(N):
        sum_sq += (data[j] - mean_val) ** 2
    return sum_sq / (N - 1)


def manual_mse(data, true_value):
    N = len(data)
    sum_sq = 0.0
    for j in range(N):
        sum_sq += (data[j] - true_value) ** 2
    return sum_sq / N


X = np.random.uniform(0, 10, 100)
beta0_true = 1
beta1_true = 2.5
sigma = 2

results = {'mnk': [], 'alt': [], 'mm': []}

for i in range(10000):
    epsilon = np.random.normal(0, sigma, 100)
    Y = beta0_true + beta1_true * X + epsilon

    beta0_mnk, beta1_mnk = mnk_estimator(X, Y)

    beta0_alt, beta1_alt = alt_estimator(X, Y)

    beta0_mm, beta1_mm = method_of_moments(X, Y)

    results['mnk'].append([beta0_mnk, beta1_mnk])
    results['alt'].append([beta0_alt, beta1_alt])
    results['mm'].append([beta0_mm, beta1_mm])

results_mnk = np.array(results['mnk'])
results_alt = np.array(results['alt'])
results_mm = np.array(results['mm'])

beta1_mnk_vals = results_mnk[:, 1]
beta1_alt_vals = results_alt[:, 1]
beta1_mm_vals = results_mm[:, 1]

mean_beta1_mnk = beta1_mnk_vals.mean()
mean_beta1_alt = beta1_alt_vals.mean()
mean_beta1_mm = beta1_mm_vals.mean()

print("1. Проверка на несмещенность:")
print(f"Среднее beta1 МНК: {mean_beta1_mnk:.6f} (смещение: {mean_beta1_mnk - beta1_true:.6f})")
print(f"Среднее beta1 альт: {mean_beta1_alt:.6f} (смещение: {mean_beta1_alt - beta1_true:.6f})")
print(f"Среднее beta1 ММ: {mean_beta1_mm:.6f} (смещение: {mean_beta1_mm - beta1_true:.6f})")
print(f"Истинное значение: {beta1_true:.6f}")

print("2. Проверка на эффективность:")
var_beta1_mnk = manual_variance(beta1_mnk_vals, mean_beta1_mnk)
var_beta1_alt = manual_variance(beta1_alt_vals, mean_beta1_alt)
var_beta1_mm = manual_variance(beta1_mm_vals, mean_beta1_mm)

print(f"Дисперсия beta1 МНК: {var_beta1_mnk:.6f}")
print(f"Дисперсия beta1 альт: {var_beta1_alt:.6f}")
print(f"Дисперсия beta1 ММ: {var_beta1_mm:.6f}")

print("3. Среднеквадратическая ошибка:")
mse_beta1_mnk = manual_mse(beta1_mnk_vals, beta1_true)
mse_beta1_alt = manual_mse(beta1_alt_vals, beta1_true)
mse_beta1_mm = manual_mse(beta1_mm_vals, beta1_true)

print(f"MSE beta1 МНК: {mse_beta1_mnk:.6f}")
print(f"MSE beta1 альт: {mse_beta1_alt:.6f}")
print(f"MSE beta1 ММ: {mse_beta1_mm:.6f}")

print("4. Относительная эффективность:")
eff_alt = var_beta1_mnk / var_beta1_alt
eff_mm = var_beta1_mnk / var_beta1_mm
print(f"Эффективность альт метода: {eff_alt:.4f}")
print(f"Эффективность ММ метода: {eff_mm:.4f}")


print("5. Доверительные интервалы:")

def calculate_coverage(beta1_estimates, true_value, confidence=0.95):
    N = len(beta1_estimates)
    if N == 0:
        return 0.0

    coverage_count = 0

    for i in range(N):
        estimate = beta1_estimates[i]
        Sxx = np.sum((X - np.mean(X)) ** 2)
        theoretical_se = sigma / np.sqrt(Sxx)

        z_value = 1.96
        margin = z_value * theoretical_se
        ci_low = estimate - margin
        ci_high = estimate + margin

        if ci_low <= true_value <= ci_high:
            coverage_count += 1

    coverage_percentage = (coverage_count / N) * 100
    return coverage_percentage

coverage_mnk = calculate_coverage(beta1_mnk_vals, beta1_true)
coverage_alt = calculate_coverage(beta1_alt_vals, beta1_true)
coverage_mm = calculate_coverage(beta1_mm_vals, beta1_true)

print(f"МНК: {coverage_mnk:.2f}% покрытия")
print(f"Альт: {coverage_alt:.2f}% покрытия")
print(f"ММ: {coverage_mm:.2f}% покрытия")

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(beta1_mnk_vals, alpha=0.7, label='МНК', density=True, color='blue')
plt.hist(beta1_alt_vals, alpha=0.7, label='Альтернативный', density=True, color='red')
plt.hist(beta1_mm_vals, alpha=0.7, label='Метод моментов', density=True, color='green')
plt.axvline(beta1_true, color='black', linestyle='--', linewidth=2, label='Истинное значение')
plt.xlabel('beta1')
plt.ylabel('Плотность')
plt.title('Распределения оценок beta1')
plt.legend()

plt.subplot(2, 3, 2)
box_data = [beta1_mnk_vals, beta1_alt_vals, beta1_mm_vals]
plt.boxplot(box_data, tick_labels=['МНК', 'Альтернативный', 'Метод моментов'])
plt.axhline(beta1_true, color='black', linestyle='--', linewidth=2, label='Истинное значение')
plt.title('Разброс оценок beta1')
plt.ylabel('beta1')

plt.subplot(2, 3, 3)
probplot(beta1_mnk_vals, dist="norm", plot=plt)
plt.title('Q-Q plot: МНК оценки')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
probplot(beta1_alt_vals, dist="norm", plot=plt)
plt.title('Q-Q plot: Альтернативные оценки')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
probplot(beta1_mm_vals, dist="norm", plot=plt)
plt.title('Q-Q plot: Метод моментов')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
