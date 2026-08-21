import numpy as np
import scipy.stats as stats

data_array = np.array([
    [501, 501, 502],
    [504, 503, 505],
    [506, 503, 507],
    [503, 501, 503],
    [501, 503, 503],
    [499, 503, 499],
    [498, 501, 500],
    [501, 503, 505],
    [502, 501, 500],
    [501, 500, 501],
    [501, 501, 501],
    [502, 500, 501],
    [502, 503, 504],
    [501, 503, 503],
    [503, 503, 503],
    [498, 503, 500],
    [498, 500, 501],
    [503, 503, 504],
    [500, 500, 502],
    [500, 501, 501],
    [501, 501, 499],
    [502, 501, 505],
    [500, 501, 502],
    [501, 500, 499],
    [498, 503, 501],
    [501, 501, 500],
    [500, 503, 500],
    [503, 503, 504],
    [503, 500, 502],
    [501, 503, 502]
])


a = 5
b = 6
n = 3
N = a * b * n
alpha = 0.1

data = data_array.reshape(b, a, n)

grand_mean = np.mean(data_array)

SS_T = np.sum((data_array - grand_mean) ** 2)

supplier_means = np.mean(data, axis=(0, 2))
SS_A = b * n * np.sum((supplier_means - grand_mean) ** 2)

machine_means = np.mean(data, axis=(1, 2))
SS_B = a * n * np.sum((machine_means - grand_mean) ** 2)

cell_means = np.mean(data, axis=2)

SS_AB = 0
for i in range(b):
    for j in range(a):
        interaction_effect = (cell_means[i, j] - supplier_means[j] - machine_means[i] + grand_mean)
        SS_AB += n * (interaction_effect ** 2)

SS_E = 0
for i in range(b):
    for j in range(a):
        cell_mean = cell_means[i, j]
        for k in range(n):
            SS_E += (data[i, j, k] - cell_mean) ** 2


df_A = a - 1
df_B = b - 1
df_AB = (a - 1) * (b - 1)
df_E = a * b * (n - 1)
df_T = N - 1

F_A = (SS_A / df_A) / (SS_E / df_E)
F_B = (SS_B / df_B) / (SS_E / df_E)
F_AB = (SS_AB / df_AB) / (SS_E / df_E)

p_value_A = 1 - stats.f.cdf(F_A, df_A, df_E)
p_value_B = 1 - stats.f.cdf(F_B, df_B, df_E)
p_value_AB = 1 - stats.f.cdf(F_AB, df_AB, df_E)

print(f"Средний вес не зависит от поставщика:")
if p_value_A < alpha:
    print(f"p-value={p_value_A} < {alpha}  отвергаем Н0")
else:
    print(f"p-value={p_value_A} >= {alpha} не отвергаем Н0")

print(f"Средний вес не изменяется от машины к машине:")
if p_value_B < alpha:
    print(f"p-value={p_value_B} < {alpha} отвергаем Н0")
else:
    print(f"p-value={p_value_B} >= {alpha} не отвергаем Н0")

print(f"Отсутствует взаимодействие между поставщиком и машиной:")
if p_value_AB < alpha:
    print(f"p-value={p_value_AB} < {alpha} отвергаем Н0")
else:
    print(f"p-value={p_value_AB} >= {alpha} не отвергаем Н0")

