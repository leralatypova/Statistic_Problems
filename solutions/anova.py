import numpy as np
import scipy.stats as stats

data = np.array([
    [164, 172, 163, 150, 164],
    [177, 197, 177, 172, 169],
    [168, 167, 144, 146, 145],
    [146, 161, 165, 141, 149],
    [172, 180, 166, 169, 170],
    [196, 190, 178, 183, 167]
])

a = data.shape[0]
b = data.shape[1]
N = a * b
alpha = 0.1

total_sum = np.sum(data)

general_mean = np.mean(data)

row_means = np.mean(data, axis=0)

col_means = np.mean(data, axis=1)


SS_T = 0
for i in range(a):
    for j in range(b):
        SS_T += (data[i, j] - general_mean) ** 2

SS_A = np.sum(b * (row_means - general_mean) ** 2)

SS_B = np.sum(a * (col_means - general_mean) ** 2)

SS_E = SS_T - SS_A - SS_B

df_A = a - 1
df_B = b - 1
df_E = (a - 1) * (b - 1)

F_A = (SS_A / df_A) / (SS_E / df_E)
F_B = (SS_B / df_B) / (SS_E / df_E)

p_value_A = 1 - stats.f.cdf(F_A, df_A, df_E)
p_value_B = 1 - stats.f.cdf(F_B, df_B, df_E)

print(f"Гипотеза об отсутствии различий между сортами:")
if p_value_B < alpha:
    print(f"p-value={p_value_B} < {alpha} тогда отвергаем Н0")
else:
    print(f"p-value={p_value_B} >= {alpha} тогда не отвергаем Н0")

print(f"Гипотеза об отсутствии различий между днями:")
if p_value_A < alpha:
    print(f"p-value={p_value_A} < {alpha} тогда отвергаем Н0")
else:
    print(f"p-value={p_value_A} >= {alpha} тогда не отвергаем Н0")