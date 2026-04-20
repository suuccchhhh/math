import numpy as np
import matplotlib.pyplot as plt
import time

#ИСХОДНЫЕ ДАННЫЕ
def f(x):
    return x ** 2
def F(x):
    return np.sqrt(x)

# Аналитические значения интегралов
ANALYTICAL_LEBESGUE = (4 ** 3 - 1 ** 3) / 3
ANALYTICAL_STIELTJES = (1 / 5) * (4 ** (5 / 2) - 1)

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# Строит простую функцию аппроксимирующую снизу
def create_simple_function(n, x_values):
    x_k = np.array([1 + 3 * k / n for k in range(n + 1)])
    c_k = np.array([(1 + 3 * (k - 1) / n) ** 2 for k in range(1, n + 1)])
    f_n_values = np.zeros_like(x_values)

    for i in range(len(x_values)):
        x = x_values[i]
        for k in range(n):
            if x_k[k] <= x < x_k[k + 1] or (k == n - 1 and x == x_k[k + 1]):
                f_n_values[i] = c_k[k]
                break
    return f_n_values, x_k, c_k

def lebesgue_integral_simple(n):
    x_k = np.array([1 + 3 * k / n for k in range(n + 1)])
    c_k = np.array([(1 + 3 * (k - 1) / n) ** 2 for k in range(1, n + 1)])
    measures = np.array([x_k[k + 1] - x_k[k] for k in range(n)])
    return np.sum(c_k * measures)

def lebesgue_stieltjes_integral_simple(n):
    x_k = np.array([1 + 3 * k / n for k in range(n + 1)])
    c_k = np.array([(1 + 3 * (k - 1) / n) ** 2 for k in range(1, n + 1)])
    measures_F = np.array([F(x_k[k + 1]) - F(x_k[k]) for k in range(n)])
    return np.sum(c_k * measures_F)

#2.1 ГРАФИКИ f_n
print("2.1 Построение графиков простых функций f_n")
t_start = time.time()

x_fine = np.linspace(1, 4, 1000)
n_plot = [5, 10, 20, 50]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i in range(len(n_plot)):
    n = n_plot[i]
    f_n_vals, x_k, c_k = create_simple_function(n, x_fine)
    ax = axes[i]
    ax.plot(x_fine, f(x_fine), 'r-', linewidth=2.5, label='f(x) = x^2')
    ax.step(np.concatenate([[x_k[0]], x_k[1:]]),
            np.concatenate([[c_k[0]], c_k]),
            'b-', where='post', linewidth=1.8, label=f'f_{n}(x)')
    ax.fill_between(x_fine, f_n_vals, alpha=0.25, color='blue')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title(f'Аппроксимация f(x) = x^2 через f_{n}(x)', fontsize=12, pad=10)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.35, linestyle='--')
    if i < len(n_plot) - 1:
        ax.set_xlabel('')

plt.tight_layout(pad=2.0)
plt.show()
print(f"Графики 2.1 построены за {time.time() - t_start:.4f} с\n")

#2.2 ИНТЕГРАЛ ЛЕБЕГА
print("2.2 Вычисление интеграла Лебега")

print(f"\nАналитическое значение: {ANALYTICAL_LEBESGUE:.10f}")
n_test = [10, 100, 1000, 5000, 10000]

print("\n{:<12} {:<20} {:<20} {:<15}".format("n", "Численный интеграл", "Абсолютная погрешность",
                                             "Относит. погрешность, %"))
print("-" * 70)

res_leb = []
for n in n_test:
    val = lebesgue_integral_simple(n)
    abs_dev = abs(val - ANALYTICAL_LEBESGUE)
    rel_dev = (abs_dev / ANALYTICAL_LEBESGUE) * 100
    res_leb.append({'n': n, 'val': val, 'abs': abs_dev, 'rel': rel_dev})
    print("{:<12} {:<20.10f} {:<20.10f} {:<15.6f}".format(n, val, abs_dev, rel_dev))

t_start = time.time()
fig, ax = plt.subplots(figsize=(10, 6))
n_arr = np.array([r['n'] for r in res_leb])
dev_arr = np.array([r['abs'] for r in res_leb])
ax.loglog(n_arr, dev_arr, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('n (число разбиений)')
ax.set_ylabel('Абсолютная погрешность')
ax.set_title('Сходимость интеграла Лебега к аналитическому значению')
ax.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()
plt.show()
print(f"График 2.2 построен за {time.time() - t_start:.4f} с\n")

#  2.3 ИНТЕГРАЛ ЛЕБЕГА-СТИЛЬТЬЕСА
print("2.3 Вычисление интеграла Лебега-Стилтьеса")
print(f"\nАналитическое значение: {ANALYTICAL_STIELTJES:.10f}")
print("\n{:<12} {:<20} {:<20} {:<15}".format("n", "Численный интеграл", "Абсолютная погрешность",
                                             "Относит. погрешность, %"))
print("-" * 70)

res_st = []
for n in n_test:
    val = lebesgue_stieltjes_integral_simple(n)
    abs_dev = abs(val - ANALYTICAL_STIELTJES)
    rel_dev = (abs_dev / ANALYTICAL_STIELTJES) * 100
    res_st.append({'n': n, 'val': val, 'abs': abs_dev, 'rel': rel_dev})
    print("{:<12} {:<20.10f} {:<20.10f} {:<15.6f}".format(n, val, abs_dev, rel_dev))

t_start = time.time()
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(n_arr, np.array([r['abs'] for r in res_st]), 'ro-', linewidth=2, markersize=8)
ax.set_xlabel('n (число разбиений)')
ax.set_ylabel('Абсолютная погрешность')
ax.set_title('Сходимость интеграла Лебега-Стилтьеса к аналитическому значению')
ax.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()
plt.show()
print(f"График 2.3 построен за {time.time() - t_start:.4f} с\n")

#СРАВНЕНИЕ
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")

print(f"\nИнтеграл Лебега: аналитически = {ANALYTICAL_LEBESGUE:.10f}, при n=1000 = {res_leb[2]['val']:.10f}")
print(f"Интеграл Лебега-Стилтьеса: аналитически = {ANALYTICAL_STIELTJES:.10f}, при n=1000 = {res_st[2]['val']:.10f}")

t_start = time.time()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.semilogy(n_arr, [r['abs'] for r in res_leb], 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('n')
ax1.set_ylabel('Абсолютная погрешность')
ax1.set_title('Интеграл Лебега')
ax1.grid(True, which="both", ls="-", alpha=0.3)

ax2.semilogy(n_arr, [r['abs'] for r in res_st], 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('n')
ax2.set_ylabel('Абсолютная погрешность')
ax2.set_title('Интеграл Лебега-Стилтьеса')
ax2.grid(True, which="both", ls="-", alpha=0.3)

plt.tight_layout()
plt.show()
print(f"Графики сравнения построены за {time.time() - t_start:.4f} с\n")