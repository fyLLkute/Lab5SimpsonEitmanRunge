import matplotlib
matplotlib.use('TkAgg')  # окремі вікна в PyCharm (якщо не працює — спробуй 'Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

# ═══════════════════════════════════════════════════════════════
# Лабораторна робота №5
# Складова квадратурна формула Сімпсона.
# Методи підвищення точності. Адаптивний алгоритм.
# ═══════════════════════════════════════════════════════════════

def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a = 0
b = 24

# ─────────────────────────────────────────────────────────────
# 1. Дані для графіка функції
# ─────────────────────────────────────────────────────────────
x_vals = np.linspace(a, b, 1000)
y_vals = f(x_vals)

# ─────────────────────────────────────────────────────────────
# 2. Точне значення інтегралу
# ─────────────────────────────────────────────────────────────
I0, _ = integrate.quad(f, a, b)
print(f"\n{'='*55}")
print(f"Точне значення інтегралу I0 = {I0:.10f}")
print(f"{'='*55}")

# ─────────────────────────────────────────────────────────────
# 3. Складова квадратурна формула Сімпсона
# ─────────────────────────────────────────────────────────────
def simpson(func, a, b, N):
    if N % 2 != 0:
        N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = func(x)
    result = y[0] + y[-1]
    result += 4 * np.sum(y[1:-1:2])
    result += 2 * np.sum(y[2:-2:2])
    result *= h / 3
    return result

# ─────────────────────────────────────────────────────────────
# 4. Залежність точності від N
# ─────────────────────────────────────────────────────────────
N_values = np.arange(10, 1002, 2)
errors   = np.array([abs(simpson(f, a, b, N) - I0) for N in N_values])

eps_target = 1e-12
idx_opt = np.where(errors < eps_target)[0]
if len(idx_opt) > 0:
    N_opt   = N_values[idx_opt[0]]
    eps_opt = errors[idx_opt[0]]
else:
    idx_opt_min = np.argmin(errors)
    N_opt   = N_values[idx_opt_min]
    eps_opt = errors[idx_opt_min]
    print(f"\nУвага: точність {eps_target} не досягнута за N<=1000.")

print(f"\nПункт 4:")
print(f"  N_opt   = {N_opt}")
print(f"  epsopt  = {eps_opt:.2e}")

# ─────────────────────────────────────────────────────────────
# 5. Похибка при N0 ~ N_opt/10, кратному 8
# ─────────────────────────────────────────────────────────────
N0   = 32
I_N0 = simpson(f, a, b, N0)
eps0 = abs(I_N0 - I0)

print(f"\nПункт 5:")
print(f"  N0    = {N0}  (кратне 8, ~ N_opt/10)")
print(f"  I(N0) = {I_N0:.10f}")
print(f"  eps0  = {eps0:.2e}")

# ─────────────────────────────────────────────────────────────
# 6. Метод Рунге-Ромберга
# ─────────────────────────────────────────────────────────────
N0_half   = N0 // 2
I_N0_half = simpson(f, a, b, N0_half)
I_R       = I_N0 + (I_N0 - I_N0_half) / 15   # p=4 -> 2^4-1=15
eps_R     = abs(I_R - I0) # похибка

print(f"\nПункт 6 - Метод Рунге-Ромберга:")
print(f"  I(N0)   = {I_N0:.10f}")
print(f"  I(N0/2) = {I_N0_half:.10f}")
print(f"  I_R     = {I_R:.10f}")
print(f"  epsR    = {eps_R:.2e}")

# ─────────────────────────────────────────────────────────────
# 7. Уточнити значення інтегралу при N0 використовуючи метод Ейткена
# ─────────────────────────────────────────────────────────────

N_aitken1 = N0
N_aitken2 = N0 // 2
N_aitken3 = N0 // 4

I_aitken1 = simpson(f, a, b, N_aitken1) # I(N0)
I_aitken2 = simpson(f, a, b, N_aitken2) # I(N0/2)
I_aitken3 = simpson(f, a, b, N_aitken3) # I(N0/4)

# Обчислення уточненого значення інтегралу за формулою Ейткена
# I_E = (I(N0/2)^2 - I(N0)*I(N0/4)) / (2*I(N0/2) - (I(N0) + I(N0/4)))
numerator = (I_aitken2**2) - (I_aitken1 * I_aitken3)
denominator = 2 * I_aitken2 - (I_aitken1 + I_aitken3)

if abs(denominator) > 1e-18:
    I_E = numerator / denominator
else:
    I_E = I_aitken1  # Запобігання діленню на нуль

# Обчислення порядку точності p = (1/ln(q)) * ln| (I3-I2)/(I2-I1) |
# У нашому випадку крок змінюється в q=2 рази
diff_top = I_aitken3 - I_aitken2
diff_bottom = I_aitken2 - I_aitken1

if abs(diff_bottom) > 1e-18 and (diff_top / diff_bottom) > 0:
    p_aitken = np.log(abs(diff_top / diff_bottom)) / np.log(2)
else:
    p_aitken = 0.0

epsE = abs(I_E - I0) # похибка

print(f"\nПункт 7 - Метод Ейткена :")
print(f"  I(N0)   = {I_aitken1:.10f}  (N={N_aitken1})")
print(f"  I(N0/2) = {I_aitken2:.10f}  (N={N_aitken2})")
print(f"  I(N0/4) = {I_aitken3:.10f}  (N={N_aitken3})")
print(f"  Порядок точності p = {p_aitken:.4f}")
print(f"  I_E     = {I_E:.10f}")
print(f"  epsE    = {epsE:.2e}")
eps_A = epsE
# ─────────────────────────────────────────────────────────────
# 8. Порівняння методів
# ─────────────────────────────────────────────────────────────
print(f"\nПункт 8 - Порівняння методів:")
print(f"  {'Метод':<30} {'Похибка':>12}")
print(f"  {'-'*43}")
print(f"  {'Симпсон (N=N0)':<30} {eps0:>12.2e}")
print(f"  {'Рунге-Ромберг':<30} {eps_R:>12.2e}")
print(f"  {'Ейткен':<30} {eps_A:>12.2e}")

# ─────────────────────────────────────────────────────────────
# 9. Адаптивний алгоритм
# ─────────────────────────────────────────────────────────────
def adaptive_simpson(func, a, b, tol, depth=0, max_depth=50):
    mid = (a + b) / 2
    fa, fm, fb = func(a), func(mid), func(b)
    S_full = (b - a) / 6 * (fa + 4 * fm + fb)

    ml = (a + mid) / 2
    mr = (mid + b) / 2
    S_left  = (mid - a) / 6 * (func(a)   + 4 * func(ml) + func(mid))
    S_right = (b - mid) / 6 * (func(mid) + 4 * func(mr) + func(b))
    S_two   = S_left + S_right

    if depth >= max_depth or abs(S_two - S_full) < 15 * tol:
        return S_two + (S_two - S_full) / 15, 7
    Il, nl = adaptive_simpson(func, a,   mid, tol / 2, depth + 1, max_depth)
    Ir, nr = adaptive_simpson(func, mid, b,   tol / 2, depth + 1, max_depth)
    return Il + Ir, nl + nr + 7

tol_values   = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]
adapt_errors = []
adapt_ncalls = []

print(f"\nПункт 9 - Адаптивний алгоритм:")
print(f"  {'tol':<12} {'I_adapt':>15} {'Похибка':>12} {'N обчисл.':>12}")
print(f"  {'-'*52}")
for tol in tol_values:
    I_ad, nc = adaptive_simpson(f, a, b, tol)
    err = abs(I_ad - I0)
    adapt_errors.append(err)
    adapt_ncalls.append(nc)
    print(f"  {tol:<12.0e} {I_ad:>15.8f} {err:>12.2e} {nc:>12}")

print(f"\n{'='*55}")
print("ПІДСУМОК:")
print(f"  Точний інтеграл I0      = {I0:.10f}")
print(f"  N_opt                   = {N_opt}")
print(f"  Похибка Сімпсон (N0={N0}) = {eps0:.2e}")
print(f"  Похибка Рунге-Ромберг   = {eps_R:.2e}")
print(f"  Похибка Ейткен          = {eps_A:.2e}")
print(f"  Порядок методу (Ейткен) = {p_aitken:.3f}")
print(f"{'='*55}")

# ═══════════════════════════════════════════════════════════════
# ГРАФІКИ  (кожен у окремому вікні)
# ═══════════════════════════════════════════════════════════════
plt.style.use('seaborn-v0_8-whitegrid')

# ── Графік 1: функція навантаження ───────────────────────────
fig1, ax1 = plt.subplots(figsize=(12, 5))
fig1.suptitle('Графік 1 — Функція навантаження на сервер',
              fontsize=13, fontweight='bold')

ax1.plot(x_vals, y_vals, color='steelblue', lw=2.2,
         label=r'$f(x) = 50 + 20\sin\!\left(\frac{\pi x}{12}\right) + 5e^{-0.2(x-12)^2}$')
ax1.fill_between(x_vals, y_vals, alpha=0.12, color='steelblue')
ax1.set_xlabel('Час, x (год)', fontsize=12)
ax1.set_ylabel('Навантаження, f(x)', fontsize=12)
ax1.set_title(r'$f(x) = 50 + 20\sin\!\left(\frac{\pi x}{12}\right) + 5e^{-0.2(x-12)^2}$,'
              r'$\quad x \in [0,\,24]$', fontsize=11)
ax1.legend(fontsize=11)
ax1.set_xlim(a, b)
fig1.tight_layout()

# ── Графік 2: похибка від числа розбиттів N ──────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))
fig2.suptitle('Графік 2 — Залежність похибки від числа розбиттів N',
              fontsize=13, fontweight='bold')

ax2.semilogy(N_values, errors, color='darkorange', lw=2.0,
             label='Похибка складової формули Сімпсона')
ax2.axvline(N_opt, color='red', lw=1.5, linestyle='--',
            label=f'N_opt = {N_opt}')
ax2.axhline(eps_target, color='seagreen', lw=1.5, linestyle=':',
            label=f'eps = {eps_target:.0e}')
ax2.scatter([N_opt], [eps_opt], color='red', zorder=6, s=80)
ax2.set_xlabel('N', fontsize=12)
ax2.set_ylabel('|I(N) - I0|  (log)', fontsize=12)
ax2.set_title('Складова формула Сімпсона: апроксимаційна похибка', fontsize=11)
ax2.legend(fontsize=10)
fig2.tight_layout()

# ── Графік 3: порівняння методів ─────────────────────────────
fig3, ax3 = plt.subplots(figsize=(9, 5))
fig3.suptitle('Графік 3 — Порівняння похибок методів підвищення точності',
              fontsize=13, fontweight='bold')

methods_labels = [f'Сімпсон\n(N={N0})',
                  f'Сімпсон\n(N={N0 // 2})',
                  'Рунге–\nРомберг',
                  'Ейткен']
err_comp   = [eps0, abs(I_N0_half - I0), eps_R, eps_A]
bar_colors = ['#4e79a7', '#59a14f', '#e15759', '#f28e2b']

bars = ax3.bar(methods_labels, err_comp, color=bar_colors,
               edgecolor='white', linewidth=1.2, zorder=3)
ax3.set_yscale('log')
ax3.set_ylabel('|Похибка|  (log)', fontsize=11)
ax3.set_title('Точність: Сімпсон / Рунге-Ромберг / Ейткен', fontsize=11)
for bar, val in zip(bars, err_comp):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             val * 1.6, f'{val:.2e}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.grid(axis='y', zorder=0)
fig3.tight_layout()

# ── Графік 4: адаптивний алгоритм ────────────────────────────
fig4, ax4 = plt.subplots(figsize=(9, 5))
fig4.suptitle('Графік 4 — Адаптивний алгоритм Сімпсона',
              fontsize=13, fontweight='bold')

ax4.loglog(adapt_ncalls, adapt_errors, 'o-', color='mediumpurple',
           lw=2.2, markersize=8, label='Похибка адаптивного методу')
ax4.set_xlabel('Кількість обчислень f(x)  (log)', fontsize=12)
ax4.set_ylabel('|Похибка|  (log)', fontsize=12)
ax4.set_title('Залежність точності від кількості обчислень f(x)', fontsize=11)
for nc, ea in zip(adapt_ncalls, adapt_errors):
    ax4.annotate(f'{ea:.0e}', (nc, ea),
                 textcoords='offset points', xytext=(7, 5), fontsize=9)
ax4.legend(fontsize=10)
fig4.tight_layout()

plt.show()   # відкриває всі чотири вікна одночасно