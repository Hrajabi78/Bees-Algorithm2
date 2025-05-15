import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

# --- خواندن داده‌ها ---
df1 = pd.read_excel("C:/Users/this pc/Desktop/Bees/houses.xls")
df2 = pd.read_excel("C:/Users/this pc/Desktop/Bees/schools.xls")

X_houses = df1['POINT_X'].to_numpy()
Y_houses = df1['POINT_Y'].to_numpy()
X_schools = df2['POINT_X'].to_numpy()
Y_schools = df2['POINT_Y'].to_numpy()

# --- تنظیمات ---
population_size = 80
num_houses = 380
num_schools = 10
max_houses_per_school = 38

n_elite_sites = 5  # تعداد سایت‌های نخبه
n_selected_sites = 30  # تعداد سایت‌های انتخاب شده برای جستجوی همسایگی
n_recruited_bees_elite = 15  # تعداد زنبورهای اعزامی به هر سایت نخبه
n_recruited_bees_selected = 5  # تعداد زنبورهای اعزامی به سایت‌های غیرنخبه
n_scout_bees = population_size - (
            n_elite_sites  + (n_selected_sites - n_elite_sites))
max_iterations = 10000

# --- تابع برای شمارش تعداد خانه‌های تخصیص‌یافته به هر مدرسه ---
def count_assignments(solution):
    counts = Counter(solution)
    for school_id in range(1, num_schools + 1):
        assigned = counts.get(school_id, 0)
        print(f"School {school_id}: {assigned} houses assigned {'✅' if assigned <= max_houses_per_school else '❌'}")
    return counts

# --- محاسبه ماتریس فاصله ---
distance_matrix = np.zeros((num_houses, num_schools))
for i in range(num_houses):
    for j in range(num_schools):
        dx = X_houses[i] - X_schools[j]
        dy = Y_houses[i] - Y_schools[j]
        distance_matrix[i][j] = dx ** 2 + dy ** 2  # فاصله بدون جذر


# --- توابع ---
def generate_individual():
    individual = [i for i in range(1, num_schools + 1) for _ in range(max_houses_per_school)]
    random.shuffle(individual)
    return individual


def fitness(individual):
    indices = np.array(individual) - 1
    house_indices = np.arange(num_houses)
    return distance_matrix[house_indices, indices].sum()



def neighborhood_search(individual):
    neighbor = individual.copy()
    i, j = random.sample(range(num_houses), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


# --- شروع الگوریتم زنبور ---
population = [generate_individual() for _ in range(population_size)]
# --- ذخیره بهترین فرد از جمعیت اولیه ---
initial_population = population.copy()
first_gen_best = min(initial_population, key=fitness)

# --- شروع الگوریتم زنبور ---
best_solution = None
best_fitness = float('inf')
fitness_progress = []

for iteration in range(max_iterations):
    population = sorted(population, key=fitness)

    new_population = []

    # --- جستجو در سایت‌های نخبه ---
    for i in range(n_elite_sites):
        site = population[i]
        best_site_solution = site
        best_site_fitness = fitness(site)
        for _ in range(n_recruited_bees_elite):
            neighbor = neighborhood_search(site)
            neighbor_fit = fitness(neighbor)
            if neighbor_fit < best_site_fitness:
                best_site_solution = neighbor
                best_site_fitness = neighbor_fit
        new_population.append(best_site_solution)

    # --- جستجو در سایت‌های انتخاب شده ---
    for i in range(n_elite_sites, n_selected_sites):
        site = population[i]
        best_site_solution = site
        best_site_fitness = fitness(site)
        for _ in range(n_recruited_bees_selected):
            neighbor = neighborhood_search(site)
            neighbor_fit = fitness(neighbor)
            if neighbor_fit < best_site_fitness:
                best_site_solution = neighbor
                best_site_fitness = neighbor_fit
        new_population.append(best_site_solution)

    # --- تولید زنبورهای اکتشافی (تصادفی) ---
    for _ in range(n_scout_bees):
        new_population.append(generate_individual())

    # --- بروزرسانی بهترین جواب ---
    current_best = min(new_population, key=fitness)
    current_best_fit = fitness(current_best)
    if current_best_fit < best_fitness:
        best_solution = current_best
        best_fitness = current_best_fit

    population = new_population
    fitness_progress.append(best_fitness)

    print(f"Iteration {iteration}: Best Fitness = {best_fitness:.2f}")

# --- نتایج نهایی ---
print("\nBest Solution Fitness:", best_fitness)


# --- تابع جدید برای رسم مقایسه ---
def plot_comparison(first, final):
    plt.figure(figsize=(14, 6))

    # --- اولی: جمعیت اولیه ---
    plt.subplot(1, 2, 1)
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray']
    for i in range(len(first)):
        sx, sy = X_schools[first[i] - 1], Y_schools[first[i] - 1]
        hx, hy = X_houses[i], Y_houses[i]
        color = colors[first[i] - 1]
        plt.plot([sx, hx], [sy, hy], color=color, linewidth=0.5)
        plt.scatter(hx, hy, color=color, s=8)
    plt.scatter(X_schools, Y_schools, color='red', marker='*', s=200, label='Schools')
    plt.title("Initial Population - Best Individual")
    plt.grid(True)

    # --- دومی: جواب نهایی ---
    plt.subplot(1, 2, 2)
    for i in range(len(final)):
        sx, sy = X_schools[final[i] - 1], Y_schools[final[i] - 1]
        hx, hy = X_houses[i], Y_houses[i]
        color = colors[final[i] - 1]
        plt.plot([sx, hx], [sy, hy], color=color, linewidth=0.5)
        plt.scatter(hx, hy, color=color, s=8)
    plt.scatter(X_schools, Y_schools, color='red', marker='*', s=200, label='Schools')
    plt.title("Best Solution - Final Generation")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# --- رسم نهایی ---
plot_comparison(first_gen_best, best_solution)

# --- شمارش و بررسی تخصیص نهایی ---
print("\n📊 Final Assignment Summary:")
assignment_counts = count_assignments(best_solution)