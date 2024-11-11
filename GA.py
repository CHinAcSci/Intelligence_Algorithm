import random
import math
import time
import matplotlib.pyplot as plt

class GeneticTSP:
    def __init__(self, num_cities, pop_size, generations, mutation_rate=0.01, tournament_size=3):
        """
        初始化遗传算法TSP求解器

        参数:
            num_cities (int): 城市数量
            pop_size (int): 种群大小
            generations (int): 迭代代数
            mutation_rate (float): 变异率
            tournament_size (int): 锦标赛选择的规模
        """
        self.num_cities = num_cities
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.cities = None
        self.best_path = None
        self.best_distance = float('inf')
        self.execution_time = None

    def generate_cities(self):
        """生成随机城市坐标"""
        self.cities = []
        for i in range(self.num_cities):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            self.cities.append((x, y))
        return self.cities

    def set_cities(self, cities):
        """设置城市坐标"""
        self.cities = cities
        self.num_cities = len(cities)

    def _distance(self, city1, city2):
        """计算两个城市之间的距离"""
        return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

    def _calculate_total_distance(self, path):
        """计算路径总长度"""
        total = 0
        for i in range(len(path)):
            city1 = self.cities[path[i]]
            city2 = self.cities[path[(i + 1) % len(path)]]
            total += self._distance(city1, city2)
        return total

    def _generate_initial_population(self):
        """生成初始种群"""
        population = []
        for _ in range(self.pop_size):
            path = list(range(self.num_cities))
            random.shuffle(path)
            population.append(path)
        return population

    def _selection(self, population):
        """选择操作"""
        tournament = random.sample(population, self.tournament_size)
        return min(tournament, key=lambda x: self._calculate_total_distance(x))

    def _crossover(self, parent1, parent2):
        """交叉操作"""
        size = len(parent1)
        if size < 2:
            return parent1

        start = random.randint(0, size-2)
        end = random.randint(start+1, size-1)

        child = [-1] * size
        # 复制父代1的一部分
        for i in range(start, end+1):
            child[i] = parent1[i]

        # 从父代2填充剩余位置
        j = 0
        for i in range(size):
            if i < start or i > end:
                while parent2[j] in child:
                    j += 1
                child[i] = parent2[j]
                j += 1

        return child

    def _mutation(self, path):
        """变异操作"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(path)), 2)
            path[i], path[j] = path[j], path[i]
        return path

    def solve(self, verbose=True):
        """
        运行遗传算法求解TSP

        参数:
            verbose (bool): 是否打印进度信息
        """
        if self.cities is None:
            self.generate_cities()

        start_time = time.time()
        population = self._generate_initial_population()

        for gen in range(self.generations):
            # 生成新一代
            new_population = []
            for _ in range(self.pop_size):
                parent1 = self._selection(population)
                parent2 = self._selection(population)
                child = self._crossover(parent1, parent2)
                child = self._mutation(child)
                new_population.append(child)

            population = new_population

            # 找出当前最优解
            current_best = min(population, key=lambda x: self._calculate_total_distance(x))
            current_distance = self._calculate_total_distance(current_best)

            if current_distance < self.best_distance:
                self.best_distance = current_distance
                self.best_path = current_best

            if verbose and gen % 100 == 0:
                print(f"Generation {gen}: Best distance = {self.best_distance:.2f}")

        self.execution_time = time.time() - start_time

        if verbose:
            print("\nFinal Results:")
            print(f"Best distance found: {self.best_distance:.2f}")
            print(f"Time taken: {self.execution_time:.2f} seconds")

        return self.best_path, self.best_distance

    def plot_route(self):
        """绘制最优路径"""
        if self.best_path is None:
            raise ValueError("No solution found yet. Run solve() first.")

        plt.figure(figsize=(10, 8))
        # 绘制城市点
        x = [city[0] for city in self.cities]
        y = [city[1] for city in self.cities]
        plt.scatter(x, y, c='red', marker='o')

        # 绘制路径
        for i in range(len(self.best_path)):
            city1 = self.cities[self.best_path[i]]
            city2 = self.cities[self.best_path[(i + 1) % len(self.best_path)]]
            plt.plot([city1[0], city2[0]], [city1[1], city2[1]], 'b-')

        plt.title('TSP Solution using Genetic Algorithm')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)
        plt.show()