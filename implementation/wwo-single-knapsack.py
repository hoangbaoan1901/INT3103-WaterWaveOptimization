import random
import time
from typing import List, Tuple


class SingleKnapsackInstance:
    """Class to represent a Single Knapsack Problem instance"""
    
    def __init__(self):
        self.n_objects = 0
        self.capacity = 0
        self.weights = []
        self.values = []
        self.load_from_stdin()
    
    def load_from_stdin(self):
        """Load knapsack instance from standard input"""
        # Read first line: n objects and capacity W
        n, W = map(int, input().split())
        self.n_objects = n
        self.capacity = W
        
        # Read weights and values
        self.weights = []
        self.values = []
        
        for _ in range(n):
            w, v = map(int, input().split())
            self.weights.append(w)
            self.values.append(v)


class KnapsackSolution:
    """Class to represent a solution for WWO Single Knapsack"""
    
    def __init__(self, instance: SingleKnapsackInstance, solution: List[int] = None):
        self.instance = instance
        if solution is None:
            self.solution = self.generate_random_solution()
        else:
            self.solution = solution[:]
        
        self.fitness = self.calculate_fitness()
        self.wavelength = 0.5  # Initial wavelength
    
    def generate_random_solution(self) -> List[int]:
        """Generate a random feasible solution using greedy approach with value/weight ratio"""
        solution = [0] * self.instance.n_objects
        
        # Calculate value-to-weight ratio
        ratios = []
        for i in range(self.instance.n_objects):
            if self.instance.weights[i] > 0:
                ratio = self.instance.values[i] / self.instance.weights[i]
                ratios.append((i, ratio))
        
        # Sort by ratio in descending order
        ratios.sort(key=lambda x: x[1], reverse=True)
        
        # Add some randomness by shuffling the top third
        if len(ratios) > 6:
            top_third = len(ratios) // 3
            top_part = ratios[:top_third]
            random.shuffle(top_part)
            ratios[:top_third] = top_part
        
        # Greedily add items
        current_weight = 0
        for item_idx, _ in ratios:
            if current_weight + self.instance.weights[item_idx] <= self.instance.capacity:
                solution[item_idx] = 1
                current_weight += self.instance.weights[item_idx]
        
        return solution
    
    def calculate_fitness(self) -> float:
        """Calculate fitness with penalty for exceeding capacity"""
        total_value = sum(self.solution[i] * self.instance.values[i] 
                         for i in range(self.instance.n_objects))
        total_weight = sum(self.solution[i] * self.instance.weights[i] 
                          for i in range(self.instance.n_objects))
        
        # Apply penalty if capacity is exceeded
        if total_weight > self.instance.capacity:
            penalty = (total_weight - self.instance.capacity) * 1000
            return total_value - penalty
        
        return total_value
    
    def is_feasible(self) -> bool:
        """Check if solution is feasible"""
        total_weight = sum(self.solution[i] * self.instance.weights[i] 
                          for i in range(self.instance.n_objects))
        return total_weight <= self.instance.capacity
    
    def get_weight(self) -> int:
        """Get total weight of current solution"""
        return sum(self.solution[i] * self.instance.weights[i] 
                  for i in range(self.instance.n_objects))
    
    def repair_solution(self):
        """Repair infeasible solution by removing items with lowest value/weight ratio"""
        while not self.is_feasible():
            # Find selected items and their value/weight ratios
            selected_items = []
            for i in range(self.instance.n_objects):
                if self.solution[i] == 1:
                    if self.instance.weights[i] > 0:
                        ratio = self.instance.values[i] / self.instance.weights[i]
                        selected_items.append((i, ratio))
            
            if not selected_items:
                break
            
            # Remove item with lowest value/weight ratio
            selected_items.sort(key=lambda x: x[1])
            worst_item = selected_items[0][0]
            self.solution[worst_item] = 0
        
        # Recalculate fitness after repair
        self.fitness = self.calculate_fitness()
    
    def copy(self):
        """Create a copy of the solution"""
        new_solution = KnapsackSolution(self.instance, self.solution)
        new_solution.wavelength = self.wavelength
        return new_solution
    
    def propagate(self) -> 'KnapsackSolution':
        """Propagate solution using bit-flip local search"""
        new_solution = self.copy()
        
        # Determine number of changes based on wavelength
        max_changes = max(1, int(self.wavelength))
        k = random.randint(1, max_changes)
        
        for _ in range(k):
            # Random bit flip
            bit_to_flip = random.randint(0, self.instance.n_objects - 1)
            new_solution.solution[bit_to_flip] = 1 - new_solution.solution[bit_to_flip]
        
        # Repair if infeasible
        new_solution.fitness = new_solution.calculate_fitness()
        if not new_solution.is_feasible():
            new_solution.repair_solution()
        
        return new_solution
    
    def local_search_breaking(self, n_b: int) -> List['KnapsackSolution']:
        """Generate neighbors using breaking operator"""
        neighbors = []
        
        for k in range(1, min(n_b + 1, self.instance.n_objects + 1)):
            neighbor = self.copy()
            
            # Strategy based on k value
            strategy = k % 3
            
            if strategy == 0:
                # Remove k-th lowest value item and try to add highest value items
                selected_items = [(i, self.instance.values[i]) for i in range(self.instance.n_objects) 
                                if neighbor.solution[i] == 1]
                if selected_items and k <= len(selected_items):
                    selected_items.sort(key=lambda x: x[1])
                    item_to_remove = selected_items[k-1][0]
                    neighbor.solution[item_to_remove] = 0
                    
                    # Try to add highest value items that fit
                    unselected_items = [(i, self.instance.values[i]) for i in range(self.instance.n_objects) 
                                      if neighbor.solution[i] == 0]
                    unselected_items.sort(key=lambda x: x[1], reverse=True)
                    
                    for item, _ in unselected_items:
                        temp_solution = neighbor.solution[:]
                        temp_solution[item] = 1
                        temp_weight = sum(temp_solution[i] * self.instance.weights[i] 
                                        for i in range(self.instance.n_objects))
                        if temp_weight <= self.instance.capacity:
                            neighbor.solution[item] = 1
                            break
            
            elif strategy == 1:
                # Add highest value items that fit
                unselected_items = [i for i in range(self.instance.n_objects) if neighbor.solution[i] == 0]
                if unselected_items:
                    value_items = [(i, self.instance.values[i]) for i in unselected_items]
                    value_items.sort(key=lambda x: x[1], reverse=True)
                    
                    for item, _ in value_items[:k]:
                        temp_solution = neighbor.solution[:]
                        temp_solution[item] = 1
                        temp_weight = sum(temp_solution[i] * self.instance.weights[i] 
                                        for i in range(self.instance.n_objects))
                        if temp_weight <= self.instance.capacity:
                            neighbor.solution[item] = 1
                            break
            
            else:
                # Swap items based on value/weight improvement
                selected = [i for i in range(self.instance.n_objects) if neighbor.solution[i] == 1]
                unselected = [i for i in range(self.instance.n_objects) if neighbor.solution[i] == 0]
                
                if selected and unselected:
                    for _ in range(min(k, len(selected), len(unselected))):
                        # Find best swap
                        best_improvement = 0
                        best_swap = None
                        
                        for sel_item in selected[:min(5, len(selected))]:  # Limit search
                            for unsel_item in unselected[:min(5, len(unselected))]:
                                temp_solution = neighbor.solution[:]
                                temp_solution[sel_item] = 0
                                temp_solution[unsel_item] = 1
                                
                                temp_weight = sum(temp_solution[i] * self.instance.weights[i] 
                                                for i in range(self.instance.n_objects))
                                if temp_weight <= self.instance.capacity:
                                    improvement = (self.instance.values[unsel_item] - 
                                                 self.instance.values[sel_item])
                                    if improvement > best_improvement:
                                        best_improvement = improvement
                                        best_swap = (sel_item, unsel_item)
                        
                        if best_swap:
                            neighbor.solution[best_swap[0]] = 0
                            neighbor.solution[best_swap[1]] = 1
                            break
            
            neighbor.fitness = neighbor.calculate_fitness()
            if not neighbor.is_feasible():
                neighbor.repair_solution()
            
            neighbors.append(neighbor)
        
        return neighbors
    
    def get_selected_items(self) -> List[int]:
        """Get list of selected item indices (1-based)"""
        return [i + 1 for i in range(self.instance.n_objects) if self.solution[i] == 1]


class WWOSingleKnapsack:
    """Water Wave Optimization Algorithm for Single Knapsack Problem"""
    
    def __init__(self, instance: SingleKnapsackInstance, 
                 np_max: int = 50, np_min: int = 12, 
                 lambda_max: int = None, lambda_min: int = 1,
                 n_b: int = 5, max_generations: int = 1000):
        
        self.instance = instance
        self.np_max = np_max
        self.np_min = np_min
        self.lambda_max = lambda_max if lambda_max else min(15, instance.n_objects // 3)
        self.lambda_min = lambda_min
        self.n_b = n_b
        self.max_generations = max_generations
        
        self.population = []
        self.best_solution = None
        self.generation = 0
        self.current_np = np_max
    
    def initialize_population(self):
        """Initialize population with random solutions"""
        self.population = []
        for _ in range(self.np_max):
            solution = KnapsackSolution(self.instance)
            self.population.append(solution)
        
        # Find initial best solution
        self.best_solution = max(self.population, key=lambda x: x.fitness).copy()
    
    def calculate_wavelengths(self):
        """Calculate wavelengths using fitness-based strategy"""
        if not self.population:
            return
        
        # Calculate sum of all fitness values
        total_fitness = sum(max(0, sol.fitness) for sol in self.population)
        
        for solution in self.population:
            if total_fitness > 0:
                normalized_fitness = max(0, solution.fitness)
                ratio = (total_fitness - normalized_fitness) / total_fitness
                solution.wavelength = self.lambda_max * ratio
                solution.wavelength = max(self.lambda_min, min(self.lambda_max, solution.wavelength))
            else:
                solution.wavelength = self.lambda_max / 2
    
    def update_population_size(self):
        """Linearly decrease population size by removing worst solution"""
        if self.generation > 0 and len(self.population) > self.np_min:
            # Remove worst solution
            worst_idx = min(range(len(self.population)), key=lambda i: self.population[i].fitness)
            self.population.pop(worst_idx)
            self.current_np = len(self.population)
    
    def run(self) -> KnapsackSolution:
        """Run the WWO algorithm"""
        # Initialize population
        self.initialize_population()
        
        # Main optimization loop
        for generation in range(self.max_generations):
            self.generation = generation
            
            # Calculate wavelengths
            self.calculate_wavelengths()
            
            # Propagate each solution
            new_population = []
            for solution in self.population:
                # Propagate solution
                new_solution = solution.propagate()
                
                # Replace only if better (S7)
                if new_solution.fitness > solution.fitness:
                    new_population.append(new_solution)
                    
                    # Check if new best solution found
                    if new_solution.fitness > self.best_solution.fitness:
                        self.best_solution = new_solution.copy()
                        
                        # Apply breaking operator (S6)
                        neighbors = new_solution.local_search_breaking(self.n_b)
                        if neighbors:
                            best_neighbor = max(neighbors, key=lambda x: x.fitness)
                            if best_neighbor.fitness > self.best_solution.fitness:
                                self.best_solution = best_neighbor.copy()
                else:
                    new_population.append(solution)
            
            self.population = new_population
            
            # Update population size (S10)
            self.update_population_size()
            
            # Early stopping if population becomes too small or no improvement
            if len(self.population) < 2:
                break
        
        return self.best_solution


def solve_single_knapsack():
    """Main function to solve single knapsack problem"""
    try:
        # Load instance from stdin
        instance = SingleKnapsackInstance()
        
        # Adjust algorithm parameters based on problem size
        if instance.n_objects <= 20:
            np_max, max_gen = 30, 300
        elif instance.n_objects <= 50:
            np_max, max_gen = 40, 500
        elif instance.n_objects <= 100:
            np_max, max_gen = 50, 800
        else:
            np_max, max_gen = 60, 1000
        
        # Run WWO algorithm
        wwo = WWOSingleKnapsack(
            instance, 
            np_max=np_max, 
            np_min=max(8, np_max // 4),
            max_generations=max_gen
        )
        
        best_solution = wwo.run()
        
        # Get selected items (1-based indexing)
        selected_items = best_solution.get_selected_items()
        
        # Output results
        print(len(selected_items))
        if selected_items:
            print(' '.join(map(str, selected_items)))
        else:
            print()
            
    except Exception as e:
        # Fallback: output empty solution
        print(0)
        print()


if __name__ == "__main__":
    solve_single_knapsack()
