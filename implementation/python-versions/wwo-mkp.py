import random
import os
import time
from typing import List, Tuple, Dict, Any
import math


class MKPInstance:
    """Class to represent a Multiple Knapsack Problem instance"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.n_knapsacks = 0
        self.n_objects = 0
        self.weights = []
        self.capacities = []
        self.constraints = []
        self.known_optimum = 0
        self.load_instance(filename)
    
    def load_instance(self, filename: str):
        """Load MKP instance from file"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Remove empty lines and strip whitespace
        lines = [line.strip() for line in lines if line.strip()]
        
        # Parse first line: n_knapsacks, n_objects
        first_line = lines[0].split()
        self.n_knapsacks = int(first_line[0])
        self.n_objects = int(first_line[1])
        
        # Parse weights (called weights, but actually means values)
        weight_lines = []
        line_idx = 1
        while line_idx < len(lines) and len(weight_lines) < self.n_objects:
            weights_in_line = lines[line_idx].split()
            weight_lines.extend([int(w) for w in weights_in_line])
            line_idx += 1
        
        self.weights = weight_lines[:self.n_objects]
        
        # Parse capacities
        cap_lines = []
        while line_idx < len(lines) and len(cap_lines) < self.n_knapsacks:
            caps_in_line = lines[line_idx].split()
            cap_lines.extend([int(c) for c in caps_in_line])
            line_idx += 1
        
        self.capacities = cap_lines[:self.n_knapsacks]
        
        # Parse constraint matrix (n_knapsacks x n_objects)
        self.constraints = []
        for k in range(self.n_knapsacks):
            constraint_row = []
            while line_idx < len(lines) and len(constraint_row) < self.n_objects:
                values_in_line = lines[line_idx].split()
                constraint_row.extend([int(v) for v in values_in_line])
                line_idx += 1
            self.constraints.append(constraint_row[:self.n_objects])
        
        # Parse known optimum (last line)
        if line_idx < len(lines):
            self.known_optimum = int(lines[-1])


class WWOSolution:
    """Class to represent a solution for WWO-MKP"""
    
    def __init__(self, instance: MKPInstance, solution: List[int] = None):
        self.instance = instance
        if solution is None:
            self.solution = self.generate_random_solution()
        else:
            self.solution = solution.copy()
        
        self.fitness = self.calculate_fitness()
        self.wavelength = 0.5  # Initial wavelength
    
    def generate_random_solution(self) -> List[int]:
        """Generate a random feasible solution (with shuffled greedy aspect)"""
        solution = [0] * self.instance.n_objects
        
        # Greedy approach with profit density: add items in order of weight/constraint ratio
        profit_density = [(i, self.instance.weights[i] / sum(self.instance.constraints[k][i] for k in range(self.instance.n_knapsacks))) 
                         for i in range(self.instance.n_objects)]
        # profit_density.sort(key=lambda x: x[1], reverse=True)
        
        # Add some randomness by shuffling top candidates
        if len(profit_density) > 10:
            # Shuffle top half items by profit density
            # top_items = profit_density[:len(profit_density)//2]
            # random.shuffle(top_items)
            # profit_density[:len(profit_density)//2] = top_items
            random.shuffle(profit_density)
        
        for item, _ in profit_density:
            # Check if adding this item violates any knapsack constraint
            feasible = True
            for k in range(self.instance.n_knapsacks):
                current_weight = sum(solution[j] * self.instance.constraints[k][j] 
                                   for j in range(self.instance.n_objects))
                if current_weight + self.instance.constraints[k][item] > self.instance.capacities[k]:
                    feasible = False
                    break
            
            if feasible:
                solution[item] = 1
        
        return solution
    
    def calculate_fitness(self) -> float:
        """Calculate fitness (total profit) with penalty for infeasible solutions"""
        total_profit = sum(self.solution[i] * self.instance.weights[i] 
                          for i in range(self.instance.n_objects))
        
        # Check feasibility and apply penalty
        penalty = 0
        for k in range(self.instance.n_knapsacks):
            used_capacity = sum(self.solution[j] * self.instance.constraints[k][j] 
                              for j in range(self.instance.n_objects))
            if used_capacity > self.instance.capacities[k]:
                penalty += (used_capacity - self.instance.capacities[k]) * max(self.instance.weights) * 10
        
        return total_profit - penalty
    
    def is_feasible(self) -> bool:
        """Check if solution is feasible"""
        for k in range(self.instance.n_knapsacks):
            used_capacity = sum(self.solution[j] * self.instance.constraints[k][j] 
                              for j in range(self.instance.n_objects))
            if used_capacity > self.instance.capacities[k]:
                return False
        return True
    
    def repair_solution(self):
        """Repair infeasible solution by removing items with lowest profit density"""
        while not self.is_feasible():
            # Find items currently in the solution
            selected_items = [i for i in range(self.instance.n_objects) if self.solution[i] == 1]
            
            if not selected_items:
                break
            
            # Calculate profit density for selected items
            profit_density = []
            for item in selected_items:
                total_constraint = sum(self.instance.constraints[k][item] for k in range(self.instance.n_knapsacks))
                if total_constraint > 0:
                    density = self.instance.weights[item] / total_constraint
                else:
                    density = float('inf')
                profit_density.append((item, density))
            
            # Remove item with lowest profit density
            profit_density.sort(key=lambda x: x[1])
            if profit_density:
                item_to_remove = profit_density[0][0]
                self.solution[item_to_remove] = 0
        
        # Recalculate fitness after repair
        self.fitness = self.calculate_fitness()
    
    def copy(self):
        """Create a copy of the solution"""
        new_solution = WWOSolution(self.instance, self.solution)
        new_solution.wavelength = self.wavelength
        return new_solution
    
    def propagate(self) -> 'WWOSolution':
        """Propagate solution using S2 strategy (k-step local search, bit reversal)"""
        new_solution = self.copy()
        
        # Determine number of changes based on wavelength
        k = random.randint(1, max(1, int(self.wavelength)))
        
        for _ in range(k):
            # Randomly flip a bit
            item = random.randint(0, self.instance.n_objects - 1)
            new_solution.solution[item] = 1 - new_solution.solution[item]
        
        # Repair solution if infeasible
        new_solution.fitness = new_solution.calculate_fitness()
        if not new_solution.is_feasible():
            new_solution.repair_solution()
        
        return new_solution
    
    def local_search_breaking(self, n_b: int) -> List['WWOSolution']:
        """Generate k neighbors using breaking operator (S6), kth neighbor is alternated by removing kth least profit instance removed"""
        neighbors = []
        
        for k in range(1, min(n_b + 1, self.instance.n_objects + 1)):
            neighbor = self.copy()
            
            # Find items sorted by profit (ascending for removal)
            profit_items = [(i, self.instance.weights[i]) for i in range(self.instance.n_objects) 
                           if neighbor.solution[i] == 1]
            profit_items.sort(key=lambda x: x[1])
            
            # Remove k-th smallest profit item if exists
            if len(profit_items) >= k:
                item_to_remove = profit_items[k-1][0]
                neighbor.solution[item_to_remove] = 0
                
                # Try to add other items in decreasing order of profit
                remaining_items = [(i, self.instance.weights[i]) for i in range(self.instance.n_objects) 
                                 if neighbor.solution[i] == 0]
                remaining_items.sort(key=lambda x: x[1], reverse=True)
                
                for item, _ in remaining_items:
                    # Check if adding this item is feasible
                    feasible = True
                    for knap in range(self.instance.n_knapsacks):
                        current_weight = sum(neighbor.solution[j] * self.instance.constraints[knap][j] 
                                           for j in range(self.instance.n_objects))
                        if current_weight + self.instance.constraints[knap][item] > self.instance.capacities[knap]:
                            feasible = False
                            break
                    
                    if feasible:
                        neighbor.solution[item] = 1
                
                neighbor.fitness = neighbor.calculate_fitness()
                neighbors.append(neighbor)
        
        return neighbors


class WWOAlgorithm:
    """Water Wave Optimization Algorithm for Multiple Knapsack Problem"""
    
    def __init__(self, instance: MKPInstance, 
                 np_max: int = 50, np_min: int = 10, 
                 lambda_max: int = None, lambda_min: int = 1,
                 n_b: int = 5, max_generations: int = 1000):
        
        self.instance = instance
        self.np_max = np_max
        self.np_min = np_min
        self.lambda_max = lambda_max if lambda_max else instance.n_objects // 4
        self.lambda_min = lambda_min
        self.n_b = n_b
        self.max_generations = max_generations
        
        self.population = []
        self.best_solution = None
        self.generation = 0
        self.current_np = np_max
        
        # Statistics
        self.fitness_history = []
        self.best_fitness_history = []
    
    def initialize_population(self):
        """Initialize population with random solutions"""
        self.population = []
        for _ in range(self.np_max):
            solution = WWOSolution(self.instance)
            self.population.append(solution)
        
        # Find initial best solution
        self.best_solution = max(self.population, key=lambda x: x.fitness).copy()
    
    def calculate_wavelengths(self):
        """Calculate wavelengths using S3 strategy"""
        if not self.population:
            return
        
        # Calculate sum of all fitness values
        total_fitness = sum(sol.fitness for sol in self.population)
        
        for solution in self.population:
            # Using equation E3
            if total_fitness > 0:
                wavelength = self.lambda_max * (total_fitness - solution.fitness) / total_fitness
            else:
                wavelength = self.lambda_max * 0.5
            
            # Ensure wavelength is within bounds
            solution.wavelength = max(self.lambda_min, min(self.lambda_max, wavelength))
    
    def update_population_size(self):
        """Linearly decrease population size (S10)"""
        if self.generation > 0 and self.current_np > self.np_min:
            # Linear reduction
            reduction_rate = (self.np_max - self.np_min) / self.max_generations
            target_np = self.np_max - int(reduction_rate * self.generation)
            target_np = max(self.np_min, target_np)
            
            while len(self.population) > target_np:
                # Remove worst solution
                worst_idx = min(range(len(self.population)), 
                              key=lambda i: self.population[i].fitness)
                self.population.pop(worst_idx)
            
            self.current_np = len(self.population)
    
    def run(self) -> Tuple[WWOSolution, List[float]]:
        """Run the WWO algorithm"""
        print(f"Starting WWO for {self.instance.filename}")
        print(f"Problem size: {self.instance.n_knapsacks} knapsacks, {self.instance.n_objects} objects")
        print(f"Known optimum: {self.instance.known_optimum}")
        
        start_time = time.time()
        self.initialize_population()
        
        for self.generation in range(self.max_generations):
            # Calculate wavelengths
            self.calculate_wavelengths()
            
            # Propagate each solution
            for i, solution in enumerate(self.population):
                # Propagate solution
                new_solution = solution.propagate()
                
                # Solution update using S7 strategy
                if new_solution.fitness > solution.fitness:
                    self.population[i] = new_solution
                    
                    # Check if new best solution found
                    if new_solution.fitness > self.best_solution.fitness:
                        # Perform local search (breaking)
                        neighbors = new_solution.local_search_breaking(self.n_b)
                        
                        # Find best among current solution and neighbors
                        best_neighbor = new_solution
                        for neighbor in neighbors:
                            if neighbor.fitness > best_neighbor.fitness:
                                best_neighbor = neighbor
                        
                        # Update best solution
                        if best_neighbor.fitness > self.best_solution.fitness:
                            self.best_solution = best_neighbor.copy()
                            print(f"Generation {self.generation}: New best fitness = {self.best_solution.fitness:.2f}")
            
            # Update population size
            self.update_population_size()
            
            # Record statistics
            current_best = max(self.population, key=lambda x: x.fitness)
            self.fitness_history.append(current_best.fitness)
            self.best_fitness_history.append(self.best_solution.fitness)
            
            # Print progress
            if self.generation % 100 == 0:
                feasible_str = "Feasible" if self.best_solution.is_feasible() else "Infeasible"
                print(f"Generation {self.generation}: Best = {self.best_solution.fitness:.2f} ({feasible_str}), Pop size = {len(self.population)}")
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"WWO completed in {runtime:.2f} seconds")
        print(f"Final best fitness: {self.best_solution.fitness:.2f}")
        print(f"Known optimum: {self.instance.known_optimum}")
        print(f"Gap: {((self.instance.known_optimum - self.best_solution.fitness) / self.instance.known_optimum * 100):.2f}%")
        
        return self.best_solution, self.fitness_history


def run_experiments():
    """Run WWO on all test instances"""
    dataset_path = "/home/hoangbaoan/repos/INT3103-WaterWaveOptimization/dataset/mknap2-decomposed"
    output_path = "/home/hoangbaoan/repos/INT3103-WaterWaveOptimization/implementation/output/random-first_gen"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Get all .txt files (excluding decompose.py)
    test_files = [f for f in os.listdir(dataset_path) if f.endswith('.txt')]
    test_files.sort()
    
    results = []
    
    for filename in test_files:
        print(f"\n{'='*60}")
        print(f"Testing on {filename}")
        print(f"{'='*60}")
        
        # Load instance
        instance_path = os.path.join(dataset_path, filename)
        instance = MKPInstance(instance_path)
        
        # Run WWO
        wwo = WWOAlgorithm(instance, 
                          np_max=100, np_min=20, 
                          max_generations=500)
        
        best_solution, fitness_history = wwo.run()
        
        # Record results
        gap = ((instance.known_optimum - best_solution.fitness) / instance.known_optimum * 100) if instance.known_optimum > 0 else 0
        
        result = {
            'instance': filename,
            'known_optimum': instance.known_optimum,
            'best_fitness': best_solution.fitness,
            'gap_percent': gap,
            'is_feasible': best_solution.is_feasible(),
            'n_knapsacks': instance.n_knapsacks,
            'n_objects': instance.n_objects
        }
        results.append(result)
        
        # Save detailed results
        output_file = os.path.join(output_path, f"{filename}_wwo_results.txt")
        with open(output_file, 'w') as f:
            f.write(f"WWO Results for {filename}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Problem size: {instance.n_knapsacks} knapsacks, {instance.n_objects} objects\n")
            f.write(f"Known optimum: {instance.known_optimum}\n")
            f.write(f"Best fitness found: {best_solution.fitness:.2f}\n")
            f.write(f"Gap: {gap:.2f}%\n")
            f.write(f"Feasible: {best_solution.is_feasible()}\n")
            f.write(f"\nBest solution:\n")
            f.write(f"{best_solution.solution}\n")
            f.write(f"\nFitness history:\n")
            for i, fitness in enumerate(fitness_history):
                f.write(f"Generation no.{i}: {fitness:.2f}\n")
    
    # Save summary results
    summary_file = os.path.join(output_path, "wwo_summary_results.txt")
    with open(summary_file, 'w') as f:
        f.write("WWO-MKP Summary Results\n")
        f.write("="*70 + "\n")
        f.write(f"{'Instance':<20} {'Known Opt':<12} {'Best Found':<12} {'Gap %':<8} {'Feasible':<10}\n")
        f.write("-"*70 + "\n")
        
        total_gap = 0
        feasible_count = 0
        
        for result in results:
            f.write(f"{result['instance']:<20} "
                   f"{result['known_optimum']:<12} "
                   f"{result['best_fitness']:<12.2f} "
                   f"{result['gap_percent']:<8.2f} "
                   f"{result['is_feasible']:<10}\n")
            
            total_gap += result['gap_percent']
            if result['is_feasible']:
                feasible_count += 1
        
        f.write("-"*70 + "\n")
        f.write(f"Average gap: {total_gap/len(results):.2f}%\n")
        f.write(f"Feasible solutions: {feasible_count}/{len(results)}\n")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Tested {len(results)} instances")
    print(f"Average gap: {total_gap/len(results):.2f}%")
    print(f"Feasible solutions: {feasible_count}/{len(results)}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    run_experiments()
