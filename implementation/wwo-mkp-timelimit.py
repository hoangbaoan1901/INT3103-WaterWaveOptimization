import os
import random
import time
from typing import List, Tuple, Dict, Any


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
        
        # Parse weights
        weight_lines = []
        line_idx = 1
        while line_idx < len(lines) and len(weight_lines) < self.n_objects:
            values = [int(x) for x in lines[line_idx].split()]
            weight_lines.extend(values)
            line_idx += 1
        
        self.weights = weight_lines[:self.n_objects]
        
        # Parse capacities
        cap_lines = []
        while line_idx < len(lines) and len(cap_lines) < self.n_knapsacks:
            values = [int(x) for x in lines[line_idx].split()]
            cap_lines.extend(values)
            line_idx += 1
        
        self.capacities = cap_lines[:self.n_knapsacks]
        
        # Parse constraint matrix (n_knapsacks x n_objects)
        self.constraints = []
        for k in range(self.n_knapsacks):
            constraint_values = []
            while len(constraint_values) < self.n_objects and line_idx < len(lines):
                values = [int(x) for x in lines[line_idx].split()]
                constraint_values.extend(values)
                line_idx += 1
            self.constraints.append(constraint_values[:self.n_objects])
        
        # Parse known optimum (last line)
        if line_idx < len(lines):
            self.known_optimum = int(lines[line_idx])


class WWOSolution:
    """Class to represent a solution for WWO-MKP"""
    
    def __init__(self, instance: MKPInstance, solution: List[int] = None):
        self.instance = instance
        if solution is None:
            self.solution = self.generate_random_solution()
        else:
            self.solution = solution[:]
        
        self.fitness = self.calculate_fitness()
        self.wavelength = 0.5  # Initial wavelength
    
    def generate_random_solution(self) -> List[int]:
        """Generate a random feasible solution using greedy approach"""
        solution = [0] * self.instance.n_objects
        
        # Calculate profit density (profit/weight ratio)
        profit_density = []
        for i in range(self.instance.n_objects):
            total_constraint = sum(self.instance.constraints[k][i] for k in range(self.instance.n_knapsacks))
            if total_constraint > 0:
                density = self.instance.weights[i] / total_constraint
                profit_density.append((i, density))
        
        profit_density.sort(key=lambda x: x[1], reverse=True)
        
        # Add some randomness by shuffling top candidates
        if len(profit_density) > 10:
            top_part = profit_density[:len(profit_density)//3]
            random.shuffle(top_part)
            profit_density[:len(profit_density)//3] = top_part
        
        # Try to add items greedily
        for item, _ in profit_density:
            temp_solution = solution[:]
            temp_solution[item] = 1
            if self.is_solution_feasible(temp_solution):
                solution[item] = 1
        
        return solution
    
    def is_solution_feasible(self, solution: List[int]) -> bool:
        """Check if a solution is feasible"""
        for k in range(self.instance.n_knapsacks):
            capacity_used = sum(solution[i] * self.instance.constraints[k][i] 
                              for i in range(self.instance.n_objects))
            if capacity_used > self.instance.capacities[k]:
                return False
        return True
    
    def calculate_fitness(self) -> float:
        """Calculate fitness with penalty for infeasible solutions"""
        total_profit = sum(self.solution[i] * self.instance.weights[i] 
                          for i in range(self.instance.n_objects))
        
        # Apply penalty for infeasible solutions
        penalty = 0
        for k in range(self.instance.n_knapsacks):
            capacity_used = sum(self.solution[i] * self.instance.constraints[k][i] 
                              for i in range(self.instance.n_objects))
            if capacity_used > self.instance.capacities[k]:
                penalty += (capacity_used - self.instance.capacities[k]) * 1000
        
        return total_profit - penalty
    
    def is_feasible(self) -> bool:
        """Check if solution is feasible"""
        return self.is_solution_feasible(self.solution)
    
    def repair_solution(self):
        """Repair infeasible solution by removing items with lowest profit density"""
        while not self.is_feasible():
            # Find items to remove based on profit density
            selected_items = [i for i in range(self.instance.n_objects) if self.solution[i] == 1]
            if not selected_items:
                break
            
            # Calculate profit density for selected items
            densities = []
            for i in selected_items:
                total_constraint = sum(self.instance.constraints[k][i] for k in range(self.instance.n_knapsacks))
                if total_constraint > 0:
                    density = self.instance.weights[i] / total_constraint
                    densities.append((i, density))
            
            if not densities:
                break
            
            # Remove item with lowest density
            densities.sort(key=lambda x: x[1])
            worst_item = densities[0][0]
            self.solution[worst_item] = 0
        
        # Recalculate fitness after repair
        self.fitness = self.calculate_fitness()
    
    def copy(self):
        """Create a copy of the solution"""
        new_solution = WWOSolution(self.instance, self.solution)
        new_solution.wavelength = self.wavelength
        return new_solution
    
    def propagate(self) -> 'WWOSolution':
        """Propagate solution using S2 strategy (k-step local search)"""
        new_solution = self.copy()
        
        # Determine number of changes based on wavelength
        k = random.randint(1, max(1, int(self.wavelength)))
        
        for _ in range(k):
            # Random bit flip
            bit_to_flip = random.randint(0, self.instance.n_objects - 1)
            new_solution.solution[bit_to_flip] = 1 - new_solution.solution[bit_to_flip]
        
        # Repair solution if infeasible
        new_solution.fitness = new_solution.calculate_fitness()
        if not new_solution.is_feasible():
            new_solution.repair_solution()
        
        return new_solution
    
    def local_search_breaking(self, n_b: int) -> List['WWOSolution']:
        """Generate neighbors using breaking operator (S6)"""
        neighbors = []
        
        for k in range(1, min(n_b + 1, self.instance.n_objects + 1)):
            neighbor = self.copy()
            
            # Try different strategies for breaking
            strategy = k % 3
            
            if strategy == 0:
                # Strategy 1: Remove k-th smallest profit item and try to add others
                selected_items = [(i, self.instance.weights[i]) for i in range(self.instance.n_objects) 
                                if neighbor.solution[i] == 1]
                if selected_items:
                    selected_items.sort(key=lambda x: x[1])
                    if k <= len(selected_items):
                        item_to_remove = selected_items[k-1][0]
                        neighbor.solution[item_to_remove] = 0
                        
                        # Try to add other items
                        unselected_items = [(i, self.instance.weights[i]) for i in range(self.instance.n_objects) 
                                          if neighbor.solution[i] == 0]
                        unselected_items.sort(key=lambda x: x[1], reverse=True)
                        
                        for item, _ in unselected_items:
                            temp_solution = neighbor.solution[:]
                            temp_solution[item] = 1
                            if neighbor.is_solution_feasible(temp_solution):
                                neighbor.solution[item] = 1
                                break
            
            elif strategy == 1:
                # Strategy 2: Flip a random bit for higher profit
                unselected_items = [i for i in range(self.instance.n_objects) if neighbor.solution[i] == 0]
                if unselected_items:
                    # Sort by profit and try to add high-profit items
                    profit_items = [(i, self.instance.weights[i]) for i in unselected_items]
                    profit_items.sort(key=lambda x: x[1], reverse=True)
                    
                    for item, _ in profit_items[:k]:
                        temp_solution = neighbor.solution[:]
                        temp_solution[item] = 1
                        if neighbor.is_solution_feasible(temp_solution):
                            neighbor.solution[item] = 1
                            break
            
            else:
                # Strategy 3: Interchange two opposite components
                selected = [i for i in range(self.instance.n_objects) if neighbor.solution[i] == 1]
                unselected = [i for i in range(self.instance.n_objects) if neighbor.solution[i] == 0]
                
                if selected and unselected:
                    # Try swapping items
                    for _ in range(min(k, len(selected), len(unselected))):
                        sel_item = random.choice(selected)
                        unsel_item = random.choice(unselected)
                        
                        temp_solution = neighbor.solution[:]
                        temp_solution[sel_item] = 0
                        temp_solution[unsel_item] = 1
                        
                        if neighbor.is_solution_feasible(temp_solution):
                            neighbor.solution[sel_item] = 0
                            neighbor.solution[unsel_item] = 1
                            break
            
            neighbor.fitness = neighbor.calculate_fitness()
            if not neighbor.is_feasible():
                neighbor.repair_solution()
            
            neighbors.append(neighbor)
        
        return neighbors


class WWOAlgorithmTimeLimited:
    """Time-limited Water Wave Optimization Algorithm for Multiple Knapsack Problem"""
    
    def __init__(self, instance: MKPInstance, 
                 np_max: int = 30, np_min: int = 8, 
                 lambda_max: int = None, lambda_min: int = 1,
                 n_b: int = 3, time_limit: float = 5.0):
        
        self.instance = instance
        self.np_max = np_max
        self.np_min = np_min
        self.lambda_max = lambda_max if lambda_max else min(10, instance.n_objects // 4)
        self.lambda_min = lambda_min
        self.n_b = n_b
        self.time_limit = time_limit
        
        self.population = []
        self.best_solution = None
        self.generation = 0
        self.current_np = np_max
        self.start_time = None
        
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
            if total_fitness > 0:
                ratio = (total_fitness - solution.fitness) / total_fitness
                solution.wavelength = self.lambda_max * ratio
                solution.wavelength = max(self.lambda_min, min(self.lambda_max, solution.wavelength))
            else:
                solution.wavelength = self.lambda_max / 2
    
    def update_population_size(self):
        """Linearly decrease population size (S10)"""
        if self.generation > 0 and self.current_np > self.np_min:
            # Remove worst solution
            if len(self.population) > self.np_min:
                worst_idx = min(range(len(self.population)), key=lambda i: self.population[i].fitness)
                self.population.pop(worst_idx)
                self.current_np = len(self.population)
    
    def time_remaining(self) -> float:
        """Check remaining time"""
        if self.start_time is None:
            return self.time_limit
        return self.time_limit - (time.time() - self.start_time)
    
    def run(self) -> Tuple[WWOSolution, List[float]]:
        """Run the WWO algorithm with time limit"""
        self.start_time = time.time()
        print(f"Starting WWO with {self.time_limit}s time limit...")
        
        # Initialize population
        self.initialize_population()
        self.fitness_history = [sol.fitness for sol in self.population]
        self.best_fitness_history = [self.best_solution.fitness]
        
        self.generation = 0
        
        # Main optimization loop
        while self.time_remaining() > 0:
            # Calculate wavelengths
            self.calculate_wavelengths()
            
            # Propagate each solution
            new_population = []
            for solution in self.population:
                if self.time_remaining() <= 0:
                    break
                
                # Propagate solution
                new_solution = solution.propagate()
                
                # Apply S7: Replace only if better
                if new_solution.fitness > solution.fitness:
                    new_population.append(new_solution)
                    
                    # Check if new best solution found
                    if new_solution.fitness > self.best_solution.fitness:
                        self.best_solution = new_solution.copy()
                        
                        # Apply breaking operator (S6) - but only if time allows
                        if self.time_remaining() > 0.1:  # Leave some time buffer
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
            
            # Record statistics
            current_fitness = [sol.fitness for sol in self.population]
            self.fitness_history.extend(current_fitness)
            self.best_fitness_history.append(self.best_solution.fitness)
            
            self.generation += 1
            
            # Print progress every few generations
            if self.generation % 10 == 0:
                elapsed = time.time() - self.start_time
                print(f"Gen {self.generation}: Best={self.best_solution.fitness}, "
                      f"Pop size={len(self.population)}, Time={elapsed:.2f}s")
        
        elapsed_time = time.time() - self.start_time
        print(f"Optimization completed in {elapsed_time:.3f}s after {self.generation} generations")
        
        return self.best_solution, self.best_fitness_history


def run_time_limited_experiments():
    """Run time-limited WWO on test instances"""
    dataset_path = "/home/hoangbaoan/repos/INT3103-WaterWaveOptimization/dataset/mknap2-decomposed"
    output_path = "/home/hoangbaoan/repos/INT3103-WaterWaveOptimization/implementation/output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Get all .txt files (excluding decompose.py)
    test_files = [f for f in os.listdir(dataset_path) if f.endswith('.txt')]
    test_files.sort()
    
    results = []
    time_limit = 5.0  # 5 seconds
    
    print(f"Running time-limited WWO (5 seconds) on {len(test_files)} instances...")
    print("=" * 70)
    
    for filename in test_files:
        print(f"\nTesting {filename}...")
        instance_path = os.path.join(dataset_path, filename)
        
        try:
            # Load instance
            instance = MKPInstance(instance_path)
            
            # Run WWO with time limit
            wwo = WWOAlgorithmTimeLimited(
                instance, 
                np_max=30, 
                np_min=8, 
                time_limit=time_limit
            )
            
            start_time = time.time()
            best_solution, fitness_history = wwo.run()
            elapsed_time = time.time() - start_time
            
            # Calculate gap
            gap = 0.0
            if instance.known_optimum > 0:
                gap = ((instance.known_optimum - best_solution.fitness) / instance.known_optimum) * 100
            
            # Check feasibility
            is_feasible = best_solution.is_feasible()
            
            result = {
                'instance': filename,
                'best_fitness': best_solution.fitness,
                'known_optimum': instance.known_optimum,
                'gap': gap,
                'feasible': is_feasible,
                'time': elapsed_time,
                'generations': wwo.generation
            }
            results.append(result)
            
            # Save detailed results
            output_file = os.path.join(output_path, f"{filename}_wwo_timelimited_results.txt")
            with open(output_file, 'w') as f:
                f.write(f"WWO Time-Limited Results for {filename}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Instance: {filename}\n")
                f.write(f"Objects: {instance.n_objects}\n")
                f.write(f"Knapsacks: {instance.n_knapsacks}\n")
                f.write(f"Known Optimum: {instance.known_optimum}\n")
                f.write(f"Time Limit: {time_limit}s\n")
                f.write(f"Actual Time: {elapsed_time:.3f}s\n")
                f.write(f"Generations: {wwo.generation}\n\n")
                f.write(f"Best Fitness: {best_solution.fitness}\n")
                f.write(f"Gap: {gap:.2f}%\n")
                f.write(f"Feasible: {is_feasible}\n\n")
                f.write(f"Solution: {best_solution.solution}\n\n")
                f.write("Fitness History:\n")
                for i, fitness in enumerate(fitness_history[-20:]):  # Last 20 values
                    f.write(f"  {len(fitness_history)-20+i}: {fitness}\n")
            
            # Print summary
            status = "✓" if is_feasible else "✗"
            print(f"  Result: {best_solution.fitness} | Gap: {gap:.2f}% | {status} | "
                  f"{elapsed_time:.2f}s | {wwo.generation} gen")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue
    
    # Save summary results
    summary_file = os.path.join(output_path, "wwo_timelimited_summary_results.txt")
    with open(summary_file, 'w') as f:
        f.write("WWO Time-Limited Algorithm Results Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Time Limit: {time_limit} seconds per instance\n\n")
        f.write(f"{'Instance':<20} {'Best':<10} {'Optimum':<10} {'Gap%':<8} {'Feasible':<8} {'Time':<6} {'Gen':<5}\n")
        f.write("-" * 80 + "\n")
        
        total_gap = 0
        feasible_count = 0
        total_time = 0
        total_generations = 0
        
        for result in results:
            f.write(f"{result['instance']:<20} {result['best_fitness']:<10} "
                   f"{result['known_optimum']:<10} {result['gap']:<8.2f} "
                   f"{'Yes' if result['feasible'] else 'No':<8} "
                   f"{result['time']:<6.2f} {result['generations']:<5}\n")
            
            total_gap += result['gap']
            total_time += result['time']
            total_generations += result['generations']
            if result['feasible']:
                feasible_count += 1
        
        f.write("-" * 80 + "\n")
        f.write(f"Average Gap: {total_gap/len(results):.2f}%\n")
        f.write(f"Feasible Solutions: {feasible_count}/{len(results)}\n")
        f.write(f"Average Time: {total_time/len(results):.3f}s\n")
        f.write(f"Average Generations: {total_generations/len(results):.1f}\n")
    
    print(f"\n{'='*70}")
    print("TIME-LIMITED SUMMARY")
    print(f"{'='*70}")
    print(f"Tested {len(results)} instances")
    print(f"Time limit: {time_limit} seconds per instance")
    print(f"Average gap: {total_gap/len(results):.2f}%")
    print(f"Average time: {total_time/len(results):.3f}s")
    print(f"Average generations: {total_generations/len(results):.1f}")
    print(f"Feasible solutions: {feasible_count}/{len(results)}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    run_time_limited_experiments()
