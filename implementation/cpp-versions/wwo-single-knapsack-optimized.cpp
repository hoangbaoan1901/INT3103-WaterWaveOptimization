#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <cmath>

using namespace std;
using namespace std::chrono;

// Fast Random Number Generator for better performance
class FastRandomGenerator {
private:
    uint64_t state;
    
public:
    FastRandomGenerator() {
        uint64_t seed = (uint64_t)steady_clock::now().time_since_epoch().count();
        seed ^= (seed << 13) ^ (seed >> 7) ^ (seed << 17);
        state = seed ? seed : 0x9e3779b97f4a7c15ULL;
    }
    
    uint64_t next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 2685821657736338717ULL;
    }
    
    int random_int(int min_val, int max_val) {
        return min_val + (int)(next() % (uint64_t)(max_val - min_val + 1));
    }
    
    double random_double() {
        return (next() >> 11) * (1.0 / 9007199254740992.0);
    }
};

// Enhanced Knapsack Instance with precomputed orderings
class EnhancedKnapsackInstance {
public:
    int n_objects;
    long long capacity;
    vector<long long> weights;
    vector<long long> values;
    vector<double> value_density;  // value/weight ratio
    
    // Precomputed item orderings for efficient access
    vector<int> items_by_density;  // sorted by value/weight ratio (descending)
    vector<int> items_by_value;    // sorted by value (descending)
    vector<int> items_by_weight;   // sorted by weight (ascending)
    
    EnhancedKnapsackInstance() {
        load_from_input();
        precompute_orderings();
    }
    
private:
    void load_from_input() {
        ios::sync_with_stdio(false);
        cin.tie(nullptr);
        
        if (!(cin >> n_objects >> capacity)) {
            n_objects = 0;
            return;
        }
        
        weights.resize(n_objects);
        values.resize(n_objects);
        
        for (int i = 0; i < n_objects; i++) {
            cin >> weights[i] >> values[i];
        }
    }
    
    void precompute_orderings() {
        value_density.resize(n_objects);
        items_by_density.resize(n_objects);
        items_by_value.resize(n_objects);
        items_by_weight.resize(n_objects);
        
        // Initialize indices
        for (int i = 0; i < n_objects; i++) {
            items_by_density[i] = i;
            items_by_value[i] = i;
            items_by_weight[i] = i;
            
            // Calculate value density (value/weight ratio)
            value_density[i] = weights[i] ? (double)values[i] / (double)weights[i] : 0.0;
        }
        
        // Sort by density (highest first, ties broken by value)
        sort(items_by_density.begin(), items_by_density.end(), [&](int a, int b) {
            if (value_density[a] == value_density[b]) {
                return values[a] > values[b];
            }
            return value_density[a] > value_density[b];
        });
        
        // Sort by value (highest first)
        sort(items_by_value.begin(), items_by_value.end(), [&](int a, int b) {
            return values[a] > values[b];
        });
        
        // Sort by weight (lightest first)
        sort(items_by_weight.begin(), items_by_weight.end(), [&](int a, int b) {
            return weights[a] < weights[b];
        });
    }
};

// Optimized Solution representation
class OptimizedKnapsackSolution {
public:
    const EnhancedKnapsackInstance* instance;
    vector<unsigned char> solution;  // Use char for memory efficiency
    long long total_weight;
    long long total_value;
    double fitness;
    int age;  // Track solution age for diversity management
    
    OptimizedKnapsackSolution() = default;
    
    OptimizedKnapsackSolution(const EnhancedKnapsackInstance* inst, const vector<int>* init_solution = nullptr) 
        : instance(inst), total_weight(0), total_value(0), age(0) {
        
        solution.assign(instance->n_objects, 0);
        
        if (init_solution) {
            for (int i = 0; i < instance->n_objects; i++) {
                if ((*init_solution)[i]) {
                    solution[i] = 1;
                    total_weight += instance->weights[i];
                    total_value += instance->values[i];
                }
            }
        }
        
        fitness = calculate_fitness();
    }
    
    double calculate_fitness() const {
        if (total_weight <= instance->capacity) {
            return (double)total_value;
        }
        // Heavy penalty for constraint violation
        return (double)total_value - 1000.0 * (double)(total_weight - instance->capacity);
    }
    
    void flip_item(int item_index) {
        if (solution[item_index]) {
            // Remove item
            solution[item_index] = 0;
            total_weight -= instance->weights[item_index];
            total_value -= instance->values[item_index];
        } else {
            // Add item
            solution[item_index] = 1;
            total_weight += instance->weights[item_index];
            total_value += instance->values[item_index];
        }
    }
    
    bool is_feasible() const {
        return total_weight <= instance->capacity;
    }
    
    // Intelligent repair: remove worst items then fill with best items
    void repair_and_fill() {
        // Remove items with worst density until feasible
        if (total_weight > instance->capacity) {
            for (int k = instance->n_objects - 1; k >= 0 && total_weight > instance->capacity; k--) {
                int item = instance->items_by_density[k];
                if (solution[item]) {
                    flip_item(item);
                }
            }
        }
        
        // Fill remaining capacity with best density items
        for (int item : instance->items_by_density) {
            if (!solution[item] && total_weight + instance->weights[item] <= instance->capacity) {
                flip_item(item);
            }
        }
        
        fitness = calculate_fitness();
    }
    
    // Advanced local search with multiple neighborhood structures
    bool perform_local_improvement(const vector<int>& best_unselected_items, 
                                  const vector<int>& worst_selected_items) {
        const long long capacity = instance->capacity;
        const auto& weights = instance->weights;
        const auto& values = instance->values;
        
        // 1-1 swaps: replace one selected item with one unselected item
        for (int unselected : best_unselected_items) {
            for (int selected : worst_selected_items) {
                if (total_weight - weights[selected] + weights[unselected] <= capacity && 
                    values[unselected] > values[selected]) {
                    flip_item(selected);
                    flip_item(unselected);
                    fitness = calculate_fitness();
                    return true;
                }
            }
        }
        
        // 2-1 swaps: replace two selected items with one unselected item
        for (int unselected : best_unselected_items) {
            for (size_t i = 0; i < worst_selected_items.size(); i++) {
                for (size_t j = i + 1; j < worst_selected_items.size(); j++) {
                    int selected1 = worst_selected_items[i];
                    int selected2 = worst_selected_items[j];
                    
                    if (total_weight - weights[selected1] - weights[selected2] + weights[unselected] <= capacity &&
                        values[unselected] > values[selected1] + values[selected2]) {
                        flip_item(selected1);
                        flip_item(selected2);
                        flip_item(unselected);
                        fitness = calculate_fitness();
                        return true;
                    }
                }
            }
        }
        
        // 1-2 swaps: replace one selected item with two unselected items
        for (int selected : worst_selected_items) {
            for (size_t i = 0; i < best_unselected_items.size(); i++) {
                for (size_t j = i + 1; j < best_unselected_items.size(); j++) {
                    int unselected1 = best_unselected_items[i];
                    int unselected2 = best_unselected_items[j];
                    
                    long long new_weight = total_weight - weights[selected] + weights[unselected1] + weights[unselected2];
                    long long new_value = total_value - values[selected] + values[unselected1] + values[unselected2];
                    
                    if (new_weight <= capacity && new_value > total_value) {
                        flip_item(selected);
                        flip_item(unselected1);
                        flip_item(unselected2);
                        fitness = (double)new_value;
                        return true;
                    }
                }
            }
        }
        
        // Limited 2-2 swaps: replace two selected items with two unselected items
        int max_unselected = min((int)best_unselected_items.size(), 12);
        int max_selected = min((int)worst_selected_items.size(), 12);
        
        for (int i = 0; i < max_unselected; i++) {
            for (int j = i + 1; j < max_unselected; j++) {
                int unselected1 = best_unselected_items[i];
                int unselected2 = best_unselected_items[j];
                long long add_weight = weights[unselected1] + weights[unselected2];
                long long add_value = values[unselected1] + values[unselected2];
                
                for (int p = 0; p < max_selected; p++) {
                    for (int q = p + 1; q < max_selected; q++) {
                        int selected1 = worst_selected_items[p];
                        int selected2 = worst_selected_items[q];
                        
                        long long new_weight = total_weight - (weights[selected1] + weights[selected2]) + add_weight;
                        long long new_value = total_value - (values[selected1] + values[selected2]) + add_value;
                        
                        if (new_weight <= capacity && new_value > total_value) {
                            flip_item(selected1);
                            flip_item(selected2);
                            flip_item(unselected1);
                            flip_item(unselected2);
                            fitness = (double)new_value;
                            return true;
                        }
                    }
                }
            }
        }
        
        return false;
    }
    
    // Enhanced local search with multiple rounds
    void perform_local_search(int max_rounds, FastRandomGenerator& rng) {
        vector<int> best_unselected_items;
        vector<int> worst_selected_items;
        best_unselected_items.reserve(28);
        worst_selected_items.reserve(28);
        
        for (int round = 0; round < max_rounds; round++) {
            best_unselected_items.clear();
            worst_selected_items.clear();
            
            // Get best unselected items (by density)
            for (int item : instance->items_by_density) {
                if (!solution[item]) {
                    best_unselected_items.push_back(item);
                    if ((int)best_unselected_items.size() >= 24) break;
                }
            }
            
            // Get worst selected items (by density, reverse order)
            for (int k = instance->n_objects - 1; k >= 0; k--) {
                int item = instance->items_by_density[k];
                if (solution[item]) {
                    worst_selected_items.push_back(item);
                    if ((int)worst_selected_items.size() >= 24) break;
                }
            }
            
            // Try systematic improvements
            if (!perform_local_improvement(best_unselected_items, worst_selected_items)) {
                // If no systematic improvement found, try random beneficial swap
                if (!worst_selected_items.empty() && !best_unselected_items.empty()) {
                    int selected = worst_selected_items[rng.random_int(0, worst_selected_items.size() - 1)];
                    int unselected = best_unselected_items[rng.random_int(0, best_unselected_items.size() - 1)];
                    
                    if (total_weight - instance->weights[selected] + instance->weights[unselected] <= instance->capacity &&
                        instance->values[unselected] >= instance->values[selected]) {
                        flip_item(selected);
                        flip_item(unselected);
                        fitness = calculate_fitness();
                    }
                }
            }
        }
        
        // Ensure feasibility after local search
        if (total_weight > instance->capacity) {
            repair_and_fill();
        }
    }
    
    // Smart propagation using heap-based selection of items to flip
    OptimizedKnapsackSolution propagate(FastRandomGenerator& rng, int num_flips) const {
        OptimizedKnapsackSolution new_solution = *this;
        
        struct FlipCandidate {
            double score;
            int item_index;
        };
        
        auto compare_candidates = [](const FlipCandidate& a, const FlipCandidate& b) {
            return a.score > b.score; // Min-heap by score
        };
        
        const int heap_size = min(instance->n_objects, max(8, num_flips * 4));
        vector<FlipCandidate> candidate_heap;
        candidate_heap.reserve(heap_size);
        
        // Build heap of best flip candidates
        for (int i = 0; i < instance->n_objects; i++) {
            double density = instance->value_density[i];
            double score;
            
            if (new_solution.solution[i]) {
                // For selected items, score is inverse of density (lower density = higher flip priority)
                score = (density > 0) ? 1.0 / density : 1e9;
            } else {
                // For unselected items, score is based on density + small value bonus
                score = 1.2 * density + 0.0001 * instance->values[i];
            }
            
            if ((int)candidate_heap.size() < heap_size) {
                candidate_heap.push_back({score, i});
                push_heap(candidate_heap.begin(), candidate_heap.end(), compare_candidates);
            } else if (score > candidate_heap.front().score) {
                pop_heap(candidate_heap.begin(), candidate_heap.end(), compare_candidates);
                candidate_heap.back() = {score, i};
                push_heap(candidate_heap.begin(), candidate_heap.end(), compare_candidates);
            }
        }
        
        // Flip the best candidates
        for (int flip_count = 0; flip_count < num_flips && !candidate_heap.empty(); flip_count++) {
            pop_heap(candidate_heap.begin(), candidate_heap.end(), compare_candidates);
            int item_to_flip = candidate_heap.back().item_index;
            candidate_heap.pop_back();
            
            new_solution.flip_item(item_to_flip);
        }
        
        // Repair and evaluate the new solution
        if (!new_solution.is_feasible()) {
            new_solution.repair_and_fill();
        } else {
            new_solution.fitness = new_solution.calculate_fitness();
        }
        
        return new_solution;
    }
    
    vector<int> get_selected_items() const {
        vector<int> selected_items;
        selected_items.reserve(instance->n_objects);
        
        for (int i = 0; i < instance->n_objects; i++) {
            if (solution[i]) {
                selected_items.push_back(i + 1); // Convert to 1-based indexing
            }
        }
        
        return selected_items;
    }
};

// Enhanced WWO Algorithm
class EnhancedWWOSolver {
public:
    const EnhancedKnapsackInstance* instance;
    int max_population_size;
    int min_population_size;
    int max_wavelength;
    int min_wavelength;
    double time_limit;
    
    vector<OptimizedKnapsackSolution> population;
    unique_ptr<OptimizedKnapsackSolution> best_solution;
    FastRandomGenerator rng;
    
    EnhancedWWOSolver(const EnhancedKnapsackInstance* inst, 
                     int max_pop_size, int min_pop_size, 
                     int max_lambda, int min_lambda, double time_limit_sec)
        : instance(inst), max_population_size(max_pop_size), min_population_size(min_pop_size),
          max_wavelength(max_lambda), min_wavelength(min_lambda), time_limit(time_limit_sec) {}
    
    // Generate diverse initial solutions using different construction strategies
    vector<int> generate_seed_solution() {
        vector<int> solution(instance->n_objects, 0);
        long long current_weight = 0;
        
        int strategy = rng.random_int(0, 2);
        
        if (strategy == 0) {
            // GRASP-like construction by density with randomization
            vector<int> ordered_items = instance->items_by_density;
            
            // Add randomization to the greedy order
            for (int k = 0; k < (int)ordered_items.size(); k++) {
                int random_range = min((int)ordered_items.size() - 1, k + 6 + rng.random_int(0, 10));
                int random_pick = rng.random_int(k, random_range);
                if (random_pick != k) {
                    swap(ordered_items[k], ordered_items[random_pick]);
                }
            }
            
            for (int item : ordered_items) {
                if (current_weight + instance->weights[item] <= instance->capacity) {
                    solution[item] = 1;
                    current_weight += instance->weights[item];
                }
            }
        } else if (strategy == 1) {
            // Value-based construction with soft threshold
            double total_value = 0;
            for (long long value : instance->values) {
                total_value += value;
            }
            double average_value = total_value / max(1, instance->n_objects);
            double threshold = average_value * (0.6 + 0.4 * rng.random_double());
            
            for (int item : instance->items_by_value) {
                if (instance->values[item] >= threshold && 
                    current_weight + instance->weights[item] <= instance->capacity) {
                    solution[item] = 1;
                    current_weight += instance->weights[item];
                }
            }
        } else {
            // Weight-based construction (lightest first)
            for (int item : instance->items_by_weight) {
                if (current_weight + instance->weights[item] <= instance->capacity) {
                    solution[item] = 1;
                    current_weight += instance->weights[item];
                }
            }
        }
        
        // Fill remaining capacity greedily by density
        for (int item : instance->items_by_density) {
            if (!solution[item] && current_weight + instance->weights[item] <= instance->capacity) {
                solution[item] = 1;
                current_weight += instance->weights[item];
            }
        }
        
        return solution;
    }
    
    OptimizedKnapsackSolution solve() {
        auto start_time = steady_clock::now();
        auto deadline = start_time + duration_cast<steady_clock::duration>(duration<double>(time_limit));
        
        // Initialize population with diverse solutions
        population.clear();
        population.reserve(max_population_size);
        
        for (int i = 0; i < max_population_size; i++) {
            auto seed_solution = generate_seed_solution();
            OptimizedKnapsackSolution solution(instance, &seed_solution);
            solution.repair_and_fill();
            population.push_back(move(solution));
        }
        
        // Find initial best solution
        best_solution = make_unique<OptimizedKnapsackSolution>(
            *max_element(population.begin(), population.end(), 
                        [](const OptimizedKnapsackSolution& a, const OptimizedKnapsackSolution& b) {
                            return a.fitness < b.fitness;
                        }));
        
        double last_best_fitness = best_solution->fitness;
        int stagnation_counter = 0;
        int iteration_counter = 0;
        
        // Main optimization loop
        while (steady_clock::now() < deadline) {
            // Calculate fitness statistics for wavelength scaling
            double min_fitness = population[0].fitness;
            double max_fitness = population[0].fitness;
            
            for (const auto& solution : population) {
                min_fitness = min(min_fitness, solution.fitness);
                max_fitness = max(max_fitness, solution.fitness);
            }
            
            double fitness_range = max(1e-9, max_fitness - min_fitness);
            
            // Propagate each solution
            vector<OptimizedKnapsackSolution> new_population;
            new_population.reserve(population.size());
            
            for (size_t i = 0; i < population.size(); i++) {
                // Check time limit periodically
                if ((++iteration_counter & 7) == 0 && !(steady_clock::now() < deadline)) {
                    break;
                }
                
                auto& current_solution = population[i];
                
                // Calculate wavelength-based number of flips
                double normalized_fitness = (max_fitness - current_solution.fitness) / fitness_range;
                int num_flips = max(1, min(max_wavelength, 
                    1 + (int)((min_wavelength + (max_wavelength - min_wavelength) * normalized_fitness) * 0.7 + 0.5)));
                
                // Propagate solution
                OptimizedKnapsackSolution propagated_solution = current_solution.propagate(rng, num_flips);
                
                // Apply local search periodically
                if ((current_solution.age % 3) == 0) {
                    propagated_solution.perform_local_search(2, rng);
                }
                
                propagated_solution.age = current_solution.age + 1;
                
                // Keep better solution
                if (propagated_solution.fitness >= current_solution.fitness) {
                    new_population.push_back(move(propagated_solution));
                } else {
                    new_population.push_back(move(current_solution));
                }
                
                // Update global best
                if (new_population.back().fitness > best_solution->fitness) {
                    *best_solution = new_population.back();
                    stagnation_counter = 0;
                }
            }
            
            if (new_population.empty()) break;
            population.swap(new_population);
            
            // Stagnation detection and diversification
            if (abs(best_solution->fitness - last_best_fitness) < 1e-9) {
                stagnation_counter++;
            } else {
                stagnation_counter = 0;
                last_best_fitness = best_solution->fitness;
            }
            
            // Diversification when stagnated
            if (stagnation_counter >= 6 && 
                steady_clock::now() + duration_cast<steady_clock::duration>(duration<double>(0.2)) < deadline) {
                
                int solutions_to_replace = max(1, (int)population.size() / 5);
                
                // Find worst solutions
                nth_element(population.begin(), population.begin() + solutions_to_replace, population.end(),
                           [](const OptimizedKnapsackSolution& a, const OptimizedKnapsackSolution& b) {
                               return a.fitness < b.fitness;
                           });
                
                // Replace worst solutions with new diverse ones
                for (int i = 0; i < solutions_to_replace; i++) {
                    auto new_seed = generate_seed_solution();
                    population[i] = OptimizedKnapsackSolution(instance, &new_seed);
                    population[i].repair_and_fill();
                }
                
                stagnation_counter = 0;
            }
            
            // Population size reduction
            if ((int)population.size() > min_population_size) {
                auto best_iter = max_element(population.begin(), population.end(),
                                           [](const OptimizedKnapsackSolution& a, const OptimizedKnapsackSolution& b) {
                                               return a.fitness < b.fitness;
                                           });
                auto worst_iter = min_element(population.begin(), population.end(),
                                            [](const OptimizedKnapsackSolution& a, const OptimizedKnapsackSolution& b) {
                                                return a.fitness < b.fitness;
                                            });
                
                if (worst_iter != best_iter) {
                    population.erase(worst_iter);
                }
            }
            
            if (population.size() < 2) break;
        }
        
        // Final polishing of best solution
        if (steady_clock::now() + duration_cast<steady_clock::duration>(duration<double>(0.01)) < deadline) {
            best_solution->perform_local_search(5, rng);
        }
        
        return *best_solution;
    }
};

int main() {
    try {
        EnhancedKnapsackInstance instance;
        
        if (instance.n_objects <= 0) {
            cout << "0\n";
            return 0;
        }
        
        // Adaptive parameter selection based on problem size
        int max_population_size, max_wavelength;
        
        if (instance.n_objects <= 20) {
            max_population_size = 32;
            max_wavelength = 8;
        } else if (instance.n_objects <= 50) {
            max_population_size = 40;
            max_wavelength = 12;
        } else if (instance.n_objects <= 100) {
            max_population_size = 48;
            max_wavelength = 16;
        } else {
            max_population_size = 60;
            max_wavelength = max(18, instance.n_objects / 6);
        }
        
        int min_population_size = max(8, max_population_size / 4);
        
        // Create and run enhanced WWO solver
        EnhancedWWOSolver solver(&instance, max_population_size, min_population_size, 
                                max_wavelength, 1, 4.0);
        
        OptimizedKnapsackSolution best_solution = solver.solve();
        auto selected_items = best_solution.get_selected_items();
        
        // Output results
        cout << selected_items.size() << "\n";
        for (size_t i = 0; i < selected_items.size(); i++) {
            if (i > 0) cout << " ";
            cout << selected_items[i];
        }
        if (!selected_items.empty()) {
            cout << "\n";
        }
        
    } catch (...) {
        cout << "0\n";
    }
    
    return 0;
}
