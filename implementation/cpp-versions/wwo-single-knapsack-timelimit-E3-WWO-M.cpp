#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <climits>
#include <cmath>
#include <exception>

using namespace std;
using namespace std::chrono;

class SingleKnapsackInstance {
public:
    int n_objects;
    long long capacity;
    vector<long long> weights;
    vector<long long> values;
    
    SingleKnapsackInstance() {
        load_from_stdin();
    }
    
private:
    void load_from_stdin() {
        cin >> n_objects >> capacity;
        weights.resize(n_objects);
        values.resize(n_objects);
        
        for (int i = 0; i < n_objects; i++) {
            cin >> weights[i] >> values[i];
        }
    }
};

class KnapsackSolution {
public:
    const SingleKnapsackInstance* instance;
    vector<int> solution;
    double fitness;
    double wavelength;
    int age; // Track how long solution has been in population
    
    KnapsackSolution(const SingleKnapsackInstance* inst, const vector<int>* sol = nullptr) 
        : instance(inst), wavelength(0.5), age(0) {
        if (sol == nullptr) {
            solution = generate_random_solution();
        } else {
            solution = *sol;
        }
        fitness = calculate_fitness();
    }
    
    vector<int> generate_random_solution() {
        static random_device rd;
        static mt19937 gen(rd());
        uniform_real_distribution<double> dis(0.0, 1.0);
        uniform_int_distribution<int> strategy_dis(0, 3);
        
        vector<int> sol(instance->n_objects, 0);
        int strategy = strategy_dis(gen);
        
        if (strategy == 0) {
            // Value-to-weight ratio greedy with randomization
            vector<pair<double, int>> ratios;
            for (int i = 0; i < instance->n_objects; i++) {
                if (instance->weights[i] > 0) {
                    double ratio = (double)instance->values[i] / instance->weights[i];
                    // Add noise to ratio for diversity
                    ratio *= (0.9 + 0.2 * dis(gen));
                    ratios.push_back({ratio, i});
                }
            }
            sort(ratios.begin(), ratios.end(), greater<pair<double, int>>());
            
            long long current_weight = 0;
            for (auto& p : ratios) {
                int item_idx = p.second;
                if (current_weight + instance->weights[item_idx] <= instance->capacity) {
                    sol[item_idx] = 1;
                    current_weight += instance->weights[item_idx];
                }
            }
        }
        else if (strategy == 1) {
            // Pure value greedy with randomization
            vector<pair<long long, int>> values;
            for (int i = 0; i < instance->n_objects; i++) {
                values.push_back({instance->values[i], i});
            }
            sort(values.begin(), values.end(), greater<pair<long long, int>>());
            
            long long current_weight = 0;
            for (auto& p : values) {
                int item_idx = p.second;
                if (current_weight + instance->weights[item_idx] <= instance->capacity && dis(gen) > 0.1) {
                    sol[item_idx] = 1;
                    current_weight += instance->weights[item_idx];
                }
            }
        }
        else if (strategy == 2) {
            // Lightest items first with value filter
            vector<pair<long long, int>> weights;
            for (int i = 0; i < instance->n_objects; i++) {
                if (instance->values[i] > 0) {
                    weights.push_back({instance->weights[i], i});
                }
            }
            sort(weights.begin(), weights.end());
            
            long long current_weight = 0;
            double avg_value = 0;
            for (int i = 0; i < instance->n_objects; i++) {
                avg_value += instance->values[i];
            }
            avg_value /= instance->n_objects;
            
            for (auto& p : weights) {
                int item_idx = p.second;
                if (current_weight + instance->weights[item_idx] <= instance->capacity && 
                    instance->values[item_idx] >= avg_value * dis(gen)) {
                    sol[item_idx] = 1;
                    current_weight += instance->weights[item_idx];
                }
            }
        }
        else {
            // Random selection with bias towards good ratio items
            vector<pair<double, int>> ratios;
            for (int i = 0; i < instance->n_objects; i++) {
                if (instance->weights[i] > 0) {
                    double ratio = (double)instance->values[i] / instance->weights[i];
                    ratios.push_back({ratio, i});
                }
            }
            sort(ratios.begin(), ratios.end(), greater<pair<double, int>>());
            
            long long current_weight = 0;
            for (int i = 0; i < (int)ratios.size(); i++) {
                int item_idx = ratios[i].second;
                double prob = 1.0 / (1.0 + i * 0.1); // Higher probability for better ratios
                if (current_weight + instance->weights[item_idx] <= instance->capacity && 
                    dis(gen) < prob) {
                    sol[item_idx] = 1;
                    current_weight += instance->weights[item_idx];
                }
            }
        }
        
        return sol;
    }
    
    double calculate_fitness() {
        long long total_value = 0;
        long long total_weight = 0;
        
        for (int i = 0; i < instance->n_objects; i++) {
            total_value += solution[i] * instance->values[i];
            total_weight += solution[i] * instance->weights[i];
        }
        
        // Apply penalty if capacity is exceeded
        if (total_weight > instance->capacity) {
            long long penalty = (total_weight - instance->capacity) * 1000;
            return (double)total_value - penalty;
        }
        
        return (double)total_value;
    }
    
    bool is_feasible() {
        long long total_weight = 0;
        for (int i = 0; i < instance->n_objects; i++) {
            total_weight += solution[i] * instance->weights[i];
        }
        return total_weight <= instance->capacity;
    }
    
    long long get_weight() {
        long long total_weight = 0;
        for (int i = 0; i < instance->n_objects; i++) {
            total_weight += solution[i] * instance->weights[i];
        }
        return total_weight;
    }
    
    void repair_solution() {
        // More intelligent repair: try to maintain high-value items
        while (!is_feasible()) {
            // Find selected items and their value/weight ratios
            vector<pair<double, int>> selected_items;
            for (int i = 0; i < instance->n_objects; i++) {
                if (solution[i] == 1 && instance->weights[i] > 0) {
                    double ratio = (double)instance->values[i] / instance->weights[i];
                    selected_items.push_back({ratio, i});
                }
            }
            
            if (selected_items.empty()) break;
            
            // Remove item with lowest value/weight ratio
            sort(selected_items.begin(), selected_items.end());
            int worst_item = selected_items[0].second;
            solution[worst_item] = 0;
        }
        
        // Try to add more items after repair (fill remaining capacity)
        vector<pair<double, int>> unselected_ratios;
        for (int i = 0; i < instance->n_objects; i++) {
            if (solution[i] == 0 && instance->weights[i] > 0) {
                double ratio = (double)instance->values[i] / instance->weights[i];
                unselected_ratios.push_back({ratio, i});
            }
        }
        
        sort(unselected_ratios.begin(), unselected_ratios.end(), greater<pair<double, int>>());
        
        for (auto& p : unselected_ratios) {
            int item_idx = p.second;
            if (get_weight() + instance->weights[item_idx] <= instance->capacity) {
                solution[item_idx] = 1;
            }
        }
        
        // Recalculate fitness after repair
        fitness = calculate_fitness();
    }

    KnapsackSolution copy() {
        KnapsackSolution copied(instance, &solution);
        copied.wavelength = wavelength;
        copied.age = age;
        return copied;
    }
    
    // Add perturbation method for escaping local optima
    KnapsackSolution perturb(mt19937& rng, double intensity = 0.3) {
        KnapsackSolution perturbed = copy();
        
        uniform_real_distribution<double> prob_dist(0.0, 1.0);
        
        // Randomly flip bits with given intensity
        for (int i = 0; i < instance->n_objects; i++) {
            if (prob_dist(rng) < intensity) {
                perturbed.solution[i] = 1 - perturbed.solution[i];
            }
        }
        
        // Repair and optimize
        perturbed.fitness = perturbed.calculate_fitness();
        if (!perturbed.is_feasible()) {
            perturbed.repair_solution();
        }
        
        perturbed.age = 0; // Reset age for perturbed solution
        return perturbed;
    }
    
    KnapsackSolution propagate(mt19937& rng) {
        KnapsackSolution new_solution = copy();
        new_solution.wavelength = wavelength;
        
        // Determine number of changes based on wavelength
        int max_changes = max(1, (int)wavelength);
        uniform_int_distribution<int> changes_dist(1, max_changes);
        int k = changes_dist(rng);
        
        // Smart bit selection: prefer items with better value/weight ratios for flipping
        vector<pair<double, int>> candidates;
        for (int i = 0; i < instance->n_objects; i++) {
            double ratio = instance->weights[i] > 0 ? (double)instance->values[i] / instance->weights[i] : 0;
            
            if (solution[i] == 0) {
                // For unselected items, higher ratio = higher flip probability
                candidates.push_back({ratio * 2.0, i});
            } else {
                // For selected items, lower ratio = higher flip probability
                candidates.push_back({1.0 / (ratio + 0.1), i});
            }
        }
        
        sort(candidates.begin(), candidates.end(), greater<pair<double, int>>());
        
        // Select items to flip with bias towards better candidates
        uniform_real_distribution<double> prob_dist(0.0, 1.0);
        int flipped = 0;
        
        for (int i = 0; i < (int)candidates.size() && flipped < k; i++) {
            double flip_prob = 1.0 / (1.0 + i * 0.2); // Decreasing probability
            if (prob_dist(rng) < flip_prob) {
                int bit_to_flip = candidates[i].second;
                new_solution.solution[bit_to_flip] = 1 - new_solution.solution[bit_to_flip];
                flipped++;
            }
        }
        
        // If we haven't flipped enough, do random flips
        uniform_int_distribution<int> bit_dist(0, instance->n_objects - 1);
        while (flipped < k) {
            int bit_to_flip = bit_dist(rng);
            new_solution.solution[bit_to_flip] = 1 - new_solution.solution[bit_to_flip];
            flipped++;
        }
        
        // Repair if infeasible
        new_solution.fitness = new_solution.calculate_fitness();
        if (!new_solution.is_feasible()) {
            new_solution.repair_solution();
        }
        
        return new_solution;
    }
    
    vector<KnapsackSolution> local_search_breaking(int n_b, mt19937& rng) {
        vector<KnapsackSolution> neighbors;
        
        // WWO-M: Dynamic selection among three breaking operators
        uniform_int_distribution<int> operator_dist(0, 2);
        
        for (int k = 1; k <= min(n_b, instance->n_objects); k++) {
            // Try all three operators and keep the best result
            vector<KnapsackSolution> operator_results;
            
            // Operator 1: Remove k-th smallest profit item and add items in decreasing order of profit
            {
                KnapsackSolution neighbor = copy();
                vector<pair<long long, int>> selected_items;
                for (int i = 0; i < instance->n_objects; i++) {
                    if (neighbor.solution[i] == 1) {
                        selected_items.push_back({instance->values[i], i});
                    }
                }
                
                if (!selected_items.empty() && k <= (int)selected_items.size()) {
                    sort(selected_items.begin(), selected_items.end()); // Sort by value (ascending)
                    int item_to_remove = selected_items[k-1].second;
                    neighbor.solution[item_to_remove] = 0;
                    
                    // Add items in decreasing order of profit until no more can be added
                    vector<pair<long long, int>> unselected_items;
                    for (int i = 0; i < instance->n_objects; i++) {
                        if (neighbor.solution[i] == 0) {
                            unselected_items.push_back({instance->values[i], i});
                        }
                    }
                    
                    sort(unselected_items.begin(), unselected_items.end(), greater<pair<long long, int>>());
                    
                    for (auto& p : unselected_items) {
                        int item = p.second;
                        if (neighbor.get_weight() + instance->weights[item] <= instance->capacity) {
                            neighbor.solution[item] = 1;
                        }
                    }
                }
                
                neighbor.fitness = neighbor.calculate_fitness();
                if (!neighbor.is_feasible()) {
                    neighbor.repair_solution();
                }
                operator_results.push_back(neighbor);
            }
            
            // Operator 2: Reverse a component for higher profit (flip bit of high-value unselected item)
            {
                KnapsackSolution neighbor = copy();
                
                // Find unselected items with high value/weight ratio
                vector<pair<double, int>> unselected_ratios;
                for (int i = 0; i < instance->n_objects; i++) {
                    if (neighbor.solution[i] == 0 && instance->weights[i] > 0) {
                        double ratio = (double)instance->values[i] / instance->weights[i];
                        unselected_ratios.push_back({ratio, i});
                    }
                }
                
                if (!unselected_ratios.empty()) {
                    sort(unselected_ratios.begin(), unselected_ratios.end(), greater<pair<double, int>>());
                    
                    // Try to add the k-th best unselected item (if within bounds)
                    int target_idx = min(k-1, (int)unselected_ratios.size()-1);
                    int item_to_add = unselected_ratios[target_idx].second;
                    
                    // Try adding this item, remove items if necessary to make space
                    if (neighbor.get_weight() + instance->weights[item_to_add] > instance->capacity) {
                        // Need to remove some items first
                        vector<pair<double, int>> selected_ratios;
                        for (int i = 0; i < instance->n_objects; i++) {
                            if (neighbor.solution[i] == 1 && instance->weights[i] > 0) {
                                double ratio = (double)instance->values[i] / instance->weights[i];
                                selected_ratios.push_back({ratio, i});
                            }
                        }
                        
                        sort(selected_ratios.begin(), selected_ratios.end()); // Sort by ratio (ascending)
                        
                        // Remove items with lowest ratios until the new item fits
                        for (auto& p : selected_ratios) {
                            if (neighbor.get_weight() + instance->weights[item_to_add] <= instance->capacity) {
                                break;
                            }
                            neighbor.solution[p.second] = 0;
                        }
                    }
                    
                    // Add the target item
                    if (neighbor.get_weight() + instance->weights[item_to_add] <= instance->capacity) {
                        neighbor.solution[item_to_add] = 1;
                    }
                }
                
                neighbor.fitness = neighbor.calculate_fitness();
                if (!neighbor.is_feasible()) {
                    neighbor.repair_solution();
                }
                operator_results.push_back(neighbor);
            }
            
            // Operator 3: Interchange two opposite components for higher profit
            {
                KnapsackSolution neighbor = copy();
                
                vector<int> selected, unselected;
                for (int i = 0; i < instance->n_objects; i++) {
                    if (neighbor.solution[i] == 1) {
                        selected.push_back(i);
                    } else {
                        unselected.push_back(i);
                    }
                }
                
                if (!selected.empty() && !unselected.empty()) {
                    double best_improvement = -1e9;
                    int best_sel = -1, best_unsel = -1;
                    
                    // Consider more swaps for better exploration
                    int max_check_sel = min(k + 2, (int)selected.size());
                    int max_check_unsel = min(k + 2, (int)unselected.size());
                    
                    for (int i = 0; i < max_check_sel; i++) {
                        int sel_item = selected[i];
                        for (int j = 0; j < max_check_unsel; j++) {
                            int unsel_item = unselected[j];
                            
                            // Calculate weight change
                            long long weight_change = instance->weights[unsel_item] - instance->weights[sel_item];
                            
                            if (neighbor.get_weight() + weight_change <= instance->capacity) {
                                // Calculate value improvement
                                double improvement = (double)instance->values[unsel_item] - instance->values[sel_item];
                                
                                // Prefer swaps with higher value improvement and better ratios
                                double unsel_ratio = instance->weights[unsel_item] > 0 ? 
                                    (double)instance->values[unsel_item] / instance->weights[unsel_item] : 0;
                                double sel_ratio = instance->weights[sel_item] > 0 ? 
                                    (double)instance->values[sel_item] / instance->weights[sel_item] : 0;
                                
                                improvement += (unsel_ratio - sel_ratio) * 10; // Bias towards better ratios
                                
                                if (improvement > best_improvement) {
                                    best_improvement = improvement;
                                    best_sel = sel_item;
                                    best_unsel = unsel_item;
                                }
                            }
                        }
                    }
                    
                    if (best_sel != -1 && best_unsel != -1) {
                        neighbor.solution[best_sel] = 0;
                        neighbor.solution[best_unsel] = 1;
                    }
                }
                
                neighbor.fitness = neighbor.calculate_fitness();
                if (!neighbor.is_feasible()) {
                    neighbor.repair_solution();
                }
                operator_results.push_back(neighbor);
            }
            
            // Select the best result from the three operators
            auto best_operator_result = max_element(operator_results.begin(), operator_results.end(),
                [](const KnapsackSolution& a, const KnapsackSolution& b) {
                    return a.fitness < b.fitness;
                });
            
            if (best_operator_result != operator_results.end()) {
                neighbors.push_back(*best_operator_result);
            }
        }
        
        return neighbors;
    }
    
    vector<int> get_selected_items() {
        vector<int> selected;
        for (int i = 0; i < instance->n_objects; i++) {
            if (solution[i] == 1) {
                selected.push_back(i + 1); // 1-based indexing
            }
        }
        return selected;
    }
};

class WWOSingleKnapsack {
public:
    const SingleKnapsackInstance* instance;
    int np_max, np_min;
    int lambda_max, lambda_min;
    int n_b;
    double time_limit;
    
    vector<KnapsackSolution> population;
    KnapsackSolution* best_solution;
    int generation;
    int current_np;
    high_resolution_clock::time_point start_time;
    mt19937 rng;
    
    WWOSingleKnapsack(const SingleKnapsackInstance* inst, 
                      int np_max = 30, int np_min = 8,
                      int lambda_max = -1, int lambda_min = 1,
                      int n_b = 3, double time_limit = 5.0)
        : instance(inst), np_max(np_max), np_min(np_min), 
          lambda_min(lambda_min), n_b(n_b), time_limit(time_limit),
          generation(0), current_np(np_max), best_solution(nullptr) {
        
        if (lambda_max == -1) {
            this->lambda_max = min(10, instance->n_objects / 4);
        } else {
            this->lambda_max = lambda_max;
        }
        
        // Initialize random number generator
        random_device rd;
        rng.seed(rd());
    }
    
    ~WWOSingleKnapsack() {
        delete best_solution;
    }
    
    void initialize_population() {
        population.clear();
        
        // Generate more diverse initial solutions using different strategies
        vector<int> strategies = {0, 1, 2, 3}; // Different construction strategies
        
        for (int i = 0; i < np_max; i++) {
            KnapsackSolution sol(instance);
            
            // Apply local improvement to some initial solutions
            if (i < np_max / 2) {
                // Apply more intensive local search to more solutions
                bool improved = true;
                int iterations = 0;
                while (improved && iterations < 15) {
                    improved = false;
                    iterations++;
                    
                    // Try swapping items
                    for (int j = 0; j < instance->n_objects && !improved; j++) {
                        for (int k = j + 1; k < instance->n_objects && !improved; k++) {
                            if (sol.solution[j] != sol.solution[k]) {
                                // Try swapping
                                vector<int> new_sol = sol.solution;
                                new_sol[j] = 1 - new_sol[j];
                                new_sol[k] = 1 - new_sol[k];
                                
                                KnapsackSolution temp_sol(instance, &new_sol);
                                if (temp_sol.is_feasible() && temp_sol.fitness > sol.fitness) {
                                    sol = temp_sol;
                                    improved = true;
                                }
                            }
                        }
                    }
                    
                    // Try 3-opt moves (swap 3 items)
                    if (!improved && iterations % 3 == 0) {
                        for (int j = 0; j < instance->n_objects && !improved; j++) {
                            for (int k = j + 1; k < instance->n_objects && !improved; k++) {
                                for (int l = k + 1; l < instance->n_objects && !improved; l++) {
                                    vector<int> new_sol = sol.solution;
                                    new_sol[j] = 1 - new_sol[j];
                                    new_sol[k] = 1 - new_sol[k];
                                    new_sol[l] = 1 - new_sol[l];
                                    
                                    KnapsackSolution temp_sol(instance, &new_sol);
                                    if (temp_sol.is_feasible() && temp_sol.fitness > sol.fitness) {
                                        sol = temp_sol;
                                        improved = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            population.push_back(sol);
        }
        
        // Ensure diversity by adding some very different solutions
        for (int i = 0; i < min(5, np_max / 4); i++) {
            KnapsackSolution diverse_sol = population[0].perturb(rng, 0.5);
            population.push_back(diverse_sol);
        }
        
        // Trim population to target size
        if ((int)population.size() > np_max) {
            // Sort by fitness and keep best solutions
            sort(population.begin(), population.end(),
                [](const KnapsackSolution& a, const KnapsackSolution& b) {
                    return a.fitness > b.fitness;
                });
            population.erase(population.begin() + np_max, population.end());
        }
        
        // Find initial best solution
        auto best_it = max_element(population.begin(), population.end(),
            [](const KnapsackSolution& a, const KnapsackSolution& b) {
                return a.fitness < b.fitness;
            });
        
        delete best_solution;
        best_solution = new KnapsackSolution(best_it->copy());
    }
    
    void calculate_wavelengths() {
        if (population.empty()) return;
        
        // Calculate sum of all fitness values (using only positive fitness values)
        double total_fitness = 0;
        for (const auto& sol : population) {
            total_fitness += max(0.0, sol.fitness);
        }
        
        // Calculate wavelength using E3: λ_x = λ_max * (Σf(x') - f(x)) / Σf(x')
        for (auto& solution : population) {
            if (total_fitness > 0) {
                double normalized_fitness = max(0.0, solution.fitness);
                solution.wavelength = lambda_max * (total_fitness - normalized_fitness) / total_fitness;
            } else {
                // If all fitness values are negative or zero, set to middle value
                solution.wavelength = lambda_max / 2.0;
            }
            
            // Ensure wavelength is within bounds
            solution.wavelength = max((double)lambda_min, min((double)lambda_max, solution.wavelength));
        }
    }
    
    void update_population_size() {
        if (generation > 0 && (int)population.size() > np_min) {
            // Remove worst solution
            auto worst_it = min_element(population.begin(), population.end(),
                [](const KnapsackSolution& a, const KnapsackSolution& b) {
                    return a.fitness < b.fitness;
                });
            population.erase(worst_it);
            current_np = population.size();
        }
    }
    
    double time_remaining() {
        auto current_time = high_resolution_clock::now();
        auto elapsed = duration_cast<duration<double>>(current_time - start_time);
        return time_limit - elapsed.count();
    }
    
    KnapsackSolution run() {
        start_time = high_resolution_clock::now();
        
        // Initialize population
        initialize_population();
        
        generation = 0;
        double last_best_fitness = best_solution->fitness;
        int stagnation_counter = 0;
        double target_ratio = 0.76; // Try to detect 0.75 trap more precisely
        
        // Main optimization loop
        while (time_remaining() > 0) {
            // Age all solutions
            for (auto& sol : population) {
                sol.age++;
            }
            
            // Calculate wavelengths
            calculate_wavelengths();
            
            // Propagate each solution
            vector<KnapsackSolution> new_population;
            for (auto& solution : population) {
                if (time_remaining() <= 0) break;
                
                // Propagate solution
                KnapsackSolution new_solution = solution.propagate(rng);
                
                // Replace only if better (S7)
                if (new_solution.fitness > solution.fitness) {
                    new_population.push_back(new_solution);
                    
                    // Check if new best solution found
                    if (new_solution.fitness > best_solution->fitness) {
                        delete best_solution;
                        best_solution = new KnapsackSolution(new_solution.copy());
                        
                        // Apply breaking operator (S6) - but only if time allows
                        if (time_remaining() > 0.1) {
                            vector<KnapsackSolution> neighbors = new_solution.local_search_breaking(n_b, rng);
                            if (!neighbors.empty()) {
                                auto best_neighbor_it = max_element(neighbors.begin(), neighbors.end(),
                                    [](const KnapsackSolution& a, const KnapsackSolution& b) {
                                        return a.fitness < b.fitness;
                                    });
                                
                                if (best_neighbor_it->fitness > best_solution->fitness) {
                                    delete best_solution;
                                    best_solution = new KnapsackSolution(best_neighbor_it->copy());
                                }
                            }
                        }
                        stagnation_counter = 0; // Reset stagnation counter
                    }
                } else {
                    new_population.push_back(solution);
                }
            }
            
            population = new_population;
            
            // Check for stagnation and apply diversification
            if (abs(best_solution->fitness - last_best_fitness) < 1e-6) {
                stagnation_counter++;
            } else {
                stagnation_counter = 0;
                last_best_fitness = best_solution->fitness;
            }
            
            // Aggressive escape strategies for local optima
            bool potential_trap = false;
            
            // Check if we're potentially in the 0.75 trap
            if (stagnation_counter >= 5) {
                // Calculate theoretical maximum (greedy upper bound)
                vector<pair<double, int>> ratios;
                for (int i = 0; i < instance->n_objects; i++) {
                    if (instance->weights[i] > 0) {
                        double ratio = (double)instance->values[i] / instance->weights[i];
                        ratios.push_back({ratio, i});
                    }
                }
                sort(ratios.begin(), ratios.end(), greater<pair<double, int>>());
                
                long long greedy_value = 0, greedy_weight = 0;
                for (auto& p : ratios) {
                    if (greedy_weight + instance->weights[p.second] <= instance->capacity) {
                        greedy_value += instance->values[p.second];
                        greedy_weight += instance->weights[p.second];
                    }
                }
                
                double current_ratio = best_solution->fitness / greedy_value;
                if (current_ratio >= 0.74 && current_ratio <= 0.76) {
                    potential_trap = true;
                }
            }
            
            // Apply different escape strategies based on stagnation level
            if (stagnation_counter >= 8 && time_remaining() > 1.5) {
                if (potential_trap) {
                    // Aggressive escape from 0.75 trap
                    vector<KnapsackSolution> escape_candidates;
                    
                    // Strategy 1: Large perturbation of best solution
                    for (double intensity : {0.2, 0.3, 0.4, 0.5}) {
                        KnapsackSolution perturbed = best_solution->perturb(rng, intensity);
                        
                        // Apply intensive local search to perturbed solution
                        bool improved = true;
                        int iter = 0;
                        while (improved && iter < 10 && time_remaining() > 0.5) {
                            improved = false;
                            iter++;
                            
                            vector<KnapsackSolution> local_neighbors = perturbed.local_search_breaking(n_b * 2, rng);
                            for (auto& neighbor : local_neighbors) {
                                if (neighbor.fitness > perturbed.fitness) {
                                    perturbed = neighbor;
                                    improved = true;
                                    break;
                                }
                            }
                        }
                        escape_candidates.push_back(perturbed);
                    }
                    
                    // Strategy 2: Build solutions using different construction methods
                    for (int construction_method = 0; construction_method < 4; construction_method++) {
                        vector<int> alt_solution(instance->n_objects, 0);
                        
                        if (construction_method == 0) {
                            // Reverse greedy: start with worst ratios
                            vector<pair<double, int>> rev_ratios;
                            for (int i = 0; i < instance->n_objects; i++) {
                                if (instance->weights[i] > 0) {
                                    double ratio = (double)instance->values[i] / instance->weights[i];
                                    rev_ratios.push_back({ratio, i});
                                }
                            }
                            sort(rev_ratios.begin(), rev_ratios.end()); // Ascending order
                            
                            long long current_weight = 0;
                            for (auto& p : rev_ratios) {
                                if (current_weight + instance->weights[p.second] <= instance->capacity) {
                                    alt_solution[p.second] = 1;
                                    current_weight += instance->weights[p.second];
                                }
                            }
                        } else if (construction_method == 1) {
                            // Random construction with bias
                            uniform_real_distribution<double> prob(0.0, 1.0);
                            vector<int> items(instance->n_objects);
                            iota(items.begin(), items.end(), 0);
                            shuffle(items.begin(), items.end(), rng);
                            
                            long long current_weight = 0;
                            for (int item : items) {
                                if (current_weight + instance->weights[item] <= instance->capacity && prob(rng) > 0.3) {
                                    alt_solution[item] = 1;
                                    current_weight += instance->weights[item];
                                }
                            }
                        } else {
                            // Alternative greedy methods
                            continue;
                        }
                        
                        KnapsackSolution alt_sol(instance, &alt_solution);
                        if (alt_sol.is_feasible()) {
                            escape_candidates.push_back(alt_sol);
                        }
                    }
                    
                    // Find best escape candidate
                    if (!escape_candidates.empty()) {
                        auto best_escape = max_element(escape_candidates.begin(), escape_candidates.end(),
                            [](const KnapsackSolution& a, const KnapsackSolution& b) {
                                return a.fitness < b.fitness;
                            });
                        
                        if (best_escape->fitness > best_solution->fitness) {
                            delete best_solution;
                            best_solution = new KnapsackSolution(best_escape->copy());
                            stagnation_counter = 0;
                        }
                    }
                }
                
                // Regular diversification
                if (stagnation_counter >= 8) {
                    // Replace older solutions with new diverse ones
                    sort(population.begin(), population.end(), 
                         [](const KnapsackSolution& a, const KnapsackSolution& b) {
                             return a.age > b.age; // Sort by age descending
                         });
                    
                    int replace_count = min(population.size() / 2, (size_t)3);
                    for (int i = 0; i < replace_count && i < (int)population.size(); i++) {
                        population[i] = KnapsackSolution(instance);
                    }
                    stagnation_counter = max(0, stagnation_counter - 5);
                }
            }
            
            // Intensified local search when stagnated
            if (stagnation_counter >= 6 && time_remaining() > 1.0) {
                vector<KnapsackSolution> intensive_neighbors = best_solution->local_search_breaking(n_b * 3, rng);
                for (auto& neighbor : intensive_neighbors) {
                    if (neighbor.fitness > best_solution->fitness) {
                        delete best_solution;
                        best_solution = new KnapsackSolution(neighbor.copy());
                        stagnation_counter = 0;
                        break;
                    }
                }
            }
            
            // Update population size (S10)
            update_population_size();
            
            generation++;
            
            // Early stopping if population becomes too small
            if (population.size() < 2) break;
        }
        
        // Final optimization: Apply greedy post-processing to best solution
        if (time_remaining() > 0.1) {
            KnapsackSolution final_solution = *best_solution;
            
            // Try to improve by local moves
            bool improved = true;
            int post_iterations = 0;
            while (improved && post_iterations < 20 && time_remaining() > 0.05) {
                improved = false;
                post_iterations++;
                
                // Try all possible single item swaps
                for (int i = 0; i < instance->n_objects && !improved; i++) {
                    for (int j = i + 1; j < instance->n_objects && !improved; j++) {
                        if (final_solution.solution[i] != final_solution.solution[j]) {
                            vector<int> test_solution = final_solution.solution;
                            test_solution[i] = 1 - test_solution[i];
                            test_solution[j] = 1 - test_solution[j];
                            
                            KnapsackSolution test_sol(instance, &test_solution);
                            if (test_sol.is_feasible() && test_sol.fitness > final_solution.fitness) {
                                final_solution = test_sol;
                                improved = true;
                            }
                        }
                    }
                }
                
                // Try single bit flips
                for (int i = 0; i < instance->n_objects && !improved; i++) {
                    vector<int> test_solution = final_solution.solution;
                    test_solution[i] = 1 - test_solution[i];
                    
                    KnapsackSolution test_sol(instance, &test_solution);
                    if (test_sol.is_feasible() && test_sol.fitness > final_solution.fitness) {
                        final_solution = test_sol;
                        improved = true;
                    }
                }
            }
            
            if (final_solution.fitness > best_solution->fitness) {
                delete best_solution;
                best_solution = new KnapsackSolution(final_solution);
            }
        }
        
        return *best_solution;
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    try {
        // Load instance from stdin
        SingleKnapsackInstance instance;
        
        // Adjust algorithm parameters based on problem size
        int np_max, lambda_max;
        if (instance.n_objects <= 20) {
            np_max = 30;
            lambda_max = 8;
        } else if (instance.n_objects <= 50) {
            np_max = 40;
            lambda_max = 12;
        } else if (instance.n_objects <= 100) {
            np_max = 50;
            lambda_max = 15;
        } else {
            np_max = 60;
            lambda_max = 20;
        }
        
        // Run WWO algorithm with 5-second time limit
        WWOSingleKnapsack wwo(&instance, np_max, max(8, np_max / 4), lambda_max, 1, 5, 5.0);
        
        KnapsackSolution best_solution = wwo.run();
        
        // Get selected items (1-based indexing)
        vector<int> selected_items = best_solution.get_selected_items();
        
        // Output results
        cout << selected_items.size() << "\n";
        if (!selected_items.empty()) {
            for (int i = 0; i < (int)selected_items.size(); i++) {
                if (i > 0) cout << " ";
                cout << selected_items[i];
            }
            cout << "\n";
        }
        
    } catch (const exception& e) {
        // Fallback: output empty solution
        cout << "0\n";
    }
    
    return 0;
}
