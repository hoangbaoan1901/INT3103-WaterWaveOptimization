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
    double previous_fitness;  // For E7 calculation
    
    KnapsackSolution(const SingleKnapsackInstance* inst, const vector<int>* sol = nullptr) 
        : instance(inst), wavelength(0.5), previous_fitness(0.0) {
        if (sol == nullptr) {
            solution = generate_random_solution();
        } else {
            solution = *sol;
        }
        fitness = calculate_fitness();
        previous_fitness = fitness;
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
        copied.previous_fitness = previous_fitness;
        return copied;
    }
    
    KnapsackSolution propagate(mt19937& rng) {
        KnapsackSolution new_solution = copy();
        new_solution.wavelength = wavelength;
        new_solution.previous_fitness = fitness;
        
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
        
        uniform_int_distribution<int> strategy_dist(0, 2);
        uniform_int_distribution<int> item_dist(0, instance->n_objects - 1);
        
        for (int k = 1; k <= min(n_b, instance->n_objects); k++) {
            KnapsackSolution neighbor = copy();
            
            int strategy = k % 3;
            
            if (strategy == 0) {
                // Remove k-th lowest value item and try to add highest value items
                vector<pair<long long, int>> selected_items;
                for (int i = 0; i < instance->n_objects; i++) {
                    if (neighbor.solution[i] == 1) {
                        selected_items.push_back({instance->values[i], i});
                    }
                }
                
                if (!selected_items.empty() && k <= (int)selected_items.size()) {
                    sort(selected_items.begin(), selected_items.end());
                    int item_to_remove = selected_items[k-1].second;
                    neighbor.solution[item_to_remove] = 0;
                    
                    // Try to add highest value items that fit
                    vector<pair<long long, int>> unselected_items;
                    for (int i = 0; i < instance->n_objects; i++) {
                        if (neighbor.solution[i] == 0) {
                            unselected_items.push_back({instance->values[i], i});
                        }
                    }
                    
                    sort(unselected_items.begin(), unselected_items.end(), greater<pair<long long, int>>());
                    
                    for (auto& p : unselected_items) {
                        int item = p.second;
                        vector<int> temp_solution = neighbor.solution;
                        temp_solution[item] = 1;
                        
                        long long temp_weight = 0;
                        for (int i = 0; i < instance->n_objects; i++) {
                            temp_weight += temp_solution[i] * instance->weights[i];
                        }
                        
                        if (temp_weight <= instance->capacity) {
                            neighbor.solution[item] = 1;
                            break;
                        }
                    }
                }
            }
            else if (strategy == 1) {
                // Add highest value items that fit
                vector<int> unselected_items;
                for (int i = 0; i < instance->n_objects; i++) {
                    if (neighbor.solution[i] == 0) {
                        unselected_items.push_back(i);
                    }
                }
                
                if (!unselected_items.empty()) {
                    vector<pair<long long, int>> value_items;
                    for (int item : unselected_items) {
                        value_items.push_back({instance->values[item], item});
                    }
                    
                    sort(value_items.begin(), value_items.end(), greater<pair<long long, int>>());
                    
                    for (int i = 0; i < min(k, (int)value_items.size()); i++) {
                        int item = value_items[i].second;
                        vector<int> temp_solution = neighbor.solution;
                        temp_solution[item] = 1;
                        
                        long long temp_weight = 0;
                        for (int j = 0; j < instance->n_objects; j++) {
                            temp_weight += temp_solution[j] * instance->weights[j];
                        }
                        
                        if (temp_weight <= instance->capacity) {
                            neighbor.solution[item] = 1;
                            break;
                        }
                    }
                }
            }
            else {
                // Swap items based on value improvement
                vector<int> selected, unselected;
                for (int i = 0; i < instance->n_objects; i++) {
                    if (neighbor.solution[i] == 1) {
                        selected.push_back(i);
                    } else {
                        unselected.push_back(i);
                    }
                }
                
                if (!selected.empty() && !unselected.empty()) {
                    long long best_improvement = 0;
                    int best_sel = -1, best_unsel = -1;
                    
                    int max_check = min(5, (int)selected.size());
                    for (int i = 0; i < max_check; i++) {
                        int sel_item = selected[i];
                        int max_check_unsel = min(5, (int)unselected.size());
                        for (int j = 0; j < max_check_unsel; j++) {
                            int unsel_item = unselected[j];
                            
                            vector<int> temp_solution = neighbor.solution;
                            temp_solution[sel_item] = 0;
                            temp_solution[unsel_item] = 1;
                            
                            long long temp_weight = 0;
                            for (int l = 0; l < instance->n_objects; l++) {
                                temp_weight += temp_solution[l] * instance->weights[l];
                            }
                            
                            if (temp_weight <= instance->capacity) {
                                long long improvement = instance->values[unsel_item] - instance->values[sel_item];
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
            }
            
            neighbor.fitness = neighbor.calculate_fitness();
            if (!neighbor.is_feasible()) {
                neighbor.repair_solution();
            }
            
            neighbors.push_back(neighbor);
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
        
        // Generate diverse initial solutions
        for (int i = 0; i < np_max; i++) {
            KnapsackSolution sol(instance);
            
            // Apply local improvement to some initial solutions
            if (i < np_max / 3) {
                // Apply simple local search to improve quality
                bool improved = true;
                int iterations = 0;
                while (improved && iterations < 10) {
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
                }
            }
            
            population.push_back(sol);
        }
        
        // Find initial best solution
        auto best_it = max_element(population.begin(), population.end(),
            [](const KnapsackSolution& a, const KnapsackSolution& b) {
                return a.fitness < b.fitness;
            });
        
        delete best_solution;
        best_solution = new KnapsackSolution(best_it->copy());
    }
    
    void update_wavelength_for_solution(KnapsackSolution& solution, double f_max) {
        // E7: λ_x' = min(λ_x + α * (f(x) - f(x')) / f_max, λ_max)
        double alpha = 0.2;  // Increased control parameter for more responsiveness
        
        if (f_max > 0) {
            double delta_fitness = solution.previous_fitness - solution.fitness;
            double adjustment = alpha * delta_fitness / f_max;
            solution.wavelength = min(solution.wavelength + adjustment, (double)lambda_max);
            solution.wavelength = max((double)lambda_min, solution.wavelength);
        }
        
        // Update previous fitness for next iteration
        solution.previous_fitness = solution.fitness;
    }
    
    void calculate_wavelengths() {
        if (population.empty()) return;
        
        // Find f_max in population
        double f_max = -1e9;
        for (const auto& sol : population) {
            f_max = max(f_max, sol.fitness);
        }
        
        // Update wavelengths using E7 formula
        for (auto& solution : population) {
            update_wavelength_for_solution(solution, f_max);
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
        
        // Main optimization loop
        while (time_remaining() > 0) {
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
                    }
                } else {
                    new_population.push_back(solution);
                }
            }
            
            population = new_population;
            
            // Calculate wavelengths AFTER propagation (E7 needs previous and current fitness)
            calculate_wavelengths();
            
            // Update population size (S10)
            update_population_size();
            
            generation++;
            
            // Early stopping if population becomes too small
            if (population.size() < 2) break;
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
