#include <bits/stdc++.h>
using namespace std;
using namespace std::chrono;

// Forward declarations for optional-like functionality
template<typename T>
struct Optional {
    bool has_value;
    T value;
    
    Optional() : has_value(false) {}
    Optional(const T& val) : has_value(true), value(val) {}
    
    bool has_value_() const { return has_value; }
    T& operator*() { return value; }
    const T& operator*() const { return value; }
    T* operator->() { return &value; }
    const T* operator->() const { return &value; }
};

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
    int age;
    
    KnapsackSolution(const SingleKnapsackInstance* inst, const vector<int>* sol = nullptr) 
        : instance(inst), wavelength(0.5), age(0) {
        if (sol == nullptr) {
            solution = generate_smart_solution();
        } else {
            solution = *sol;
        }
        fitness = calculate_fitness();
    }
    
    // Enhanced solution generation with multiple strategies
    vector<int> generate_smart_solution() {
        static mt19937 gen(chrono::steady_clock::now().time_since_epoch().count());
        uniform_real_distribution<double> dis(0.0, 1.0);
        uniform_int_distribution<int> strategy_dis(0, 4);
        
        vector<int> sol(instance->n_objects, 0);
        int strategy = strategy_dis(gen);
        
        if (strategy == 0) {
            // Ratio-based greedy with noise
            vector<pair<double, int>> ratios;
            for (int i = 0; i < instance->n_objects; i++) {
                if (instance->weights[i] > 0) {
                    double ratio = (double)instance->values[i] / instance->weights[i];
                    ratio *= (0.85 + 0.3 * dis(gen)); // Add noise for diversity
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
            // Pure greedy then improve
            sol = greedy_improve_additions(sol);
            // Add small random perturbations
            for (int i = 0; i < instance->n_objects; i++) {
                if (dis(gen) < 0.05) {
                    sol[i] = 1 - sol[i];
                }
            }
            sol = repair_and_improve(sol);
        }
        else if (strategy == 2) {
            // Value-based with randomization
            vector<pair<long long, int>> values;
            for (int i = 0; i < instance->n_objects; i++) {
                values.push_back({instance->values[i], i});
            }
            sort(values.begin(), values.end(), greater<pair<long long, int>>());
            
            long long current_weight = 0;
            for (auto& p : values) {
                int item_idx = p.second;
                if (current_weight + instance->weights[item_idx] <= instance->capacity && dis(gen) > 0.15) {
                    sol[item_idx] = 1;
                    current_weight += instance->weights[item_idx];
                }
            }
        }
        else if (strategy == 3) {
            // Random selection with bias
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
                double prob = 1.0 / (1.0 + i * 0.15);
                if (current_weight + instance->weights[item_idx] <= instance->capacity && dis(gen) < prob) {
                    sol[item_idx] = 1;
                    current_weight += instance->weights[item_idx];
                }
            }
        }
        else {
            // Completely random then repair
            for (int i = 0; i < instance->n_objects; i++) {
                sol[i] = (dis(gen) < 0.5) ? 1 : 0;
            }
            sol = repair_and_improve(sol);
        }
        
        return sol;
    }
    
    // Greedy improvement from empty solution
    vector<int> greedy_improve_additions(vector<int> x) {
        vector<double> loads(1, 0.0);
        for (int i = 0; i < instance->n_objects; i++) {
            if (x[i]) loads[0] += instance->weights[i];
        }
        
        vector<int> order(instance->n_objects);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a, int b) {
            double ratio_a = instance->weights[a] > 0 ? (double)instance->values[a] / instance->weights[a] : 0;
            double ratio_b = instance->weights[b] > 0 ? (double)instance->values[b] / instance->weights[b] : 0;
            return ratio_a > ratio_b;
        });
        
        for (int i : order) {
            if (x[i] == 0) {
                if (loads[0] + instance->weights[i] <= instance->capacity) {
                    x[i] = 1;
                    loads[0] += instance->weights[i];
                }
            }
        }
        return x;
    }
    
    // Advanced repair with intelligent item removal
    vector<int> repair_to_feasible(vector<int> x) {
        long long total_weight = get_weight_from_solution(x);
        if (total_weight <= instance->capacity) return x;
        
        vector<int> selected_items;
        for (int i = 0; i < instance->n_objects; i++) {
            if (x[i]) selected_items.push_back(i);
        }
        
        // Sort by ratio (ascending - worst first)
        sort(selected_items.begin(), selected_items.end(), [&](int a, int b) {
            double ratio_a = instance->weights[a] > 0 ? (double)instance->values[a] / instance->weights[a] : 0;
            double ratio_b = instance->weights[b] > 0 ? (double)instance->values[b] / instance->weights[b] : 0;
            return ratio_a < ratio_b;
        });
        
        for (int item : selected_items) {
            if (total_weight <= instance->capacity) break;
            x[item] = 0;
            total_weight -= instance->weights[item];
        }
        
        return x;
    }
    
    vector<int> repair_and_improve(vector<int> x) {
        x = repair_to_feasible(x);
        x = greedy_improve_additions(x);
        return x;
    }
    
    long long get_weight_from_solution(const vector<int>& sol) {
        long long total_weight = 0;
        for (int i = 0; i < instance->n_objects; i++) {
            total_weight += sol[i] * instance->weights[i];
        }
        return total_weight;
    }
    
    double calculate_fitness() {
        long long total_value = 0;
        long long total_weight = 0;
        
        for (int i = 0; i < instance->n_objects; i++) {
            total_value += solution[i] * instance->values[i];
            total_weight += solution[i] * instance->weights[i];
        }
        
        // Heavy penalty for constraint violation
        if (total_weight > instance->capacity) {
            // Calculate average profit for penalty scaling
            double avg_profit = 0;
            for (int i = 0; i < instance->n_objects; i++) {
                avg_profit += instance->values[i];
            }
            avg_profit /= instance->n_objects;
            
            long long violation = total_weight - instance->capacity;
            return (double)total_value - avg_profit * violation;
        }
        
        return (double)total_value;
    }
    
    bool is_feasible() {
        return get_weight_from_solution(solution) <= instance->capacity;
    }
    
    long long get_weight() {
        return get_weight_from_solution(solution);
    }
    
    KnapsackSolution copy() {
        KnapsackSolution copied(instance, &solution);
        copied.wavelength = wavelength;
        copied.age = age;
        return copied;
    }
    
    // Enhanced propagation with intelligent bit selection
    KnapsackSolution propagate(mt19937& rng) {
        KnapsackSolution new_solution = copy();
        new_solution.wavelength = wavelength;
        
        uniform_int_distribution<int> changes_dist(1, max(1, (int)wavelength));
        int k = changes_dist(rng);
        
        // Apply k random bit flips
        uniform_int_distribution<int> bit_dist(0, instance->n_objects - 1);
        for (int t = 0; t < k; t++) {
            int i = bit_dist(rng);
            new_solution.solution[i] = 1 - new_solution.solution[i];
        }
        
        // Repair and improve
        new_solution.solution = repair_and_improve(new_solution.solution);
        new_solution.fitness = new_solution.calculate_fitness();
        new_solution.age = 0;
        
        return new_solution;
    }
    
    // Adaptive breaking operators with learning
    vector<KnapsackSolution> adaptive_breaking(int n_b, mt19937& rng, 
                                               unordered_map<string, int>& use_count,
                                               unordered_map<string, int>& success_count,
                                               int learning_period) {
        vector<KnapsackSolution> neighbors;
        vector<string> ops = {"replace_low_profit", "flip_improve", "swap_improve"};
        
        // Select operation adaptively
        string selected_op;
        int total_use = 0;
        for (auto& kv : use_count) total_use += kv.second;
        
        if (total_use < learning_period) {
            uniform_int_distribution<int> op_dist(0, ops.size() - 1);
            selected_op = ops[op_dist(rng)];
        } else {
            // Use success rates to select operation
            vector<pair<double, string>> scores;
            for (auto& op : ops) {
                double u = use_count[op], s = success_count[op];
                double score = (s + 1.0) / (u + 1.0);
                scores.push_back({score, op});
            }
            
            double total_score = 0;
            for (auto& pr : scores) total_score += pr.first;
            
            uniform_real_distribution<double> prob_dist(0.0, 1.0);
            double r = prob_dist(rng) * total_score;
            double acc = 0;
            selected_op = ops.back();
            for (auto& pr : scores) {
                acc += pr.first;
                if (r <= acc) {
                    selected_op = pr.second;
                    break;
                }
            }
        }
        
        use_count[selected_op]++;
        double base_fitness = fitness;
        bool improved = false;
        
        if (selected_op == "replace_low_profit") {
            KnapsackSolution neighbor = greedy_repair_after_remove(n_b, rng);
            if (neighbor.fitness > base_fitness) {
                neighbors.push_back(neighbor);
                improved = true;
            }
        }
        else if (selected_op == "flip_improve") {
            auto improving_flip = pick_improving_flip(rng);
            if (improving_flip.has_value_()) {
                KnapsackSolution neighbor = copy();
                neighbor.solution[*improving_flip] = 1 - neighbor.solution[*improving_flip];
                neighbor.solution = repair_and_improve(neighbor.solution);
                neighbor.fitness = neighbor.calculate_fitness();
                if (neighbor.fitness > base_fitness) {
                    neighbors.push_back(neighbor);
                    improved = true;
                }
            }
        }
        else { // swap_improve
            auto improving_swap = pick_improving_swap(rng);
            if (improving_swap.has_value_()) {
                KnapsackSolution neighbor = copy();
                neighbor.solution[improving_swap->first] = 0;
                neighbor.solution[improving_swap->second] = 1;
                neighbor.solution = repair_and_improve(neighbor.solution);
                neighbor.fitness = neighbor.calculate_fitness();
                if (neighbor.fitness > base_fitness) {
                    neighbors.push_back(neighbor);
                    improved = true;
                }
            }
        }
        
        if (improved) {
            success_count[selected_op]++;
        }
        
        return neighbors;
    }
    
    KnapsackSolution greedy_repair_after_remove(int nb_choices, mt19937& rng) {
        vector<int> selected_items;
        for (int i = 0; i < instance->n_objects; i++) {
            if (solution[i]) selected_items.push_back(i);
        }
        
        if (!selected_items.empty()) {
            // Sort by profit (ascending - worst first)
            sort(selected_items.begin(), selected_items.end(), [&](int a, int b) {
                return instance->values[a] < instance->values[b];
            });
            
            int kmax = min(nb_choices, (int)selected_items.size());
            uniform_int_distribution<int> k_dist(1, kmax);
            int k = k_dist(rng);
            
            KnapsackSolution neighbor = copy();
            neighbor.solution[selected_items[k-1]] = 0;
            neighbor.solution = repair_and_improve(neighbor.solution);
            neighbor.fitness = neighbor.calculate_fitness();
            return neighbor;
        }
        
        return copy();
    }
    
    Optional<int> pick_improving_flip(mt19937& rng) {
        double base_fitness = fitness;
        vector<int> indices(instance->n_objects);
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), rng);
        
        for (int i : indices) {
            vector<int> test_solution = solution;
            test_solution[i] = 1 - test_solution[i];
            
            KnapsackSolution test_sol(instance, &test_solution);
            if (test_sol.is_feasible() && test_sol.fitness > base_fitness) {
                return Optional<int>(i);
            }
        }
        return Optional<int>();
    }
    
    Optional<pair<int, int>> pick_improving_swap(mt19937& rng) {
        double base_fitness = fitness;
        vector<int> ones, zeros;
        
        for (int i = 0; i < instance->n_objects; i++) {
            if (solution[i]) ones.push_back(i);
            else zeros.push_back(i);
        }
        
        if (ones.empty() || zeros.empty()) return Optional<pair<int, int>>();
        
        shuffle(ones.begin(), ones.end(), rng);
        shuffle(zeros.begin(), zeros.end(), rng);
        
        for (int i : ones) {
            for (int j : zeros) {
                vector<int> test_solution = solution;
                test_solution[i] = 0;
                test_solution[j] = 1;
                
                KnapsackSolution test_sol(instance, &test_solution);
                if (test_sol.is_feasible() && test_sol.fitness > base_fitness) {
                    return Optional<pair<int, int>>(make_pair(i, j));
                }
            }
        }
        return Optional<pair<int, int>>();
    }
    
    vector<int> get_selected_items() {
        vector<int> selected;
        for (int i = 0; i < instance->n_objects; i++) {
            if (solution[i] == 1) {
                selected.push_back(i + 1);
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
    string variant;
    int learning_period;
    
    vector<KnapsackSolution> population;
    KnapsackSolution* best_solution;
    int generation;
    high_resolution_clock::time_point start_time;
    mt19937 rng;
    
    // Adaptive operator learning
    unordered_map<string, int> use_count, success_count;
    
    WWOSingleKnapsack(const SingleKnapsackInstance* inst, 
                      int np_max = -1, int np_min = 12,
                      double lambda_max_ratio = 0.75, int lambda_min = 1,
                      int n_b = 8, string variant = "adaptive",
                      int learning_period = 30, double time_limit = 3.8)
        : instance(inst), np_min(max(np_min, 4)),
          lambda_min(lambda_min), n_b(n_b), time_limit(time_limit),
          variant(variant), learning_period(learning_period),
          generation(0), best_solution(nullptr) {
        
        // Adaptive population size based on problem complexity
        if (np_max <= 0) {
            np_max = max(20, (int)(5 * log(max(2.0, (double)instance->n_objects / 2.0))));
        }
        this->np_max = np_max;
        
        lambda_max = max(lambda_min, (int)(lambda_max_ratio * instance->n_objects));
        
        // Initialize random number generator
        random_device rd;
        rng.seed(rd());
        
        // Initialize operator counters
        vector<string> ops = {"replace_low_profit", "flip_improve", "swap_improve"};
        for (auto& op : ops) {
            use_count[op] = 0;
            success_count[op] = 0;
        }
    }
    
    ~WWOSingleKnapsack() {
        delete best_solution;
    }
    
    void initialize_population() {
        population.clear();
        
        // Generate diverse initial solutions
        for (int i = 0; i < np_max; i++) {
            if (i % 2 == 0) {
                // Smart construction
                population.push_back(KnapsackSolution(instance));
            } else {
                // Start from empty and build greedily with perturbation
                vector<int> x(instance->n_objects, 0);
                KnapsackSolution temp_sol(instance, &x);
                x = temp_sol.greedy_improve_additions(x);
                
                // Add small random perturbations
                uniform_real_distribution<double> prob_dist(0.0, 1.0);
                for (int j = 0; j < instance->n_objects; j++) {
                    if (prob_dist(rng) < 0.05) {
                        x[j] = 1 - x[j];
                    }
                }
                
                x = temp_sol.repair_and_improve(x);
                population.push_back(KnapsackSolution(instance, &x));
            }
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
        
        vector<double> fvals;
        for (const auto& sol : population) {
            fvals.push_back(sol.fitness);
        }
        
        vector<int> wavelengths;
        if (variant == "adaptive") {
            wavelengths = wavelengths_exponential(fvals, lambda_min, lambda_max);
        } else {
            wavelengths = wavelengths_linear(fvals, lambda_min, lambda_max);
        }
        
        for (size_t i = 0; i < population.size(); i++) {
            population[i].wavelength = wavelengths[i];
        }
    }
    
    vector<int> wavelengths_linear(const vector<double>& fvals, int lam_min, int lam_max) {
        double eps = 1e-9;
        double fmax = *max_element(fvals.begin(), fvals.end());
        double fmin = *min_element(fvals.begin(), fvals.end());
        
        vector<int> wl;
        wl.reserve(fvals.size());
        
        for (double f : fvals) {
            double lam = lam_min + (lam_max - lam_min) * ((fmax - f + eps) / (fmax - fmin + eps));
            int L = (int)llround(lam);
            L = max(lam_min, min(lam_max, L));
            wl.push_back(L);
        }
        return wl;
    }
    
    vector<int> wavelengths_exponential(const vector<double>& fvals, int lam_min, int lam_max) {
        double eps = 1e-9;
        double fmax = *max_element(fvals.begin(), fvals.end());
        double fmin = *min_element(fvals.begin(), fvals.end());
        double base = max(1.0, (double)lam_max / max(1, lam_min));
        
        vector<int> wl;
        wl.reserve(fvals.size());
        
        for (double f : fvals) {
            double expv = (fmax - f + eps) / (fmax - fmin + eps);
            double lam = lam_min * pow(base, expv);
            int L = (int)llround(lam);
            L = max(lam_min, min(lam_max, L));
            wl.push_back(L);
        }
        return wl;
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
        while (!population.empty() && time_remaining() > 0) {
            // Calculate wavelengths
            calculate_wavelengths();
            
            // Propagate each solution
            vector<KnapsackSolution> new_population;
            for (auto& solution : population) {
                if (time_remaining() <= 0) break;
                
                // Age tracking
                solution.age++;
                
                // Propagate solution
                KnapsackSolution new_solution = solution.propagate(rng);
                
                double fx = solution.fitness;
                double fxp = new_solution.fitness;
                
                // Replace only if better
                if (fxp > fx) {
                    new_population.push_back(move(new_solution));
                    fx = fxp;
                } else {
                    new_population.push_back(solution);
                }
                
                // Update best solution
                if (fx > best_solution->fitness) {
                    delete best_solution;
                    best_solution = new KnapsackSolution(new_population.back().copy());
                }
            }
            
            population = move(new_population);
            generation++;
            
            // Population size reduction
            if (generation > 0 && (int)population.size() > np_min) {
                auto worst_it = min_element(population.begin(), population.end(),
                    [](const KnapsackSolution& a, const KnapsackSolution& b) {
                        return a.fitness < b.fitness;
                    });
                population.erase(worst_it);
            }
        }
        
        // Final intensification phase with adaptive breaking
        for (int t = 0; t < 100 && time_remaining() > 0; t++) {
            vector<KnapsackSolution> neighbors = best_solution->adaptive_breaking(
                n_b, rng, use_count, success_count, learning_period);
            
            for (auto& neighbor : neighbors) {
                if (neighbor.fitness > best_solution->fitness) {
                    delete best_solution;
                    best_solution = new KnapsackSolution(neighbor.copy());
                    break;
                }
            }
        }
        
        return *best_solution;
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    try {
        // Load instance from stdin
        SingleKnapsackInstance instance;
        
        // Run WWO algorithm with adaptive parameters
        WWOSingleKnapsack wwo(&instance, -1, 12, 0.75, 1, 8, "adaptive", 30, 4.85);
        
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
