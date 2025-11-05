#!/usr/bin/env python3
"""Update AppConfig to support GA settings"""

# Update app_config.h
with open('include/app_config.h', 'r') as f:
    header = f.read()

# Add GA settings struct before AppConfig constructor
ga_struct = '''    // Genetic Algorithm settings
    struct GeneticAlgorithmSettings {
        int population_size = 30;           // Number of genomes per generation
        int num_generations = 20;           // Number of generations to evolve
        float mutation_rate = 0.15f;        // Probability of mutating each gene
        float crossover_rate = 0.7f;        // Probability of crossover vs cloning
        int frames_per_eval = 30;           // Frames to capture per fitness evaluation

        // Parameters to optimize (0=fixed, 1=optimize)
        bool optimize_bias_diff = true;
        bool optimize_bias_refr = true;
        bool optimize_bias_fo = true;
        bool optimize_bias_hpf = true;
        bool optimize_bias_pr = true;
        bool optimize_accumulation = true;
        bool optimize_trail_filter = false;
        bool optimize_antiflicker = false;
        bool optimize_erc = false;
    };

'''

# Insert before "AppConfig();"
if 'struct GeneticAlgorithmSettings' not in header:
    header = header.replace('    AppConfig();', ga_struct + '    AppConfig();')

# Add accessor before private section
ga_accessor = '''    GeneticAlgorithmSettings& ga_settings() { return ga_settings_; }
    const GeneticAlgorithmSettings& ga_settings() const { return ga_settings_; }

'''

if 'ga_settings()' not in header:
    header = header.replace('private:', ga_accessor + 'private:')

# Add member variable
if 'GeneticAlgorithmSettings ga_settings_;' not in header:
    header = header.replace('    AlgorithmSettings algorithm_settings_;',
                           '    AlgorithmSettings algorithm_settings_;\n    GeneticAlgorithmSettings ga_settings_;')

with open('include/app_config.h', 'w') as f:
    f.write(header)

print("Updated app_config.h")

# Update app_config.cpp to load GA settings
with open('src/app_config.cpp', 'r') as f:
    impl = f.read()

# Find the end of the load() function and add GA loading code
ga_load_code = '''
    // Load genetic algorithm settings
    if (ini.sections.count("GeneticAlgorithm")) {
        auto& ga_section = ini.sections["GeneticAlgorithm"];

        if (ga_section.count("population_size"))
            ga_settings_.population_size = std::stoi(ga_section["population_size"]);
        if (ga_section.count("num_generations"))
            ga_settings_.num_generations = std::stoi(ga_section["num_generations"]);
        if (ga_section.count("mutation_rate"))
            ga_settings_.mutation_rate = std::stof(ga_section["mutation_rate"]);
        if (ga_section.count("crossover_rate"))
            ga_settings_.crossover_rate = std::stof(ga_section["crossover_rate"]);
        if (ga_section.count("frames_per_eval"))
            ga_settings_.frames_per_eval = std::stoi(ga_section["frames_per_eval"]);

        if (ga_section.count("optimize_bias_diff"))
            ga_settings_.optimize_bias_diff = (std::stoi(ga_section["optimize_bias_diff"]) != 0);
        if (ga_section.count("optimize_bias_refr"))
            ga_settings_.optimize_bias_refr = (std::stoi(ga_section["optimize_bias_refr"]) != 0);
        if (ga_section.count("optimize_bias_fo"))
            ga_settings_.optimize_bias_fo = (std::stoi(ga_section["optimize_bias_fo"]) != 0);
        if (ga_section.count("optimize_bias_hpf"))
            ga_settings_.optimize_bias_hpf = (std::stoi(ga_section["optimize_bias_hpf"]) != 0);
        if (ga_section.count("optimize_bias_pr"))
            ga_settings_.optimize_bias_pr = (std::stoi(ga_section["optimize_bias_pr"]) != 0);
        if (ga_section.count("optimize_accumulation"))
            ga_settings_.optimize_accumulation = (std::stoi(ga_section["optimize_accumulation"]) != 0);
        if (ga_section.count("optimize_trail_filter"))
            ga_settings_.optimize_trail_filter = (std::stoi(ga_section["optimize_trail_filter"]) != 0);
        if (ga_section.count("optimize_antiflicker"))
            ga_settings_.optimize_antiflicker = (std::stoi(ga_section["optimize_antiflicker"]) != 0);
        if (ga_section.count("optimize_erc"))
            ga_settings_.optimize_erc = (std::stoi(ga_section["optimize_erc"]) != 0);

        std::cout << "  GA population size: " << ga_settings_.population_size << std::endl;
        std::cout << "  GA generations: " << ga_settings_.num_generations << std::endl;
    }
'''

if 'Load genetic algorithm settings' not in impl:
    # Find the return true statement in load() function
    impl = impl.replace('    return true;\n}', ga_load_code + '\n    return true;\n}')

with open('src/app_config.cpp', 'w') as f:
    f.write(impl)

print("Updated app_config.cpp")
