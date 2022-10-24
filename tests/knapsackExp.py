import os
import textwrap
import random
import json
import time
import multiprocessing as mp
from os.path import join
import grape
import algorithms
from EvalTree import EvalKnapSackTree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import creator, base, tools
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Genrate a random knapsack
# def gen_knapsack(n_items, value_range, weight_range, out_file):
#     items = {i: {"value": random.randint(
#         *value_range), "weight": random.randint(*weight_range)} for i in range(1, n_items+1)}

#     with open(out_file, 'w') as file:
#         json.dump(items, file)

#     print(f"Data written in file at path: {join(os.getcwd(), out_file)}")

# gen_knapsack(n_items=100, value_range=(1, 50), weight_range=(1, 100), out_file='knapsackWeights.json')

# load saved weights
meta_file_path = './knapsackWeights.json'
with open(meta_file_path) as file:
    items = json.load(file)['meta']

# fitness and helper method


def get_sack_meta(phenotype):
    value, weight = 0, 0
    for i, bit in enumerate(phenotype):
        value += int(bit) * items[f'{i+1}'].get('value')
        weight += int(bit) * items[f'{i+1}'].get('weight')
    return value, weight


# weight_threshold = int(input("Input weight threshold for knapsack: "))
weight_threshold = 1100


def fitness_eval(individual, points=None):
    if individual.invalid == True:
        return np.NaN,
    if individual.phenotype is None:
        return np.NaN,
    value, weight = get_sack_meta(individual.phenotype)
    if weight > weight_threshold:
        return np.NaN,
    else:
        return -value,

# solution using DP


# def knapSack(W_thres, weights, values, n_items):
#     K = [[0 for x in range(W_thres + 1)] for x in range(n_items + 1)]

#     # Build table K[][] in bottom up manner
#     for i in range(n_items + 1):
#         for w in range(W_thres + 1):
#             if i == 0 or w == 0:
#                 K[i][w] = 0
#             elif weights[i-1] <= w:
#                 K[i][w] = max(values[i-1]
#                               + K[i-1][w-weights[i-1]],
#                               K[i-1][w])
#             else:
#                 K[i][w] = K[i-1][w]

#     return K[n_items][W_thres]


# # Driver code
# values = [item['value'] for item in items]
# weights = [item['weight'] for item in items]
# W_thres = weight_threshold
# n_items = len(values)
# print(knapSack(W_thres, weights, values, n_items))

# Define toolbox objects for running experiments


toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)

eval_tree = EvalKnapSackTree(
    meta_file_path=meta_file_path, w_threshold=weight_threshold)

creator.create('Individual', grape.KnapsackIndividual,
               fitness=creator.FitnessMin)

pop_init_method = getattr(grape, input(
    "Enter Population Initialization method: "))
toolbox.register("populationCreator",
                 pop_init_method, creator.Individual)
# toolbox.register("populationCreator",
#  grape.sensible_initialisation_, creator.Individual)
# toolbox.register("populationCreator",
#                  grape.PI_Grow_knapsack_AG, creator.Individual)


toolbox.register("evaluate", fitness_eval)

# Tournament selection:
toolbox.register("select", tools.selTournament, tournsize=3)

# Single-point crossover:
# toolbox.register("mate", grape.crossover_onepoint)
toolbox.register("mate", grape.crossover_onepoint_knapsack)

# Flip-int mutation:
# toolbox.register("mutate", grape.mutation_int_flip_per_codon)
toolbox.register("mutate", grape.mutation_int_flip_per_codon_knapsack)

# Grammar and hyperparameters

GRAMMAR_FILE = 'knapsack.bnf'
BNF_GRAMMAR = grape.Grammar(
    join("./grammars", GRAMMAR_FILE))

N_RUNS = 1
POPULATION_SIZE = 100
MAX_GENERATIONS = 150
P_CROSSOVER = 0.9
P_MUTATION = 0.01
ELITE_SIZE = round(0.01*POPULATION_SIZE)

MAX_INIT_TREE_DEPTH = 27
MIN_INIT_TREE_DEPTH = 23
MAX_TREE_DEPTH = 31
MAX_WRAPS = 0
CODON_SIZE = 255

maxListFitness = []
avgListFitness = []
minListFitness = []
stdListFitness = []

maxListSize = []
avgListSize = []
minListSize = []
stdListSize = []

# Validate initialized population

# population = toolbox.populationCreator(pop_size=POPULATION_SIZE,
#                                        w_threshold=weight_threshold,
#                                        meta_file_path=meta_file_path,
#                                        bnf_grammar=BNF_GRAMMAR,
#                                        min_init_depth=MIN_INIT_TREE_DEPTH,
#                                        max_init_depth=MAX_INIT_TREE_DEPTH,
#                                        codon_size=CODON_SIZE)

# for ind in population:
#     value, weight = get_sack_meta(ind.phenotype)
#     if weight <= weight_threshold:
#         print(f'Individual: {ind.phenotype}, Value: {value}, Weight: {weight}')
#     else:
#         print(f'Individual: {ind.phenotype}, Value: {value}, Weight: {weight} !!!!! warning')


n_workers = mp.cpu_count()


def run_experiment():
    population = toolbox.populationCreator(pop_size=POPULATION_SIZE,
                                           bnf_grammar=BNF_GRAMMAR,
                                           min_init_depth=MIN_INIT_TREE_DEPTH,
                                           max_init_depth=MAX_INIT_TREE_DEPTH,
                                           codon_size=CODON_SIZE,
                                           eval_tree=eval_tree)
    # define the hall-of-fame object:
    hof = tools.HallOfFame(ELITE_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    # perform the Grammatical Evolution flow:
    population, logbook = algorithms.ge_eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                            ngen=MAX_GENERATIONS, elite_size=ELITE_SIZE,
                                                            bnf_grammar=BNF_GRAMMAR, codon_size=CODON_SIZE,
                                                            max_tree_depth=MAX_TREE_DEPTH,
                                                            points_train=None,
                                                            stats=stats, halloffame=hof, verbose=True,
                                                            type='knapsack', eval_tree=eval_tree)

    max_fitness_values, mean_fitness_values = logbook.select("max", "avg")
    min_fitness_values, std_fitness_values = logbook.select("min", "std")

    exp_meta = {
        'mean_fitness_values': mean_fitness_values,
        'std_fitness_values': std_fitness_values,
        'min_fitness_values': min_fitness_values,
        'max_fitness_values': max_fitness_values,
    }
    return exp_meta


def plot_exp_results(avgListFitness, stdListFitness, minListFitness, maxListFitness):
    x = np.arange(MAX_GENERATIONS+1)
    avgArray = np.array(avgListFitness)
    stdArray = np.array(stdListFitness)
    minArray = np.array(minListFitness)
    maxArray = np.array(maxListFitness)
    print(len(x), len(avgArray))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(
        f'Max, Min and Average Fitness for Knapsack with {pop_init_method.__name__}')
    plt.errorbar(x, -avgArray.mean(0), yerr=-stdArray.mean(0),
                 label="Average", color="Blue")
    plt.errorbar(x, -minArray.mean(0), yerr=-
                 minArray.std(0), label="Best", color="Green")
    plt.errorbar(x, -maxArray.mean(0), yerr=-
                 maxArray.std(0), label="Worst", color="Red")
    plt.savefig(f"{pop_init_method.__name__}.png")
    print(f"Best Fitness Achieved: {np.max(-minArray)}")


if __name__ == '__main__':
    # pool = mp.Pool()
    start_time = time.perf_counter()
    # processes = [pool.apply_async(run_experiment) for i in range(N_RUNS)]
    processes = [run_experiment() for i in tqdm(range(N_RUNS))]
    # exp_metas = [p.get() for p in processes]

    # for meta in exp_metas:
    for meta in processes:
        avgListFitness.append(meta['mean_fitness_values'])
        stdListFitness.append(meta['std_fitness_values'])
        minListFitness.append(meta['min_fitness_values'])
        maxListFitness.append(meta['max_fitness_values'])

    plot_exp_results(avgListFitness, stdListFitness,
                     minListFitness, maxListFitness)

    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
