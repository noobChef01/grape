from ponyge2_adapted_files import Grammar, ge
import algorithms
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import creator, base, tools
from sklearn.model_selection import train_test_split
import math
import textwrap
import csv
import random
import json
from parser_files_knap.parser import weight_parser, value_parser

import warnings
warnings.filterwarnings("ignore")

# generate weights for knapsack
NBR_ITEMS = 10
items = dict()
for i in range(1, 1+NBR_ITEMS):
    items[i] = (random.randint(1, 10), random.randint(1, 100))

with open("/mnt/d/college_notes/internship/grape/knapsack_ag/data/knapsack_weights.json", "w") as file:
    json.dump(items, file)


GRAMMAR_FILE = 'knapsack_new.bnf'
BNF_GRAMMAR = Grammar(
    path.join("/mnt/d/college_notes/internship/grape/grammars", GRAMMAR_FILE))


def fitness_eval(individual, points=None):
    if individual.phenotype is None:
        return np.NaN,
    total_weight = weight_parser(individual.phenotype)
    if total_weight > 200:
        return np.NaN,
    else:
        fitness = -value_parser(individual.phenotype)
        return fitness,


toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))


# creator.create('Individual', ge.Individual, fitness=creator.FitnessMax)
creator.create('Individual', ge.Individual, fitness=creator.FitnessMin)

toolbox.register("populationCreator",
                 ge.initialisation_PI_Grow, creator.Individual)

toolbox.register("evaluate", fitness_eval)  # , points=[X_train, Y_train])

# Tournament selection:
toolbox.register("select", ge.selTournament, tournsize=3)

# Single-point crossover:
toolbox.register("mate", ge.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", ge.mutation_int_flip_per_codon)


N_RUNS = 1
POPULATION_SIZE = 200
MAX_GENERATIONS = 150
P_CROSSOVER = 0.9
P_MUTATION = 0.01
ELITE_SIZE = round(0.01*POPULATION_SIZE)

HALL_OF_FAME_SIZE = 1
MAX_INIT_TREE_DEPTH = 7
MIN_INIT_TREE_DEPTH = 3
MAX_TREE_DEPTH = 8
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

for r in range(0, N_RUNS):
    # create initial population (generation 0):
    population = toolbox.populationCreator(size=POPULATION_SIZE,
                                           bnf_grammar=BNF_GRAMMAR,
                                           min_init_tree_depth=MIN_INIT_TREE_DEPTH,
                                           max_init_tree_depth=MAX_INIT_TREE_DEPTH,
                                           max_tree_depth=MAX_TREE_DEPTH,
                                           max_wraps=MAX_WRAPS,
                                           codon_size=CODON_SIZE
                                           )
    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    # prepare the statistics object:
    # stats = tools.Statistics(key=lambda ind: ind.fitness.values if math.isnan(ind.fitness.values[0]) else None)#ind.fitness.values != np.inf else None)
    # stats = tools.Statistics(key=lambda ind: ind.fitness.values[0] if not math.isnan(ind.fitness.values[0]) else np.NaN)#ind.fitness.values != np.inf else None)
    # if not ind.invalid else (np.NaN,))#ind.fitness.values != np.inf else None)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    # Which run are we on?
    print("\n\nCurrently on run", r, "of", N_RUNS)
    # perform the Grammatical Evolution flow:
    population, logbook = algorithms.ge_eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                            ngen=MAX_GENERATIONS, elite_size=ELITE_SIZE,
                                                            bnf_grammar=BNF_GRAMMAR, codon_size=CODON_SIZE,
                                                            max_tree_depth=MAX_TREE_DEPTH, max_wraps=MAX_WRAPS,
                                                            points_train=[
                                                                None, None],
                                                            points_test=[
                                                                None, None],
                                                            stats=stats, halloffame=hof, verbose=True)

    max_fitness_values, mean_fitness_values = logbook.select("max", "avg")
    min_fitness_values, std_fitness_values = logbook.select("min", "std")

    # fitness_test = logbook.select("fitness_test")
    #best_ind_length = logbook.select("best_ind_length")
    # avg_length = logbook.select("avg_length")
    # max_length = logbook.select("max_length")
    # selection_time = logbook.select("selection_time")
    # generation_time = logbook.select("generation_time")
    # gen, invalid = logbook.select("gen", "invalid")ssssss

    # Save statistics for this run:
    avgListFitness.append(mean_fitness_values)
    stdListFitness.append(std_fitness_values)
    minListFitness.append(min_fitness_values)
    maxListFitness.append(max_fitness_values)

    # avgListSize.append(meanSizeValues)
    # stdListSize.append(stdSizeValues)
    # minListSize.append(minSizeValues)
    # maxListSize.append(maxSizeValues)

    # best = hof.items[0].phenotype # parser to change the individual
    best = hof.items[0].phenotype  # parser to change the individual
    print("Best individual: \n", "\n".join(textwrap.wrap(best, 80)))
    print("\nTraining Fitness: ", hof.items[0].fitness.values[0])
    print("Depth: ", hof.items[0].depth)
    print("Length of the genome: ", len(hof.items[0].genome))
    print(
        f'Used portion of the genome: {hof.items[0].used_codons/len(hof.items[0].genome):.2f}')
