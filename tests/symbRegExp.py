import textwrap
import grape
import algorithms
from EvalTree import EvalSymRegTree

import json
from os import path
import pandas as pd
import numpy as np
from deap import creator, base, tools
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")


diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    diabetes_X, diabetes_y, test_size=0.25, shuffle=True)

pca = PCA(n_components=2)
pca.fit(X_train)

X_train_2D = pca.transform(X_train)
X_test_2D = pca.transform(X_test)

# X_train_2D = X_train_2D.flatten()
# X_test_2D = X_test_2D.flatten()


GRAMMAR_FILE = 'simpleReg.bnf'
BNF_GRAMMAR = grape.Grammar(path.join("./grammars", GRAMMAR_FILE))


def replace_operators(phenotype):
    return phenotype.replace('DIVIDE', '/').replace('MINUS', '-').replace('PLUS', '+').replace('MULTIPLY', '*')


def fitness_eval(individual, points):
    #points = [X, Y]
    x = points[0]
    y = points[1]

    if individual.invalid == True:
        return np.NaN,
    try:
        pred = eval(replace_operators(individual.phenotype))
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        # except Exception as e:
        #     write_stat(e, "out-files/statistics.txt", "a")
        return np.NaN,
    assert np.isrealobj(pred)

    try:
        # fitness = 1/np.mean(np.square(y - pred))
        fitness = np.mean(np.square(y - pred))
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        # except Exception as e:
        #     write_stat(e, "out-files/statistics.txt", "a")
        fitness = np.NaN

    if fitness == float("inf"):
        return np.NaN,
    return fitness,


toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)
creator.create('Individual', grape.RegIndividual, fitness=creator.FitnessMin)


toolbox.register("populationCreator",
                 grape.sensible_initialisation_reg_AG, creator.Individual)

# toolbox.register("populationCreator",
#                  grape.PI_Grow_reg_AG, creator.Individual)

# toolbox.register("populationCreator", grape.PI_Grow, creator.Individual)


toolbox.register("evaluate", fitness_eval)

# Tournament selection:
toolbox.register("select", tools.selTournament, tournsize=3)

# Single-point crossover:
# toolbox.register("mate", grape.crossover_onepoint)
toolbox.register("mate", grape.crossover_onepoint_reg)


# Flip-int mutation:
# toolbox.register("mutate", grape.mutation_int_flip_per_codon)
toolbox.register("mutate", grape.mutation_int_flip_per_codon_reg)


N_RUNS = 1
POPULATION_SIZE = 100
MAX_GENERATIONS = 100
P_CROSSOVER = 0.9
P_MUTATION = 0.01
ELITE_SIZE = round(0.01*POPULATION_SIZE)

HALL_OF_FAME_SIZE = 1
MAX_INIT_TREE_DEPTH = 9
MIN_INIT_TREE_DEPTH = 5
MAX_TREE_DEPTH = 17
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
bestListSize = []

meta_file_path = './simpleReg.json'
eval_tree = EvalSymRegTree(meta_file_path, bnf_grammar=BNF_GRAMMAR)

for r in range(0, N_RUNS):
    # create initial population (generation 0):
    population = toolbox.populationCreator(pop_size=POPULATION_SIZE,
                                           bnf_grammar=BNF_GRAMMAR,
                                           min_init_depth=MIN_INIT_TREE_DEPTH,
                                           max_init_depth=MAX_INIT_TREE_DEPTH,
                                           codon_size=CODON_SIZE,
                                           eval_tree=eval_tree)

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
                                                            max_tree_depth=MAX_TREE_DEPTH,
                                                            points_train=[
                                                                X_train_2D, y_train],
                                                            points_test=[
                                                                X_test_2D, y_test],
                                                            stats=stats, halloffame=hof, verbose=True, type="regression",
                                                            eval_tree=eval_tree)

    max_fitness_values, mean_fitness_values = logbook.select("max", "avg")
    min_fitness_values, std_fitness_values = logbook.select("min", "std")

    # fitness_test = logbook.select("fitness_test")
    best_ind_length = logbook.select("best_ind_length")
    avg_length = logbook.select("avg_length")
    max_length = logbook.select("max_length")
    # selection_time = logbook.select("selection_time")
    # generation_time = logbook.select("generation_time")
    # gen, invalid = logbook.select("gen", "invalid")

    # Save statistics for this run:
    avgListFitness.append(mean_fitness_values)
    stdListFitness.append(std_fitness_values)
    minListFitness.append(min_fitness_values)
    maxListFitness.append(max_fitness_values)

    avgListSize.append(avg_length)
    bestListSize.append(best_ind_length)
    # stdListSize.append(stdSizeValues)
    # minListSize.append(minSizeValues)
    maxListSize.append(max_length)

    best = hof.items[0].phenotype  # parser to change the individual
    print("Best individual: \n", "\n".join(textwrap.wrap(best, 80)))
    print("\nTraining Fitness: ", hof.items[0].fitness.values[0])
    print("Depth: ", hof.items[0].depth)
    print("Length of the genome: ", len(hof.items[0].genome))
    print(
        f'Used portion of the genome: {hof.items[0].used_codons/len(hof.items[0].genome):.2f}')
