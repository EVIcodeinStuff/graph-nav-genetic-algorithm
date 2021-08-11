# Author: EVI #

# Imports
import logging
import operator
import uuid
from enum import Enum
from random import random, sample

import matplotlib.pyplot as mat_plt
import numpy as np
import pandas as pd

# Initialize Logger
logger = logging.getLogger(__name__)


class Node:
    """A Node within the Graph."""

    def __init__(self, x: float, y: float, label: str = str(uuid.uuid4())):
        self.label = label
        self.x = x
        self.y = y

    def calc_distance_to_other_node(self, node) -> float:
        """Given another Node, find the relative distance from this node it."""

        return Edge(self, node, is_directed=False).calc_length()

    def __str__(self):
        return f"{self.label}: ({self.x}, {self.y})"

    def __repr__(self):
        return str(self)


class Edge:
    """An Edge within a Graph. Connects two Nodes"""

    def __init__(self, a: Node, b: Node, is_directed: bool = False):
        self.a_node = a
        self.b_node = b
        self.is_directed = is_directed

        self.length = self.calc_length()

    def calc_length(self) -> float:
        """Calculate the Length of the Edge using distance formula"""

        x_delta = abs(self.a_node.x - self.b_node.x)
        y_delta = abs(self.a_node.y - self.b_node.y)

        return np.sqrt((x_delta ** 2) + (y_delta ** 2))


class Fitness:
    """Fitness algorithm for evaluating each generation's population"""

    def __init__(self, path):
        self.path = path
        self.fitness = 0.0
        self.distance = 0

    def calc_path_fitness(self) -> float:
        """ Calculate the Fitness of a Path """

        try:
            if self.fitness == 0:
                self.fitness = 1 / float(self.calc_path_distance())

        except ZeroDivisionError:
            return 0.0

        return self.fitness

    def calc_path_distance(self) -> float:
        """Walk each edge along a series of nodes and sum the total distance"""

        if self.distance == 0:
            path_distance = 0

            for i in range(0, len(self.path)):
                from_node = self.path[i]
                to_node = None

                if i + 1 < len(self.path):
                    to_node = self.path[i + 1]
                else:
                    to_node = self.path[0]

                path_distance += from_node.calc_distance_to_other_node(to_node)
            self.distance = path_distance

        return self.distance


class GeneticAlgorithm:
    """The necessary operations to perform a simple GA on path optimization"""

    def __init__(self,
                 population: list = None,
                 population_size: int = 100,
                 alpha_size: int = 5,
                 generations: int = 200,
                 mutation_rate: float = 0.02,
                 plot_results: bool = False):

        print('Genetic Algorithm for Optimizing Path / Edge Traversal \n')

        if population is None or len(population) == 0:
            logger.info("No Population was given, generating one randomly")
            population = list(GeneticAlgorithm.create_node_dict(size=population_size).values())

        try:
            print(f"Initial Pop: {population} \n")
            print(f"Generations: {generations} \n")

            if plot_results:
                self.run_with_plot(
                    population, population_size, alpha_size, generations, mutation_rate
                )

            else:
                self.optimized_final_result = self.run(
                    population, population_size, alpha_size, generations, mutation_rate
                )

                print(f"The Optimal Path: \n {self.optimized_final_result}")

        except Exception as ex:
            logger.error(ex, exc_info=True)
            print("Oops! Something has gone awry!")

        logger.info("Complete")

    @staticmethod
    def create_node_dict(size: int = 50, grid_height: int = 300, grid_width: int = 300) -> dict:
        """Randomly create a dictionary of Nodes to be used as Genesis Population for GA"""

        return {
            i: Node(x=int(random() * grid_width), y=int(random() * grid_height))
            for i in range(0, size)
        }

    def run(self,
            population: list,
            population_size: int,
            alpha_size: int,
            generations: int,
            mutation_rate: float) -> list:
        """Entry Point / Highest Level Function for running GA"""

        _population = self.initialize_population(nodes=population, population_size=population_size)

        print(f"Starting path length: {str(1 / self.rank_paths(_population)[0][1])}")

        for i in range(0, generations):
            _population = self.create_next_generation(
                current_gen=_population,
                alpha_size=alpha_size,
                mutation_rate=mutation_rate
            )

        print(f"Final Optimized Length: {str(1 / self.rank_paths(_population)[0][1])}")

        op_index = self.rank_paths(_population)[0][0]
        optimal_path = _population[op_index]

        return optimal_path

    def run_with_plot(self,
                      population: list,
                      population_size: int,
                      alpha_size: int,
                      generations: int,
                      mutation_rate: float, ) -> None:
        """Run GA to figure out optimal pathing and plot the result using matplotlib"""

        _population = self.initialize_population(nodes=population, population_size=population_size)
        progress = [1 / self.rank_paths(population=_population)[0][1]]

        # Run GA for N number of Generations
        progress += [
            1 / self.rank_paths(self.create_next_generation(_population, alpha_size, mutation_rate))[0][1]
            for i in range(0, generations)
        ]

        mat_plt.plot(progress)
        mat_plt.xlabel('Generation')
        mat_plt.ylabel('Length of Path')
        mat_plt.show()

    def initialize_population(self, nodes: list, population_size: int = 10) -> list:
        """Create First Population for GA"""

        return [self.create_path(node_list=nodes) for i in range(0, population_size)]

    @staticmethod
    def create_path(node_list: list) -> list:
        """Creates a Random Path (multiple Edges) using a series of Nodes"""

        path = sample(node_list, len(node_list))
        return path

    @staticmethod
    def selection(ranked_population: list, alpha_size: int) -> list:
        """Select Fit Candidates from Given Population"""

        selection_results = []

        df = pd.DataFrame(np.array(ranked_population), columns=["Index", "Fitness"])

        df['cumulative_sum'] = df.Fitness.cumsum()
        df['cumulative_percentage'] = 100 * df.cumulative_sum / df.Fitness.sum()

        for i in range(0, alpha_size):
            selection_results.append(ranked_population[i][0])

        for i in range(0, len(ranked_population) - alpha_size):
            pick = 100 * random()

            for i in range(0, len(ranked_population)):
                if pick <= df.iat[i, 3]:
                    selection_results.append(ranked_population[i][0])
                    break

        return selection_results

    @staticmethod
    def rank_paths(population: list) -> list:
        """Rank each path in the population based off its evaluated fitness"""

        fitness_results = {index: Fitness(population[index]).calc_path_fitness() for index in range(0, len(population))}
        return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)

    @staticmethod
    def reproduce(parent_one: list, parent_two: list) -> list:
        """Given two parents from the current population, create a child"""

        a_gene = int(random() * len(parent_one))
        b_gene = int(random() * len(parent_one))

        start_gene = min(a_gene, b_gene)
        end_gene = max(a_gene, b_gene)

        child_of_parent_one = [parent_one[index] for index in range(start_gene, end_gene)]
        child_of_parent_two = [item for item in parent_two if item not in child_of_parent_one]

        child = child_of_parent_one + child_of_parent_two

        return child

    @staticmethod
    def get_reproduction_candidates(population: list, selection_results: list) -> list:
        """Get Candidates from current population for reproduction ... mating ... breeding ... ya know, sex!"""

        return [population[selection_results[index]] for index in range(0, len(selection_results))]

    def mate_the_population(self,candidate_pool: list, alpha_size: int) -> list:
        """Given a list of candidates for reproduction, use them to reproduce and create a list of their offspring"""

        _offspring = []

        _length = len(candidate_pool) - alpha_size
        _pool = sample(candidate_pool, len(candidate_pool))

        # Append Alphas First
        for i in range(0, alpha_size):
            _offspring.append(candidate_pool[i])

        _offspring += [self.reproduce(_pool[i], _pool[len(candidate_pool) - i - 1]) for i in range(0, _length)]

        return _offspring

    @staticmethod
    def mutate(individual: list, mutation_rate: float) -> list:
        """Mutate the given individual path if the mutation rate passes check. A little chaos never hurt."""

        try:
            for switch_pick in range(len(individual)):
                if random() < mutation_rate:
                    switch_with = int(random() * len(individual))

                    node_one = individual[switch_pick]
                    node_two = individual[switch_with]

                    individual[switch_pick] = node_two
                    individual[switch_with] = node_one

            return individual

        except TypeError:
            logger.error(f'Individual list passed in as NoneType: {individual}', exc_info=True)
            raise

    def mutate_population(self, population: list, mutation_rate: float):
        """For each individual in a given population, pass to mutate and see what happens ... """

        mutated_population = [self.mutate(population[index], mutation_rate) for index in range(0, len(population))]
        return mutated_population

    def create_next_generation(self, current_gen, alpha_size: int, mutation_rate: float):
        """Reproduce and Mutate the current generation population candidates"""

        ranked_population = self.rank_paths(current_gen)
        selection_results = self.selection(ranked_population, alpha_size)

        _pool = self.get_reproduction_candidates(population=current_gen, selection_results=selection_results)

        _offspring = self.mate_the_population(
            candidate_pool=_pool,
            alpha_size=alpha_size
        )

        the_next_generation = self.mutate_population(_offspring, mutation_rate)

        return the_next_generation

    @staticmethod
    def user_input_builder(index: str = None, prompt: str = "Please Enter Value for --") -> dict:
        """Handles talking user CLI input for running the GA"""

        class InputFields(Enum):
            POPULATION = 'Initial Population'
            SIZE = 'Population Size'
            ALPHA = 'Number of Alpha / Elite Candidates'
            GENS = 'Number of Generations to Run'
            MUTATION_RATE = 'Mutation Rate'
            PLOT = 'Show Results Plotted? (y/n)'

        # Initial Dict with Defaults
        input_dict = {
            'population': {
                '_input': list(input(f"{prompt} {InputFields.POPULATION.value}: ")),
                'value': GeneticAlgorithm.create_node_dict(size=100)
            },
            'population_size': {
                '_input': int(input(f"{prompt} {InputFields.SIZE.value}: ").lower()),
                'value': 100
            },
            'alpha_size': {
                '_input': int(input(f"{prompt} {InputFields.ALPHA.value}: ").lower()),
                'value': 5
            },
            'generations': {
                '_input': int(input(f"{prompt} {InputFields.GENS.value}: ").lower()),
                'value': 200
            },
            'mutation_rate': {
                '_input': float(input(f"{prompt} {InputFields.MUTATION_RATE.value}: ").lower()),
                'value': 0.02
            },
            'plot_results': {
                '_input': {"y": True, "n": False}[input(f"{prompt} {InputFields.PLOT.value}: ").lower()],
                'value': False
            }
        }

        try:
            # If Index is specified only override the one value, otherwise loop and ask for all.
            if index:
                if index not in input_dict.keys():
                    raise KeyError(f"Not an acceptable Input Field for GA. Options are: {input_dict.keys()}")

                input_dict[index]['value'] = input_dict[index]['_input']

            else:
                for key, val in input_dict.items():
                    input_dict[key]['value'] = val['_input']

            return {key: val['value'] for key, val in input_dict.items()}

        except ValueError as ve:
            logger.error(ve)


# main
if __name__ == '__main__':
    try:
        print("\n HELLO! WELCOME TO THIS SIMPLE GENETIC ALGORITHM FOR OPTIMIZING TRAVERSAL PATHS \n")

        while True:
            request_input = {"y": True, "n": False}[input("Would you like to use Custom Input? (Y/N): ").lower()]

            if request_input:
                print("Requesting Input for GA: \n")
                _input = GeneticAlgorithm.user_input_builder()

                print(f"Custom Input: {_input} \n")

                GeneticAlgorithm(**_input)
            else:
                GeneticAlgorithm()

            run_again = {"y": True, "n": False}[input("Run Again? (Y/N): ").lower()]

            if run_again is False:
                print("Bye for now!")
                break

    except KeyError:
        logger.error("Bad Input")
        exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted!")
        exit(1)
