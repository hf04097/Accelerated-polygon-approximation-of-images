
import cProfile

from PIL import Image
import sys
import random

import numpy as np
import math

from PIL import Image, ImageDraw, ImageChops
import numpy as np

from numba import vectorize
from numba import cuda
import numba
import numpy as np
import math

from numba import int32

TARGET = Image.open("pokemon.png").convert('RGB')
WIDTH = TARGET.size[0]
HEIGHT = TARGET.size[1]


N = WIDTH * HEIGHT

# Number of members in a population
MUTATION_RATE = 0.02
# How many polygons are in an individual
GENOME_LENGTH = 100

MAX_VERTICES = 4
CROSSOVER_TYPE = 2
FORWARD_TYPE = 2
POPULATION_SIZE = 50


##storing each channel of image in GPU beforehand
TARGET_CHANNEL = np.array(TARGET.convert('YCbCr').getdata(0), dtype = np.int32)
TARGET_CHANNEL_GPU_0 = cuda.to_device(TARGET_CHANNEL)

TARGET_CHANNEL = np.array(TARGET.convert('YCbCr').getdata(1), dtype = np.int32)
TARGET_CHANNEL_GPU_1 = cuda.to_device(TARGET_CHANNEL)

TARGET_CHANNEL = np.array(TARGET.convert('YCbCr').getdata(2), dtype = np.int32)
TARGET_CHANNEL_GPU_2 = cuda.to_device(TARGET_CHANNEL)


##array that will store the absolute difference for each image
DIFFERENCES_CHANNEL_0 = cuda.device_array(shape=(WIDTH * HEIGHT,), dtype=np.int32)
DIFFERENCES_CHANNEL_1 = cuda.device_array(shape=(WIDTH * HEIGHT,), dtype=np.int32)
DIFFERENCES_CHANNEL_2 = cuda.device_array(shape=(WIDTH * HEIGHT,), dtype=np.int32)

##arrays where the image will be transferred
IMAGE_ARRAY_GPU_0 = cuda.device_array(shape=(WIDTH * HEIGHT,), dtype=np.int32)
IMAGE_ARRAY_GPU_1 = cuda.device_array(shape=(WIDTH * HEIGHT,), dtype=np.int32)
IMAGE_ARRAY_GPU_2 = cuda.device_array(shape=(WIDTH * HEIGHT,), dtype=np.int32)


GLOBAL_INDIVIDUALS = [] #validation array
VALIDATION = False

class Gene(object):
    ##Represents a polygon
    def __init__(self):
        self.n_vertices = random.randint(2, MAX_VERTICES)
        self.vertices = []
        self.randomize_RGBA()
        self.randomize_vertices()

    def randomize_RGBA(self):
        self.R = random.randint(0, 255)
        self.G = random.randint(0, 255)
        self.B = random.randint(0, 255)
        self.A = random.randint(0, 50)

    def randomize_vertices(self):
        for _ in range(self.n_vertices):
            vertex = random.uniform(-10, WIDTH + 10), random.uniform(-10, HEIGHT + 10)
            self.vertices.append(vertex)


class Individual(object):
    #Represents a an indvidual which is an image
    def __init__(self, mother=None, father=None):
        self.genome = []
        self.mother = mother
        self.father = father
        if mother is None and father is None:
            for _ in range(GENOME_LENGTH):
                self.add_gene(Gene())
        else:
            self.crossover()
            self.mutate()

        self.image = self.draw_individual()
        self._fitness = self.compute_fitness_GPU()
        if VALIDATION:
            GLOBAL_INDIVIDUALS.append(self)


    def get_fitness(self):
        # print('getter called')
        if self._fitness is None:
            assert self._fitness is not None
        return self._fitness

    def set_fitness(self, value):
        self._fitness = value

    fitness = property(get_fitness, set_fitness)

    def crossover(self):
        # if CROSSOVER_TYPE == 1:
        #  crossover_point = random.randint(1, GENOME_LENGTH)
        #  self.genome = mother.genome[:crossover_point]
        #  self.genome.extend(father.genome[crossover_point:])
        # elif CROSSOVER_TYPE == 2:
        # Solution of the github inspiration
        for i in range(GENOME_LENGTH):
            if random.random() > 0.5:
                self.add_gene(self.mother.genome[i])
            else:
                self.add_gene(self.father.genome[i])

    def add_gene(self, x):
        self.genome.append(x)

    def draw_individual(self):
        image = Image.new('RGB', (WIDTH, HEIGHT))
        draw = ImageDraw.Draw(image, 'RGBA')
        for i in range(GENOME_LENGTH):
            poly = self.genome[i].vertices
            color = (self.genome[i].R, self.genome[i].G, self.genome[i].B, self.genome[i].A)
            draw.polygon(poly, color)
        return image

    def compute_fitness(self):
        diff = ImageChops.difference(self.image.convert('YCbCr'), TARGET.convert('YCbCr'))
        data = diff.getdata(0)
        illness = 1.0 * sum(data) / (WIDTH * HEIGHT * 255 * 3)
        return 1 - illness
        # self.fitness = 1 - illness

    def compute_fitness_CPU(self):
        image_data = np.array(self.image.convert('YCbCr').getdata(0))
        target_data = np.array(TARGET.convert('YCbCr').getdata(0))
        diff = np.full_like(target_data, 0)

        for i in range(WIDTH * HEIGHT):
            for k in range(1):
                diff[i] = abs(image_data[i] - target_data[i])

        sum = 0
        for i in range(WIDTH * HEIGHT):
            for k in range(1):
                sum += diff[i]

        illness = 1.0 * sum / (WIDTH * HEIGHT * abs(16 - 235) * abs(16 - 240) * 2)
        return 1 - illness


    def compute_fitness_GPU(self):
        image_data_0 = np.array(self.image.convert('YCbCr').getdata(0), dtype= np.int32)
        image_data_1 = np.array(self.image.convert('YCbCr').getdata(1), dtype= np.int32)
        image_data_2 = np.array(self.image.convert('YCbCr').getdata(2), dtype= np.int32)
        IMAGE_ARRAY_GPU_0.copy_to_device(image_data_0)
        IMAGE_ARRAY_GPU_1.copy_to_device(image_data_1)
        IMAGE_ARRAY_GPU_2.copy_to_device(image_data_2)

        threads_per_block = 64
        blocks = math.ceil(WIDTH * HEIGHT / 64)

        difference[blocks, threads_per_block](IMAGE_ARRAY_GPU_0, TARGET_CHANNEL_GPU_0, DIFFERENCES_CHANNEL_0, WIDTH * HEIGHT)
        difference[blocks, threads_per_block](IMAGE_ARRAY_GPU_1, TARGET_CHANNEL_GPU_1, DIFFERENCES_CHANNEL_1,
                                              WIDTH * HEIGHT)
        difference[blocks, threads_per_block](IMAGE_ARRAY_GPU_2, TARGET_CHANNEL_GPU_2, DIFFERENCES_CHANNEL_2,
                                              WIDTH * HEIGHT)
        cuda.synchronize()
        sum = sum_reduce(DIFFERENCES_CHANNEL_0) + sum_reduce(DIFFERENCES_CHANNEL_1) + sum_reduce(DIFFERENCES_CHANNEL_2)
        illness = 1.0 * sum / (WIDTH * HEIGHT * abs(16 - 235) * abs(16 - 240) * 2)
        return 1 - illness

    def compute_fitness_BATCH(self):
        pass

    def mutate(self):
        for i in range(GENOME_LENGTH):
            if random.random() < MUTATION_RATE:
                self.genome[i] = Gene()


@cuda.jit("void(int32[:], int32[:], int32[:], int32)")
def difference(current_image, target_image, differences, length):
    i = cuda.grid(1)
    if i < length:
      differences[i] = abs(current_image[i] - target_image[i])

@cuda.reduce
def sum_reduce(a, b):
    return a + b


class Population(object):
    def __init__(self):
        self.individuals = []
        for _ in range(POPULATION_SIZE):
            ind = Individual()
            self.add_individual(ind)
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    def add_individual(self, x):
        self.individuals.append(x)

    def forward(self):
        # All individuals are sorted, so I just pick the first 2
        if FORWARD_TYPE == 1:
            del self.individuals[-1]
            del self.individuals[-1]

            fittest = self.individuals[0]
            second_fittest = self.individuals[1]
            self.add_individual(Individual(fittest, second_fittest))
            self.add_individual(Individual(second_fittest, fittest))
        elif FORWARD_TYPE == 2:
            # The solution of the github inspiration
            selection_count = int(POPULATION_SIZE / 10)
            for i in range(selection_count, POPULATION_SIZE):
                m = self.individuals[random.randint(0, selection_count - 1)]
                f = self.individuals[random.randint(0, selection_count - 1)]
                self.individuals[i] = Individual(m, f)
            self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    def fittest(self):
        return self.individuals[0]


def main():
    population = Population()
    POPULATION_SIZE = 50
    GENERATIONS = 50
    for i in range(GENERATIONS):
        print(f"Current generation: {i}")
        population.forward()

        fittest = population.fittest()
        print(f"Best fitness: {fittest.fitness}")
        if (i + 1) % 100 == 0:
            fittest.image.save(f"fittest_gen{i}.png")

import time


def compute_fitness(current_image):
    diff = ImageChops.difference(current_image.convert('YCbCr'), TARGET.convert('YCbCr'))
    data = diff.getdata()
    illness = 1.0 * sum(map(sum, data)) / (WIDTH * HEIGHT * abs(16 - 235) * abs(16 - 240) * 2)
    return 1 - illness

def run_iterations(gens, pop_size, debug = False, validation = False):
    times = []
    sizes = []
    print("starting computing for GPU naive, you might have to wait :)")
    if validation:
        global VALIDATION
        VALIDATION = True
    for GEN in gens:
        for POP in pop_size:
            print(f"running for population of size {POP} at {GEN}")
            global POPULATION_SIZE
            POPULATION_SIZE = POP
            GENERATIONS = GEN
            start = time.time()
            population = Population()
            for i in range(GENERATIONS):
                population.forward()
                fittest = population.fittest()
                if debug:
                    print(f"Current generation: {i}")
                    print(f"Best fitness: {fittest.fitness}")
                    if (i + 1) % 100 == 0:
                        fittest.image.save(f"fittest_gen{i}.png")
            end = time.time()
            times.append(end - start)
            sizes.append((GEN, POP))
    if validation:
        print(f'Timing completed, validating results!')
        GPU_fitness = [i.fitness for i in GLOBAL_INDIVIDUALS]
        CPU_fitness = [compute_fitness(i.image) for i in GLOBAL_INDIVIDUALS]
        assert np.allclose(GPU_fitness, CPU_fitness)
        print(f'Results validated :)')
        print(f"maximum difference = {np.amax(np.abs(np.array(GPU_fitness) - np.array(CPU_fitness)))}")

    return times, sizes





if __name__ == "__main__":
    #main()
    print(run_iterations([2], [15]))