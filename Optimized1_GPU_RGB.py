
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

#arrays to hold each channcel of target image
TARGET_CHANNEL = np.array(TARGET.convert('YCbCr').getdata(0), dtype = np.int32)
TARGET_CHANNEL_GPU_0 = cuda.to_device(TARGET_CHANNEL)

TARGET_CHANNEL = np.array(TARGET.convert('YCbCr').getdata(1), dtype = np.int32)
TARGET_CHANNEL_GPU_1 = cuda.to_device(TARGET_CHANNEL)

TARGET_CHANNEL = np.array(TARGET.convert('YCbCr').getdata(2), dtype = np.int32)
TARGET_CHANNEL_GPU_2 = cuda.to_device(TARGET_CHANNEL)

##Array to hold differences of current batch with target image
DIFFERENCES_CHANNEL_0 = cuda.device_array(shape=(WIDTH * HEIGHT,), dtype=np.int32)
DIFFERENCES_CHANNEL_1 = cuda.device_array(shape=(WIDTH * HEIGHT,), dtype=np.int32)
DIFFERENCES_CHANNEL_2 = cuda.device_array(shape=(WIDTH * HEIGHT,), dtype=np.int32)

## array to hold the current image
zeros = np.zeros((WIDTH, HEIGHT, 3), dtype = np.int32)
IMG_GPU = cuda.to_device(zeros)
#IMG_GPU_1 = cuda.to_device(zeros)
#IMG_GPU_2 = cuda.to_device(zeros)



GLOBAL_INDIVIDUALS = []
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
    # Represents a an indvidual which is an image
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

        #self.image = self.draw_individual()
        self._fitness = self.get_fast_fitness()
        if VALIDATION:
            GLOBAL_INDIVIDUALS.append(self)

    def get_fast_fitness(self):
        GPU_version = np.zeros((100, 10), dtype=np.int32) #array that will hold all polygons
        for i in range(len(self.genome)):
            vertices = sorted(self.genome[i].vertices)
            for j in range(len(vertices)):
                GPU_version[i, j * 2] = vertices[j][0]
                GPU_version[i, (j * 2) + 1] = vertices[j][1]

            GPU_version[i, 6] = self.genome[i].R
            GPU_version[i, 7] = self.genome[i].G
            GPU_version[i, 8] = self.genome[i].B
            GPU_version[i, 9] = self.genome[i].A

        #sending polygons to GPU
        in_GPU = cuda.to_device(GPU_version)

        threads_per_block = (16, 16, 3)
        blocks = (math.ceil(WIDTH / 16), math.ceil(HEIGHT / 16), 100) #z dimension for polygons

        #converting polygons to image
        fast_fitness[blocks, threads_per_block](IMG_GPU, in_GPU, WIDTH, HEIGHT)

        threads_per_block = (32, 32)
        blocks = (math.ceil(WIDTH / 32), math.ceil(HEIGHT / 32))

        #computing different wrt target image
        fast_difference[blocks, threads_per_block](IMG_GPU, TARGET_CHANNEL_GPU_0,
                                                   TARGET_CHANNEL_GPU_1, TARGET_CHANNEL_GPU_2, DIFFERENCES_CHANNEL_0,
                                                   DIFFERENCES_CHANNEL_1, DIFFERENCES_CHANNEL_2, WIDTH, HEIGHT)

        #sum them up
        s1 = sum_reduce(DIFFERENCES_CHANNEL_0)
        s2 = sum_reduce(DIFFERENCES_CHANNEL_1)
        s3 = sum_reduce(DIFFERENCES_CHANNEL_2)

        return 1 - (s1 + s2 + s3) / (WIDTH * HEIGHT * abs(16 - 235) * abs(16 - 240) * 2)


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

    def compute_fitness_BATCH(self):
        pass

    def mutate(self):
        for i in range(GENOME_LENGTH):
            if random.random() < MUTATION_RATE:
                self.genome[i] = Gene()


@cuda.jit("void(int32[:, :, :], int32[:, :], int32, int32)")
def fast_fitness(img, in_GPU, WIDTH, HEIGHT):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z

    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bz = cuda.blockIdx.z

    dimx = cuda.blockDim.x
    dimy = cuda.blockDim.y

    global_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    global_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    poly = cuda.shared.array((10,), int32)

    #loading whole of polygon to shared memory
    if (tx + (16 * ty) < 10):
        poly[tx + (16 * ty)] = in_GPU[bz, tx + (16 * ty)]

    cuda.syncthreads()

    y = global_y
    x = global_x


    #x2, y2 = poly[2], poly[3]
    #x3, y3 = poly[4], poly[5]

    denominator = ((poly[3] - poly[5]) * (poly[0] - poly[4]) + (poly[4] - poly[2]) * (poly[1] - poly[5]))
    a = ((poly[3] - poly[5]) * (x - poly[4]) + (poly[4] - poly[2]) * (y - poly[5])) / denominator;
    b = ((poly[5] - poly[1]) * (x - poly[4]) + (poly[0] - poly[4]) * (y - poly[5])) / denominator;
    c = 1 - a - b;

    #checks if the current pixels is inside the polygon
    if 0 <= a and a <= 1 and 0 <= b and b <= 1 and 0 <= c and c <= 1 and x < WIDTH and y < HEIGHT:
        color = poly[6 + tz]/255
        A = poly[9] / 255
        color = int((1 - A) * color) + (A * color)
        color = min((color * 255), 255)
        cuda.atomic.add(img, (x, y, tz), color)



@cuda.jit(
    "void(int32[:, :, :], int32[:], int32[:], int32[:],  int32[:], int32[:], int32[:], int32, int32)")
def fast_difference(img, TARGET_CHANNEL_GPU_0, TARGET_CHANNEL_GPU_1, TARGET_CHANNEL_GPU_2,
                    DIFFERENCES_CHANNEL_0, DIFFERENCES_CHANNEL_1, DIFFERENCES_CHANNEL_2, WIDTH, HEIGHT):
    global_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    global_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if (global_x < WIDTH and global_y < HEIGHT):
        colorR = min(img[global_x, global_y, 0], 255)
        colorG = min(img[global_x, global_y, 0], 255)
        colorB = min(img[global_x, global_y, 0], 255)


        #RGB to YCrCB conversion
        y = int(.299 * colorR + .587 * colorG + .114 * colorB)
        cb = int(128 - .168736 * colorR - .331364 * colorG + .5 * colorB)
        cr = int(128 + .5 * colorR - .418688 * colorG - .081312 * colorB)

        # = colorR
        # cb  = colorG
        # cr = colorB

        DIFFERENCES_CHANNEL_0[global_x + WIDTH * global_y] = int(
            abs(y - TARGET_CHANNEL_GPU_0[global_x + WIDTH * global_y]))
        DIFFERENCES_CHANNEL_1[global_x + WIDTH * global_y] = int(
            abs(cb - TARGET_CHANNEL_GPU_1[global_x + WIDTH * global_y]))
        DIFFERENCES_CHANNEL_2[global_x + WIDTH * global_y] = int(
            abs(cr - TARGET_CHANNEL_GPU_2[global_x + WIDTH * global_y]))
        img[global_x, global_y, 0] = 0
        img[global_x, global_y, 1] = 0
        img[global_x, global_y, 2] = 0
        #img_1[global_x, global_y] = 0
        #img_2[global_x, global_y] = 0

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
    print("starting computing for GPU!!")
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
        for i in range(len(GLOBAL_INDIVIDUALS)):
            GLOBAL_INDIVIDUALS[i].image = GLOBAL_INDIVIDUALS[i].draw_individual()
        CPU_fitness = [compute_fitness(i.image) for i in GLOBAL_INDIVIDUALS]
        assert np.allclose(GPU_fitness, CPU_fitness, atol= 1e-3, rtol=0)
        print(f'Results validated! :)')
        print(f"maximum difference = {np.amax(np.abs(np.array(GPU_fitness) - np.array(CPU_fitness)))}")

    return times, sizes





if __name__ == "__main__":
    #main()
    #print(run_iterations([2], [15]))
    print(run_iterations([5, 10, 100], [20]))