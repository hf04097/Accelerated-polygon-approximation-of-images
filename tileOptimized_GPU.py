"""
This file implements the tiled version of polygon approximation of images.
Each image is divided into nxn tiles (subimages) and the computation is parallelized.
"""
#
import cProfile
import sys
import random
import numpy as np
import math
from PIL import Image, ImageDraw, ImageChops
from numba import vectorize
from numba import cuda
from numba import int32
import numba

#Load target image
TARGET = Image.open("pokemon.png").convert('RGB')
WIDTH = TARGET.size[0]
HEIGHT = TARGET.size[1]

#Constants
N = WIDTH * HEIGHT
MUTATION_RATE = 0.02
GENOME_LENGTH = 100

MAX_VERTICES = 4
CROSSOVER_TYPE = 2
FORWARD_TYPE = 2
POPULATION_SIZE = 50
MAX_BATCH = 8
THREADS = 128

GLOBAL_INDIVIDUALS = []
VALIDATION = False


#Tile width and height should be multiple of image size.
#Else the tiles will not be made properly
tile_width = 10
tile_height = 10

SIZE = int((WIDTH / tile_width) * (HEIGHT / tile_height))
DIFFERENCES_CHANNEL = cuda.device_array(shape=(SIZE,), dtype=np.int32)

blocks = (math.ceil(WIDTH * HEIGHT/64), math.ceil(WIDTH * HEIGHT/64))

#Function to split image to tiles.
def split_image_to_tiles(im, grid_width, grid_height):
  w, h = im.size
  w_step = w / grid_width
  h_step = h / grid_height

  tiles = []
  im_coords = []
  for y in range(0, grid_height):
    for x in range(0, grid_width):
      x1 = x * w_step
      y1 = y * h_step
      x2 = x1 + w_step
      y2 = y1 + h_step
      t = im.crop((x1, y1, x2, y2))
      im_coords.append((x1, y1, x2, y2))
      tiles.append(t)
      if x == 0 and y == 0:
        np_tiles = np.array(t.convert('YCbCr').getdata(0), dtype=np.int32)
      else:
        a = np.array(t.convert('YCbCr').getdata(0), dtype=np.int32)
        np.concatenate((np_tiles, a))
  return tiles, np_tiles, im_coords

#Initialize target tiles array
TARGET_TILES = split_image_to_tiles(TARGET, tile_width, tile_height)
target_data = TARGET_TILES[1]
TARGET_TILES_GPU = cuda.to_device(target_data)

"""
This class represents genes which make up an individual.
In polygon approximation, these genes would be image RGBA and vertices.
"""
class Gene(object):
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

"""
This class represents individuals in a population.
In polygon approximation, these individuals would be an image.
"""
class Individual(object):
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
        self.tile_images, self.tiles, self.tile_coords = self.tile_image()
        self.fitness = self.compute_fitness_GPU()
        self.image = self.img_join()

        if VALIDATION:
            GLOBAL_INDIVIDUALS.append(self)

    def img_join(self): #Join tiles to form a single image
        img = Image.new('RGB', (WIDTH, HEIGHT))
        for i in range(len(self.tile_images)):
            coords = list(map(int, self.tile_coords[i]))
            img.paste(self.tile_images[i], coords)
        return img


    def get_fitness(self):
        if self._fitness is None:
            assert self._fitness is not None
        return self._fitness

    def set_fitness(self, value):
        self._fitness = value

    fitness = property(get_fitness, set_fitness)

    def crossover(self): #Crossover operation on pixel arrays
        if CROSSOVER_TYPE == 1:
            crossover_point = random.randint(1, GENOME_LENGTH)
            self.genome = mother.genome[:crossover_point]
            self.genome.extend(father.genome[crossover_point:])
        elif CROSSOVER_TYPE == 2:
            # Solution of the github inspiration
            for i in range(GENOME_LENGTH):
                if random.random() > 0.5:
                    self.add_gene(self.mother.genome[i])
                else:
                    self.add_gene(self.father.genome[i])

    def add_gene(self, x):
        self.genome.append(x)

    def draw_individual(self): #Generate image for individual
        image = Image.new('RGB', (WIDTH, HEIGHT))
        draw = ImageDraw.Draw(image, 'RGBA')
        for i in range(GENOME_LENGTH):
            poly = self.genome[i].vertices
            color = (self.genome[i].R, self.genome[i].G, self.genome[i].B, self.genome[i].A)
            draw.polygon(poly, color)
        return image

    def tile_image(self): #Divide image to tiles and set parameters
      tile_images, np_tiles, tile_coords = split_image_to_tiles(self.image, tile_width, tile_height)
      return tile_images, np_tiles, tile_coords

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

    def compute_fitness_2(self):
        diff = ImageChops.difference(self.image.convert('YCbCr'), TARGET.convert('YCbCr'))
        data = diff.getdata()
        illness = 1.0 * sum(map(sum, data)) / (WIDTH * HEIGHT * abs(16 - 235) * abs(16 - 240) * 2)
        return 1 - illness

    def compute_fitness_GPU(self):

        tiles_data_GPU = cuda.to_device(self.tiles)


        threads_per_block = 128
        blocks = math.ceil(WIDTH * HEIGHT / 128)
        difference[blocks, threads_per_block](tiles_data_GPU, TARGET_TILES_GPU, DIFFERENCES_CHANNEL)

        cuda.synchronize()
        sum = sum_reduce(DIFFERENCES_CHANNEL)
        illness = 1.0 * sum / (WIDTH * HEIGHT * abs(16 - 235) * abs(16 - 240) * 2)
        return 1 - illness

    def compute_fitness_BATCH(self):
        pass

    def mutate(self): #Mutation operation on pixel arrays
        for i in range(GENOME_LENGTH):
            if random.random() < MUTATION_RATE:
                self.genome[i] = Gene()




#CUDA kernel function to compute difference between arrays
@cuda.jit("void(int32[:], int32[:], int32[:])")
def difference(current_image, target_image, differences):
    i = cuda.grid(1)
    sm_current = cuda.shared.array(SIZE, int32)
    sm_target = cuda.shared.array(SIZE, int32)

    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x;

    if j < SIZE:
      sm_current[cuda.threadIdx.x] = current_image[j]
      sm_target[cuda.threadIdx.x] = target_image[j]
    cuda.syncthreads()

    differences[i] = abs(current_image[cuda.threadIdx.x] - target_image[cuda.threadIdx.x])

#CUDA kernel function to compute sum of array
@cuda.reduce
def sum_reduce(a, b):
    return a + b

"""
This class represents the population of the sample space.
In polygon approximation, this population is the set of all images generated.
"""
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
        # All individuals are sorted, so just pick the first 2
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

#Main function
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
    data = diff.getdata(0)
    illness = 1.0 * sum(data) / (WIDTH * HEIGHT * abs(16 - 235) * abs(16 - 240) * 2)
    return 1 - illness

#Iterations for timing and validation
def run_iterations(gens, pop_size, debug = False, validation = False):
    times = []
    sizes = []
    print("starting computing for GPU in tiled mode")
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
        print(GPU_fitness)
        CPU_fitness = [compute_fitness(i.image) for i in GLOBAL_INDIVIDUALS]
        print(CPU_fitness)
        assert np.allclose(GPU_fitness, CPU_fitness, atol= 1e-3, rtol = 0)
        print(f'Results validated :)')
        print(f'maximum difference = {np.amax(np.abs(np.array(GPU_fitness) - np.array(CPU_fitness)))}')
    return times, sizes





if __name__ == "__main__":
    #main()
    print(run_iterations([2], [15]))
