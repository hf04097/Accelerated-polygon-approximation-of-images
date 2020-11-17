import numpy as np
import math
import random
import matplotlib
from matplotlib import pyplot as plt
import time
import cv2
from PIL import Image, ImageDraw, ImagePath
from shutil import copyfile
import os

##Code inspired from Hiba Jamal's CI assginment.

class Individual:
    def __init__(self, polygons, colors, generation, id, height, width):
        self.polygons = polygons
        self.colors = colors
        self.name = f'{generation}_{id}.png'
        self.fitness = -1
        self.height = height
        self.width = width
        self.fitness = None
        self.image = None
        self.sides = 3

    def translate(self, i):
        current_polygon = self.polygons[i]
        x = [j[0] for j in current_polygon]
        y = [j[1] for j in current_polygon]
        max_x = max(x)
        max_y = max(y)
        delta_x = random.randint(0 - max_x, self.width - max_x)
        delta_y = random.randint(0 - max_y, self.height - max_y)
        self.polygons[i] = [(x[j] + delta_x, y[j] + delta_y) for j in range(len(x))]

    def generate_polygons(self, n = 100):
        sides = 3
        for i in range(n):
            vertices = [self.get_random_cord()]
            x = vertices[0][0]
            y = vertices[0][1]
            for j in range(sides - 1):
                start_x = min(x, x + self.width//8)
                end_x = max(x, x + self.width//8)
                start_y = min(y, y + self.height//8)
                end_y = max(y, y + self.height//8)
                vertices.append(self.get_random_cord(range_val=[start_x, end_x, start_y, end_y]))
            self.polygons.append(vertices)
            self.colors.append(self.get_random_color())



    def randomize_colors(self, rate = 1):
        for i in range(len(self.colors)):
            if random.random() < rate:
                self.colors[i] = self.get_random_color()

    def get_img_rep(self):
        if self.image is None:
            img = Image.new(mode = 'RGB', size = (self.width, self.height))
            draw = ImageDraw.Draw(img, 'RGBA')

            for i in range(len(self.polygons)):
                current_poly = self.polygons[i]
                current_color = self.colors[i]

                draw.polygon(xy = current_poly, fill = current_color)
            self.image = img
        return self.image

    def generate_fitness(self, target, fitness_func):
        self.fitness = fitness_func(self.get_img_rep().convert("RGB"), target)
        return self.fitness

    def save_img(self, name = ''):
        if not name:
            name = self.name
        if self.image is None:
            self.get_img_rep()
        self.image.save(name)

    def get_random_color(self, range_val = (0, 255)):

        return tuple([255 for i in range(3)] + [random.randint(range_val[0], range_val[1])])

    def get_random_cord(self, range_val = None):
        if range_val is None:
            return (random.randint(0, self.width), random.randint(0, self.height))
        return (random.randint(range_val[0], range_val[1]), random.randint(range_val[2], range_val[3]))

    def generate_img(self, n = 100):
        self.generate_polygons(n = 100)
        self.randomize_colors()

    def __add__(self, other):
        return self.fitness + other.fitness

    def cross_over(self, mate, gen):
        new_polygons = self.polygons.copy() + mate.polygons.copy()
        new_colors = self.colors.copy() + mate.colors.copy()
        new_size = random.randint(min(len(self.polygons), len(new_polygons)), max(len(self.polygons), len(new_polygons)))
        indices = np.random.permutation(len(new_polygons))
        off_spring_polygons = []
        off_spring_colors = []
        for i in range(new_size):
            off_spring_polygons.append(new_polygons[indices[i]])
            off_spring_colors.append(new_colors[indices[i]])
        return Individual(off_spring_polygons, off_spring_colors, gen, 0, self.height, self.width)

    def mutation(self):
        for i in range(len(self.polygons)):
            if random.random() < 0.5:
                if random.random() < 0.5:
                    self.colors[i] = self.get_random_color()
                else:
                    self.translate(i)







def fitness_func(current_image, target_image):
    current_array = np.array(current_image)
    target_array = np.array(target_image)

    return np.mean(np.abs(current_array - target_array))




class GeneticAlgo:
    def __init__(self, pop_size, generations, target_image):
        self.pop_size = pop_size
        self.generations = generations
        self.current_gen = 0
        self.current_population = []
        self.current_fitness = []
        self.img = target_image
        self.width = target_image.size[1]
        self.height = target_image.size[0]
        self.offspring = int(self.pop_size * 0.2)
        self.mutation_rate = 0.5

    def initialize_population(self):
        assert len(self.current_population) == 0

        for i in range(self.pop_size):
            obj = Individual([], [], 0, i, self.width, self.height)
            obj.generate_polygons(random.randint(0, 10))
            obj.generate_fitness(self.img, fitness_func)
            self.current_population.append(obj)
            self.current_fitness.append(obj.fitness)

    def binary_tournament(self, size_to_remove):
        pass



    def evolution(self):
        self.initialize_population()
        best_obj_ind = np.argmin(self.current_fitness)
        best_obj = self.current_population[best_obj_ind]
        min_fit = self.current_fitness[best_obj_ind]
        avg_fit = np.mean(self.current_fitness)
        history = [min_fit]
        best_obj.save_img()
        print(min_fit, avg_fit)

        for gen in range(self.generations):
            print(gen)
            new_generation = []
            for i in range(self.offspring):
                ind1 = (i)
                ind2 = random.randint(0, self.pop_size - 1)
                offspring = self.current_population[ind1].cross_over(self.current_population[ind2], gen)
                if (random.random() < self.mutation_rate):
                    offspring.mutation()

                offspring.generate_fitness(self.img, fitness_func)
                self.current_fitness.append(offspring.fitness)
                new_generation.append(offspring)

            self.current_population += new_generation
            self.current_fitness.sort()
            self.current_population.sort(key = lambda c: self.current_fitness.index(c.fitness))

            self.current_population = self.current_population[:self.pop_size]
            self.current_fitness = self.current_fitness[:self.pop_size]

            if self.current_fitness[0] < min_fit:
                self.current_population[0].save_img()
            min_fit = self.current_fitness[0]
            avg_fit = np.mean(self.current_fitness)

            print(min_fit, avg_fit)
            history.append(min_fit)

















polygos = [[(0,0), (50, 50), (50, 0)]]
colors = [(255, 255, 0, 255)]

image = Image.open("mona_lisa_monocrhome.jpg")
algo = GeneticAlgo(1000, 5000, image)
algo.evolution()
##algo.initialize_population()
##print(algo.current_population[0].generate_fitness(image, fitness_func))











