import os
import math
import numpy as np
from scipy import misc
import itertools
import random

class dataset_loader:
    dimensions = ()
    image_pairs_left = []
    image_pairs_right = []
    image_pairs_labels = []
    positive_examples_pairs = []
    number_categories = 0

    def __init__(self, path, dimensions):
        categories = os.listdir(path)
        images_paths = []
        self.dimensions = dimensions
        self.number_categories = len(categories)
        for category in categories:
            temp_path = path + '/' + category
            temp_category_paths = os.listdir(temp_path)
            temp_category_paths = list(map(lambda x : temp_path + '/' + x, temp_category_paths))
            images_paths.append(temp_category_paths)

        self.generate_dataset(images_paths)

    def read_image(self, path):
        img = misc.imread(path, 'L')
        return np.reshape(misc.imresize(img, self.dimensions), self.dimensions)

    def generate_dataset(self, images_paths):
        length = len(images_paths)
        for category in range(length):
            print(category)
            same_category_combinations = list(itertools.combinations(images_paths[category],2))
            no_positive_examples = len(same_category_combinations)
            self.positive_examples_pairs.append(no_positive_examples)
            for combination in same_category_combinations:
                self.image_pairs_left.append(self.read_image(combination[0]))
                self.image_pairs_right.append(self.read_image(combination[1]))
                self.image_pairs_labels.append(1)

            other_categories = [x for x in range(length) if(x != category)]
            for other_category in other_categories:
                different_category_combinations = list(itertools.product(images_paths[category],images_paths[other_category]))
                different_category_combinations = random.sample(different_category_combinations, no_positive_examples)
                for combination in different_category_combinations:
                    self.image_pairs_left.append(self.read_image(combination[0]))
                    self.image_pairs_right.append(self.read_image(combination[1]))
                    self.image_pairs_labels.append(0)

        self.image_pairs_left = np.asarray(self.image_pairs_left)
        self.image_pairs_right = np.asarray(self.image_pairs_right)
        self.image_pairs_labels = np.asarray(self.image_pairs_labels)

    def get_dataset(self):
        return self.image_pairs_left, self.image_pairs_right, self.image_pairs_labels
