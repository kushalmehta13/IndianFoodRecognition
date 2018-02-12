import os
import shutil
import h5py
import numpy as np
from scipy import misc
import itertools
import random
import json
import sys

class dataset_loader:
    dimensions = ()
    path = ''
    image_pairs_left = []
    image_pairs_right = []
    image_pairs_labels = []
    categories = []
    images_paths = {}
    default_name = 'dataset_oneshot.h5'

    def __init__(self, path, dimensions):
        files = [x for x in os.listdir(path) if os.path.isfile(path + '/' + x)]
        self.path = path
        if self.default_name in files:
            print("Reading from cache...")
            self.load_from_disk()
        else:
            print("Reading afresh")
            self.categories = [x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]
            self.dimensions = dimensions
            for category in self.categories:
                temp_path = path + '/' + category
                temp_category_paths = os.listdir(temp_path)
                temp_category_paths = list(map(lambda x : temp_path + '/' + x, temp_category_paths))
                self.images_paths[category] = temp_category_paths

            self.generate_dataset()
            self.save_to_disk()

    def read_image(self, path):
        img = misc.imread(path, 'L')
        return np.reshape(misc.imresize(img, self.dimensions), self.dimensions)

    def generate_dataset(self):
        for category in self.categories:
            same_category_combinations = list(itertools.combinations(self.images_paths[category],2))
            no_positive_examples = len(same_category_combinations)
            for combination in same_category_combinations:
                self.image_pairs_left.append(self.read_image(combination[0]))
                self.image_pairs_right.append(self.read_image(combination[1]))
                self.image_pairs_labels.append(1)

            other_categories = [x for x in self.categories if(x != category)]
            for other_category in other_categories:
                different_category_combinations = list(itertools.product(self.images_paths[category],self.images_paths[other_category]))
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

    def delete_from_disk(self):
        os.remove(self.path + '/' + self.default_name)
        os.remove(self.path + '/' + self.default_name + '.json')

    def save_to_disk(self):
        path = self.path + '/' + self.default_name
        with h5py.File(path, 'w') as hf:
            hf.create_dataset('image_pairs', data=[self.image_pairs_left, self.image_pairs_right])
            hf.create_dataset('labels', data=self.image_pairs_labels)

        path_json = self.path + '/' + self.default_name + ".json"
        json_to_write = {}
        json_to_write['categories'] = self.categories
        json_to_write['dimensions'] = list(self.dimensions)
        json_to_write['images_paths'] = self.images_paths

        with open(path_json, 'w') as jfl:
            json.dump(json_to_write, jfl, sort_keys=True, indent=4)

    def load_from_disk(self):
        path = self.path + '/' + self.default_name
        with h5py.File(path, 'r') as hf:
            [self.image_pairs_left, self.image_pairs_right] = hf['image_pairs'][:]
            self.image_pairs_labels = hf['labels'][:]

        path_json = self.path + '/' + self.default_name + ".json"
        with open(path_json, 'r') as jfl:
            data = json.load(jfl)
            self.categories = data['categories']
            self.dimensions = tuple(data['dimensions'])
            self.images_paths = data['images_paths']

    def update_dataset(self, image_label_pair):
        #image_label_pair --> [[path of image,name of category],[path...,category],...]
        return_image_pairs_left = []
        return_image_pairs_right = []
        return_image_pairs_labels = []
        print("In util")
        for pair in image_label_pair:
            shutil.move(pair[0],pair[1])
            image_path = pair[1]
            category = image_path.split('/')[-2]
            print(category)

            same_category_combinations = list(itertools.product(self.images_paths[category], [image_path]))
            additional_positive_length = len(same_category_combinations)
            for combination in same_category_combinations:
                return_image_pairs_left.append(self.read_image(combination[0]))
                return_image_pairs_right.append(self.read_image(combination[1]))
                return_image_pairs_labels.append(1)

            other_categories = [x for x in self.categories if (x != category)]
            for other_category in other_categories:
                different_category_combinations = list(itertools.product(self.images_paths[other_category], [image_path]))
                different_category_combinations = random.sample(different_category_combinations, min(additional_positive_length,len(different_category_combinations)))
                for combination in different_category_combinations:
                    return_image_pairs_left.append(self.read_image(combination[0]))
                    return_image_pairs_right.append(self.read_image(combination[1]))
                    return_image_pairs_labels.append(0)
        return_image_pairs_left = np.asarray(return_image_pairs_left)
        return_image_pairs_right = np.asarray(return_image_pairs_right)
        return_image_pairs_labels = np.asarray(return_image_pairs_labels)

        self.image_pairs_left = np.concatenate((self.image_pairs_left, return_image_pairs_left))
        self.image_pairs_right = np.concatenate((self.image_pairs_right, return_image_pairs_right))
        self.image_pairs_labels = np.concatenate((self.image_pairs_labels, return_image_pairs_labels))

        self.delete_from_disk()
        self.save_to_disk()

        return return_image_pairs_left, return_image_pairs_right, return_image_pairs_labels
