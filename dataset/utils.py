import os
from collections import defaultdict, Counter
import random
random.seed(42)

base_path = "/mnt/sharedata/ssd/users/jiaxi/dataset/"

class dataset:
    def __init__(self, name,  preprocess=None, task=None, dataset_type=None):
        self.name = name
        self.path = self.get_path()
        self.task = task
        self.dataset_type = dataset_type
        self.samples = []
        self.labels = []

    def get_path(self):
        pass

    def extract_distribution(self):
        pass

    def train_test_gen(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.samples)

def get_dirs(path):
    return [item for item in os.listdir(path) if os.path.isdir(path + item)]

def get_files(path):
    return [item for item in os.listdir(path) if os.path.isfile(path + item)]