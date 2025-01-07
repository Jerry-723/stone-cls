from .utils import *
import re

class Meteorite(dataset):
    def __init__(self):
        super().__init__("meteorite")

    def get_path(self):
        return base_path + "是陨石/"
    
    def extract_distribution(self):
        res_count = {}
        try:
            directories = get_dirs(self.path)
            for directory in directories:
                files = os.listdir(self.path + directory)
                count = len(files)
                type_name = re.sub(r'\d+|照片', '', directory)
                res_count[type_name] = count
            sorted_count_types = sorted(res_count.items(), key=lambda x: x[1], reverse=True)
            res_count = dict(sorted_count_types)
            return res_count
        except FileNotFoundError:
            print("Path not found: " + self.path)
            return
        except PermissionError:
            print("Permission denied: " + self.path)
            return