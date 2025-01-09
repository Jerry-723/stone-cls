from .utils import *
from PIL import Image, UnidentifiedImageError


class Stone(dataset):
    def __init__(self, preprocess=None, task="general", dataset_type="train", prepared=True):
        super().__init__("stone", preprocess, task, dataset_type)
        self.meta_path = "/data/home/jiaxi/home/stone-cls/dataset/assets/stone/"
        self.file_path = f"{self.meta_path}{self.task}/{dataset_type}.txt"
        self.img_preprocess = preprocess
        if prepared == False:
            self.train_test_gen()
        else:
            if self.task == "general":
                self.class_map = {"变质岩": "metamorphic rock", "沉积岩": "sedimentary rock", "火成岩": "igneous rock"}
                self.class_id_map = {"metamorphic rock": 0, "sedimentary rock": 1, "igneous rock": 2}
                with open(self.file_path, "r") as f:
                    for line in f:
                        file_path, label = line.strip().split("\t")
                        self.samples.append(file_path)
                        self.labels.append(self.class_map[label])
    
    def get_path(self):
        return base_path + "nimrf.net.cn/"
    
    def extract_distribution(self):
        res_count = {}
        try:
            directories = get_dirs(self.path)
            for directory in directories:
                files = os.listdir(self.path + directory)
                types = [file.split("_")[0] for file in files]
                count_types = dict(Counter(types))
                sorted_count_types = sorted(count_types.items(), key=lambda x: x[1], reverse=True)
                res_count[directory] = dict(sorted_count_types)
            return res_count
        except FileNotFoundError:
            print("Path not found: " + self.path)
            return
        except PermissionError:
            print("Permission denied: " + self.path)
            return
    
    def train_test_gen(self):
        train_list = []
        val_list = []
        test_list = []
        zero_shot_list = []
        assets_path = self.meta_path
        try:
            directories = get_dirs(self.path)
            if self.task == "general":
                assets_path += "general/"
                for directory in directories:
                    files = get_files(self.path + directory + "/")
                    random.shuffle(files)
                    train_size = int(len(files) * 0.8)
                    val_size = int(len(files) * 0.1)
                    for f in files[:train_size]:
                        train_list.append((self.path + directory + "/" + f, directory))
                    for f in files[train_size:train_size + val_size]:
                        val_list.append((self.path + directory + "/" + f, directory))
                    for f in files[train_size + val_size:]:
                        test_list.append((self.path + directory + "/" + f, directory))

            elif self.task == "specific":
                assets_path += "specific/"
                for directory in directories:
                    files = get_files(self.path + directory + "/")
                    label2files = defaultdict(list)

                    for file in files:
                        stone_label = file.split("_")[0]
                        label2files[stone_label].append(file)
                    
                    for stone_label, file_list in label2files.items():
                        if len(file_list) < 10:
                            for f in file_list:
                                zero_shot_list.append((self.path + directory + "/" + f, stone_label))
                        else:
                            test_subset = random.sample(file_list, 5)
                            for f in file_list:
                                if f in test_subset:
                                    test_list.append((self.path + directory + "/" + f, stone_label))
                                else:
                                    train_list.append((self.path + directory + "/" + f, stone_label))

            with open(f"{assets_path}train.txt", "w") as f:
                for item in train_list:
                    f.write(item[0] + "\t" + item[1] + "\n")
            if val_list:
                with open(f"{assets_path}val.txt", "w") as f:
                    for item in val_list:
                        f.write(item[0] + "\t" + item[1] + "\n")
            with open(f"{assets_path}test.txt", "w") as f:
                for item in test_list:
                    f.write(item[0] + "\t" + item[1] + "\n")
            if zero_shot_list:
                with open(f"{assets_path}zero_shot.txt", "w") as f:
                    for item in zero_shot_list:
                        f.write(item[0] + "\t" + item[1] + "\n")
        except FileNotFoundError as e:
            print("Path not found: " + str(e))
            return
        except PermissionError as e:
            print("Permission denied: " + str(e))
            return
        
    def __getitem__(self, index):
        img_path = self.samples[index]
        label = self.labels[index]
        try:
            image = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            print(f"Error opening image {img_path}. Skipping.")
            return None, None

        image = self.img_preprocess(image)
        return image, label