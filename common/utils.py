import torch
from sklearn.metrics import accuracy_score, f1_score
import random
import numpy as np
import torch.backends.cudnn as cudnn
import os

def skip_broken_collate_fn(batch):
    valid_samples = [(img, lbl) for img, lbl in batch if img is not None and lbl is not None]
    
    if not valid_samples:
        return None, None
    
    images, labels = zip(*valid_samples)
    images = torch.stack(images, dim=0)
    labels = list(labels)

    return images, labels

def calculate_metrics(preds, labels):
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return accuracy, f1

def seedeverything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)