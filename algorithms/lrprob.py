from .utils import *
from common import get_dataloader
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# def get_features(model, name, preprocess, task, dataset_type):
#     all_features = []
#     all_labels = []
#     dataloader = get_dataloader(name, preprocess, task, dataset_type)

#     with torch.no_grad():
#         for images, labels in tqdm(dataloader):
#             features = model.encoder_image(images.to(device))
#             tokens = 
