from torch.utils.data import DataLoader
from dataset import get_dataset

def get_dataloader(name, preprocess, task, dataset_type, batch_size=128, shuffle=True, num_workers=4):
    dataset = get_dataset(name, preprocess, task, dataset_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader