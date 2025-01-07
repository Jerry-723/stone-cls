import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_pairs(dataloader):
    all_images = []
    all_labels = []
    for images, labels in dataloader:
        all_images.append(images)
        all_labels.append(labels)
    return torch.cat(all_images), all_labels

def clip_image_features(model, dataloader):
    image_features = []
    images, _ = get_pairs(dataloader)
    with torch.no_grad():
        for images, _ in dataloader:
            features = model.encoder_image(images.to(device))
            image_features.append(features)
            
