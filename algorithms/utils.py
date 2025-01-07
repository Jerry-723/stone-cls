import torch
import clip
import itertools

device = "cuda" if torch.cuda.is_available() else "cpu"

def clip_features(model, dataloader):
    all_images = []
    all_labels = []
    all_image_features = []
    all_label_features = []
    with torch.no_grad():
        for images, labels in dataloader:
            all_images.append(images)
            all_labels.append(labels)

            image_features = model.encode_image(images.to(device))
            all_image_features.append(image_features)

            tokens = torch.cat([clip.tokenize(f'a photo of {label}') for label in labels])
            label_features = model.encode_text(tokens.to(device))
            all_label_features.append(label_features)
    return torch.cat(all_images), list(itertools.chain(*all_labels)), torch.cat(all_image_features), torch.cat(all_label_features)