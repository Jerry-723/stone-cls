from .utils import *
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class linearHeader(nn.Module):
    def __init__(self, vision_model, num_classes):
        super(linearHeader, self).__init__()
        self.vision_model = vision_model
        self.classifier = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        features = self.vision_model(x)
        logits = self.classifier(features)
        return logits

def train(vision_model, n_classes, dataloader, maps, epochs=10, lr=1e-4):
    model = linearHeader(vision_model, n_classes).to(device)
    for param in model.vision_model.parameters():
        param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.float()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in tqdm(dataloader, desc = f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device).float()
            labels = list(map(lambda x: maps[x], labels))
            labels = torch.tensor(labels).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

    return model

def get_pred(model, dataloader):
    preds = []
    model.float()
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device).float()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
    return preds