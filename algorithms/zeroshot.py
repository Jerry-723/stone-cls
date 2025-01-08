from .utils import *
import numpy as np

def get_pred(model, image, classes):
    text_inputs = torch.cat([clip.tokenize(f'a photo of {c}') for c in classes]).to(device)
    logits_per_image, _ = model(image, text_inputs)
    probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
    max_index = np.argmax(probs)
    return classes[max_index]