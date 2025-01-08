import clip
from algorithms import lrprob, zeroshot, clip_features
from common import get_dataloader, calculate_metrics, seedeverything
from models import get_model
import torch
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["clip"], default="clip")
    parser.add_argument("--method", type=str, choices=["lrprob", "zeroshot"], default="lrprob")
    parser.add_argument("--name", type=str, choices=["stone", "meteorite"], default="stone")
    parser.add_argument("--task", type=str, choices=["general", "specific"], default="general")
    parser.add_argument("--backbone", type=str, choices=["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"], default="RN50")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seedeverything(42)
    preds = []

    if args.model == "clip":
        model, preprocess = get_model(args.model, args.backbone, device=device)
        print(f"Loading clip model with backbone: {args.backbone}")
        
    if args.method == "lrprob":
        print("Performing linear probing...")
        train_dataloader = get_dataloader(args.name, preprocess, args.task, "train")
        test_dataloader = get_dataloader(args.name, preprocess, args.task, "test")

        print("Extracting training features...")
        train_images, train_labels, train_image_features, train_label_features = clip_features(model, train_dataloader)
        print("Extracting testing features...")
        test_images, test_labels, test_image_features, test_label_features = clip_features(model, test_dataloader)

        print("Training model...")
        classifier = lrprob.train(train_image_features.cpu().numpy(), train_labels)
        print("Inferencing...")
        preds = lrprob.get_preds(classifier, test_image_features.cpu().numpy())

    elif args.method == "zeroshot":
        print("Performing zero-shot classification...")
        test_dataloader = get_dataloader(args.name, preprocess, args.task, "test")
        print("Extracting testing features...")
        test_images, test_labels, test_image_features, test_label_features = clip_features(model, test_dataloader)
        classes = list(set(test_labels))

        print("Inferencing...")
        for image in tqdm(test_images):
            image = image.to(device).unsqueeze(0)
            pred = zeroshot.get_pred(model, image, classes)
            preds.append(pred)

    acc, f1 = calculate_metrics(preds, test_labels)
    print(f"Accuracy: {acc}, F1: {f1}")