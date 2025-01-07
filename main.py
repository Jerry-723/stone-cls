import clip
from algorithms import lrprob, clip_features
from common import get_dataloader, calculate_metrics, seedeverything
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["lrprob", "zeroshot"], default="lrprob")
    parser.add_argument("--name", type=str, choices=["stone", "meteorite"], default="stone")
    parser.add_argument("--task", type=str, choices=["general", "specific"], default="general")
    parser.add_argument("--backbone", type=str, choices=["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"], default="RN50")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seedeverything(42)
    if args.method == "lrprob":
        model, preprocess = clip.load(args.backbone, device=device)
        print(f"Model already load: {args.backbone}")

        train_dataloader = get_dataloader(args.name, preprocess, args.task, "train")
        test_dataloader = get_dataloader(args.name, preprocess, args.task, "test")

        print("Extracting training features...")
        train_images, train_labels, train_image_features, train_label_features = clip_features(model, train_dataloader)
        print("Extracting testing features...")
        test_images, test_labels, test_image_features, test_label_features = clip_features(model, test_dataloader)

        print("Training model...")
        classifier = lrprob.train(train_image_features.cpu().numpy(), train_labels)
        print("Inferencing...")
        preds = lrprob.get_pred(classifier, test_image_features.cpu().numpy())
        acc, f1 = calculate_metrics(preds, test_labels)
        print(f"Accuracy: {acc}, F1: {f1}")

