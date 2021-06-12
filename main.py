import os
import torch
import torch.nn.functional as F
import numpy as np
import configurations
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CatDog
from efficientnet_pytorch import EfficientNet


def save_feature_vectors(model, loader, output_size=(1,1), file="trainb7"):
    model.eval()
    images, labels = [], []

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(configurations.device)

        with torch.no_grad():
            features = model.extract_features(x)
            features = F.adaptive_avg_pool2d(features, output_size=output_size)
        images.append(features.reshape(x.shape[0], -1).detach().cpu().numpy())
        labels.append(y.numpy())

    np.save(f"X_{file}.npy", np.concatenate(images, axis=0))
    np.save(f"y_{file}.npy", np.concatenate(labels, axis=0))
    model.train()


def main():
    model = EfficientNet.from_pretrained("efficientnet-b7")
    train_dataset = CatDog(root="train/", transform=configurations.basic_transform)
    test_dataset = CatDog(root="test1/", transform=configurations.basic_transform)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=configurations.BATCH_SIZE,
        num_workers=configurations.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=configurations.BATCH_SIZE,
        num_workers=configurations.NUM_WORKERS,
    )
    model = model.to(configurations.device)
    save_feature_vectors(model, train_loader, output_size=(1, 1), file="train_b7")
    save_feature_vectors(model, test_loader, output_size=(1, 1), file="test_b7")


if __name__ == "__main__":
    main()