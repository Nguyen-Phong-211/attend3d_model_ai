# main.py
import os
import torch
from config import config
from dataset import get_dataloaders
from model import create_model
from trainer import Trainer

def main():
    print("3D Face Recognition - Training")
    print("Device:", config.DEVICE)
    if not os.path.exists(config.DATA_ROOT):
        raise RuntimeError(f"DATA_ROOT does not exist: {config.DATA_ROOT}")

    train_loader, val_loader, num_classes = get_dataloaders(config)
    print("Num classes:", num_classes)

    model = create_model(num_classes, config)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))

    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader, num_classes)

if __name__ == "__main__":
    main()